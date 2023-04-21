import numpy as np
import torch
import torch.nn as nn
from kornia.geometry import warp_affine
import torch.nn.functional as F
from pytorch3d.ops import (
    corresponding_points_alignment,
    knn_points,
    knn_gather
)

def resize_n_crop(image, M, dsize=112):
    # image: (b, c, h, w)
    # M   :  (b, 2, 3)
    return warp_affine(image, M, dsize=(dsize, dsize))

### perceptual level loss
class PerceptualLoss(nn.Module):
    def __init__(self, recog_net, input_size=112):
        super(PerceptualLoss, self).__init__()
        self.recog_net = recog_net
        self.preprocess = lambda x: 2 * x - 1
        self.input_size=input_size

    def forward(self, imageA, imageB, M):
        """
        1 - cosine distance
        Parameters:
            imageA       --torch.tensor (B, 3, H, W), range (0, 1) , RGB order
            imageB       --same as imageA
        """

        imageA = self.preprocess(resize_n_crop(imageA, M, self.input_size))
        imageB = self.preprocess(resize_n_crop(imageB, M, self.input_size))

        # freeze bn
        self.recog_net.eval()
        
        id_featureA = F.normalize(self.recog_net(imageA), dim=-1, p=2)
        id_featureB = F.normalize(self.recog_net(imageB), dim=-1, p=2)  
        cosine_d = torch.sum(id_featureA * id_featureB, dim=-1)
        # assert torch.sum((cosine_d > 1).float()) == 0
        return torch.sum(1 - cosine_d) / cosine_d.shape[0]        

def perceptual_loss(id_featureA, id_featureB):
    cosine_d = torch.sum(id_featureA * id_featureB, dim=-1)
        # assert torch.sum((cosine_d > 1).float()) == 0
    return torch.sum(1 - cosine_d) / cosine_d.shape[0]  

### image level loss
def photo_loss(imageA, imageB, mask, eps=1e-6):
    """
    l2 norm (with sqrt, to ensure backward stabililty, use eps, otherwise Nan may occur)
    Parameters:
        imageA       --torch.tensor (B, 3, H, W), range (0, 1), RGB order 
        imageB       --same as imageA
    """
    loss = torch.sqrt(eps + torch.sum((imageA - imageB) ** 2, dim=1, keepdims=True)) * mask
    loss = torch.sum(loss) / torch.max(torch.sum(mask), torch.tensor(1.0).to(mask.device))
    return loss

def landmark_loss(predict_lm, gt_lm, weight=None):
    """
    weighted mse loss
    Parameters:
        predict_lm    --torch.tensor (B, 68, 2)
        gt_lm         --torch.tensor (B, 68, 2)
        weight        --numpy.array (1, 68)
    """
    if not weight:
        weight = np.ones([68])
        weight[28:31] = 20
        weight[-8:] = 20
        weight = np.expand_dims(weight, 0)
        weight = torch.tensor(weight).to(predict_lm.device)
    loss = torch.sum((predict_lm - gt_lm)**2, dim=-1) * weight
    loss = torch.sum(loss) / (predict_lm.shape[0] * predict_lm.shape[1])
    return loss


### regulization
def reg_loss(coeffs_dict, opt=None):
    """
    l2 norm without the sqrt, from yu's implementation (mse)
    tf.nn.l2_loss https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    Parameters:
        coeffs_dict     -- a  dict of torch.tensors , keys: id, exp, tex, angle, gamma, trans

    """
    # coefficient regularization to ensure plausible 3d faces
    if opt:
        w_id, w_exp, w_tex = opt.w_id, opt.w_exp, opt.w_tex
    else:
        w_id, w_exp, w_tex = 1, 1, 1, 1
    creg_loss = w_id * torch.sum(coeffs_dict['id'] ** 2) +  \
           w_exp * torch.sum(coeffs_dict['exp'] ** 2) + \
           w_tex * torch.sum(coeffs_dict['tex'] ** 2)
    creg_loss = creg_loss / coeffs_dict['id'].shape[0]

    # gamma regularization to ensure a nearly-monochromatic light
    gamma = coeffs_dict['gamma'].reshape([-1, 3, 9])
    gamma_mean = torch.mean(gamma, dim=1, keepdims=True)
    gamma_loss = torch.mean((gamma - gamma_mean) ** 2)

    return creg_loss, gamma_loss

def reflectance_loss(texture, mask):
    """
    minimize texture variance (mse), albedo regularization to ensure an uniform skin albedo
    Parameters:
        texture       --torch.tensor, (B, N, 3)
        mask          --torch.tensor, (N), 1 or 0

    """
    mask = mask.reshape([1, mask.shape[0], 1])
    texture_mean = torch.sum(mask * texture, dim=1, keepdims=True) / torch.sum(mask)
    loss = torch.sum(((texture - texture_mean) * mask)**2) / (texture.shape[0] * torch.sum(mask))
    return loss


def lm_3d_loss(pred_lm_3d, gt_lm_3d, mask):
    loss = torch.abs(pred_lm_3d - gt_lm_3d)[mask, :]
    loss = torch.mean(loss)
    return loss


def nicp_loss(template_verts, target_verts, K=5, return_knn=False):
    """

    Args:
        template_verts: torch.tensor (1, m, 3)
        target_verts: torch.tensor (1, n, 3)

    Returns:

    """
    knn = knn_points(template_verts, target_verts, K=K)  # (1, m, k)
    close_points = knn_gather(target_verts, knn.idx)  # (1, m, k, 3)
    close_points_center = torch.mean(close_points, dim=2, keepdim=False)  # (1, m, 3)
    dist = torch.sum(torch.abs(template_verts - close_points_center), dim=2, keepdim=False)
    loss = torch.mean(dist)
    if return_knn:
        return knn
    return loss


class TVLoss(nn.Module):
    # for [N,C,H,W]
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TVLoss_std(nn.Module):
    # for [N,C,H,W]
    def __init__(self, TVLoss_weight=1):
        super(TVLoss_std, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2)
        h_tv = ((h_tv - torch.mean(h_tv)) ** 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2)
        w_tv = ((w_tv - torch.mean(w_tv)) ** 2).sum()
        return self.TVLoss_weight * 2 * (h_tv + w_tv) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def contour_aware_loss(verts, left_points, right_points, width=224):
    verts_int = torch.ceil(verts).long().clamp(0, width - 1)  # (B, n, 2)
    batch_size = verts_int.shape[0]
    verts_y = width - 1 - verts_int[:, :, 1]  # (B, n)
    indices = torch.arange(0, batch_size).long().to(verts_y.device)[:, None].repeat(1, verts_y.shape[1])  # (B, n)
    indices_verts_y = torch.stack([indices, verts_y], dim=2).reshape(-1, 2)  # (B * n, 2)

    verts_left = left_points[indices_verts_y[:, 0], indices_verts_y[:, 1]].reshape(batch_size, -1).float()  # (B, n)
    verts_right = right_points[indices_verts_y[:, 0], indices_verts_y[:, 1]].reshape(batch_size, -1).float()  # (B, n)
    verts_x = verts[:, :, 0]  # (B, n)
    dist = (verts_left - verts_x) / width * (verts_right - verts_x) / width  # (B, n)
    dist /= torch.max(torch.abs((verts_left - verts_x) / width), torch.abs((verts_right - verts_x) / width))  # (B, n)
    edge_inds = torch.where(dist > 0)
    dist += 0.01
    dist = torch.nn.functional.relu(dist).clone()
    dist -= 0.01
    dist = torch.abs(dist)  # (B, n)
    loss = torch.mean(dist)  # (1)
    return loss, edge_inds


class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()

    def forward(self, x):
        batch_size, slice_num = x.size()[:2]
        z_x = x.size()[2]
        h_x = x.size()[3]
        w_x = x.size()[4]
        count_z = self._tensor_size(x[:, :, 1:, :, :])
        count_h = self._tensor_size(x[:, :, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, :, 1:])
        z_tv = torch.pow((x[:, :, 1:, :, :] - x[:, :, :z_x - 1, :, :]), 2).sum()
        h_tv = torch.pow((x[:, :, :, 1:, :] - x[:, :, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, :, 1:] - x[:, :, :, :, :w_x - 1]), 2).sum()
        return 2 * (z_tv / count_z + h_tv / count_h + w_tv / count_w) / (batch_size * slice_num)

    def _tensor_size(self, t):
        return t.size()[2] * t.size()[3] * t.size()[4]


class LaplacianLoss_L1(nn.Module):
    def __init__(self):
        super(LaplacianLoss_L1, self).__init__()

    def forward(self, x):
        batch_size, slice_num = x.size()[:2]
        z_x = x.size()[2]
        h_x = x.size()[3]
        w_x = x.size()[4]
        count_z = self._tensor_size(x[:, :, 1:, :, :])
        count_h = self._tensor_size(x[:, :, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, :, 1:])
        z_tv = torch.abs((x[:, :, 1:, :, :] - x[:, :, :z_x - 1, :, :])).sum()
        h_tv = torch.abs((x[:, :, :, 1:, :] - x[:, :, :, :h_x - 1, :])).sum()
        w_tv = torch.abs((x[:, :, :, :, 1:] - x[:, :, :, :, :w_x - 1])).sum()
        return 2 * (z_tv / count_z + h_tv / count_h + w_tv / count_w) / (batch_size * slice_num)

    def _tensor_size(self, t):
        return t.size()[2] * t.size()[3] * t.size()[4]


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)



