import cv2
import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .bfm import ParametricFaceModel
from .losses import perceptual_loss, photo_loss, reg_loss, reflectance_loss, landmark_loss, TVLoss, TVLoss_std, contour_aware_loss
from util import util_
from util.nv_diffrast import MeshRenderer
import os
from util.util_ import read_obj, write_obj2, viz_flow, split_vis, estimate_normals, write_video, crop_mesh
import time
from models.de_retouching_module import DeRetouchingModule
from pix2pix.pix2pix_model import Pix2PixModel
from pix2pix.pix2pix_options import Pix2PixOptions
import imageio


class FaceReconModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        # net structure and parameters
        parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
        parser.add_argument('--init_path', type=str, default='checkpoints/init_model/resnet50-0676ba61.pth')
        parser.add_argument('--use_last_fc', type=util_.str2bool, nargs='?', const=True, default=False, help='zero initialize the last fc')
        parser.add_argument('--bfm_folder', type=str, default='assets/3dmm_assets/BFM')
        parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

        # renderer parameters
        parser.add_argument('--focal', type=float, default=1015.)
        parser.add_argument('--center', type=float, default=112.)
        parser.add_argument('--camera_d', type=float, default=10.)
        parser.add_argument('--z_near', type=float, default=5.)
        parser.add_argument('--z_far', type=float, default=15.)

        is_train = True
        if is_train:
            # training parameters
            parser.add_argument('--net_recog', type=str, default='r50', choices=['r18', 'r43', 'r50'], help='face recog network structure')
            parser.add_argument('--net_recog_path', type=str, default='../pretrained_models/recog_model/ms1mv3_arcface_r50_fp16/backbone.pth')
            parser.add_argument('--use_crop_face', type=util_.str2bool, nargs='?', const=True, default=False, help='use crop mask for photo loss')
            parser.add_argument('--use_predef_M', type=util_.str2bool, nargs='?', const=True, default=False, help='use predefined M for predicted face')

            
            # augmentation parameters
            parser.add_argument('--shift_pixs', type=float, default=10., help='shift pixels')
            parser.add_argument('--scale_delta', type=float, default=0.1, help='delta scale factor')
            parser.add_argument('--rot_angle', type=float, default=10., help='rot angles, degree')

            # loss weights
            parser.add_argument('--w_feat', type=float, default=0.2, help='weight for feat loss')
            parser.add_argument('--w_color', type=float, default=1.92, help='weight for loss loss')
            parser.add_argument('--w_reg', type=float, default=3.0e-4, help='weight for reg loss')
            parser.add_argument('--w_id', type=float, default=1.0, help='weight for id_reg loss')
            parser.add_argument('--w_exp', type=float, default=0.8, help='weight for exp_reg loss')
            parser.add_argument('--w_tex', type=float, default=1.7e-2, help='weight for tex_reg loss')
            parser.add_argument('--w_gamma', type=float, default=10.0, help='weight for gamma loss')
            parser.add_argument('--w_lm', type=float, default=1.6e-3, help='weight for lm loss')
            parser.add_argument('--w_reflc', type=float, default=5.0, help='weight for reflc loss')
            parser.add_argument('--w_contour', type=float, default=20.0, help='weight for contour-aware loss')
            parser.add_argument('--w_smooth', type=float, default=5.0e3, help='weight for total variation loss')
            parser.add_argument('--w_dis_reg', type=float, default=10.0, help='weight for displacement map regularization loss')
            parser.add_argument('--w_adv', type=float, default=1.0, help='weight for adversarial losses')

        opt, _ = parser.parse_known_args()
        parser.set_defaults(
                focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
            )
        if is_train:
            parser.set_defaults(
                use_crop_face=True, use_predef_M=False
            )
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel

        self.opt = opt
        self.visual_names = ['output_vis']
        self.model_names = ['net_recon', 'mid_net', 'high_net']
        # self.model_names = ['net_recon']
        self.parallel_names = self.model_names + ['renderer', 'renderer_high_res']

        self.net_recon = networks.define_net_recon(
            net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=None
        )

        self.facemodel_front = ParametricFaceModel(
            bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            is_train=True, default_name='BFM_model_front.mat'
        )

        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        self.renderer_high_res = MeshRenderer(
            rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(2 * opt.center)
        )

        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(2 * opt.center)
        )

        self.bfm_UVs = np.load('assets/3dmm_assets/template_mesh/bfm_uvs2.npy')
        self.bfm_UVs = torch.from_numpy(self.bfm_UVs).to(torch.device('cuda')).float()

        de_retouching_path = 'assets/pretrained_models/de-retouching.pth'
        self.de_retouching_module = DeRetouchingModule(de_retouching_path)

        self.mid_opt = Pix2PixOptions()
        self.mid_opt.input_nc = 6
        self.mid_opt.output_nc = 3
        self.mid_opt.name = 'mid_net'
        # self.mid_opt.netG = 'unet_32'
        self.mid_net = Pix2PixModel(self.mid_opt).netG
        self.mid_net = self.mid_net.to(self.device)

        self.high_opt = Pix2PixOptions()
        self.high_opt.input_nc = 9
        self.high_opt.output_nc = 1
        self.high_opt.name = 'high_net'
        # self.high_opt.netG = 'unet_128'
        self.high_net = Pix2PixModel(self.high_opt).netG
        self.high_net = self.high_net.to(self.device)

        # two trainable parameters to stabilize the training process
        self.alpha = (torch.ones(1, dtype=torch.float32) * 0.01).to(self.device)
        self.alpha.requires_grad = True
        self.beta = (torch.ones(1, dtype=torch.float32) * 0.01).to(self.device)
        self.beta.requires_grad = True

        self.loss_names = ['all', 'color_high', 'color_mid', 'lm', 'smooth', 'smooth_std', 'reg_displacement',
                           'smooth_displacement', 'smooth_displacement_std', 'points_horizontal']

        # self.net_recog = networks.define_net_recog(
        #     net_recog=opt.net_recog, pretrained_path=opt.net_recog_path
        #     ).cuda()
        # loss func name: (compute_%s_loss) % loss_name
        self.compute_feat_loss = perceptual_loss
        self.comupte_color_loss = photo_loss
        self.compute_lm_loss = landmark_loss
        self.compute_reg_loss = reg_loss
        self.compute_reflc_loss = reflectance_loss

        if opt.isTrain:
            train_parameters = list(self.mid_net.parameters()) + list(self.high_net.parameters()) + [self.alpha, self.beta]
            self.optimizer = torch.optim.Adam(train_parameters, lr=opt.lr)
            self.optimizers = [self.optimizer]

    def set_render(self, opt,  image_res):
        fov = 2 * np.arctan(self.opt.center / self.opt.focal) * 180 / np.pi
        if image_res is None:
            image_res = int(2 * opt.center)

        self.renderer_high_res = MeshRenderer(
            rasterize_fov=fov, znear=self.opt.z_near, zfar=self.opt.z_far, rasterize_size=image_res)

    def set_input_base(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['imgs'].to(self.device)
        self.input_img_hd = input['imgs_hd'].to(self.device) if 'imgs_hd' in input else None

        self.atten_mask = input['msks'].to(self.device) if 'msks' in input else None
        self.gt_lm = input['lms'].to(self.device)  if 'lms' in input else None
        self.gt_lm_hd = input['lms_hd'].to(self.device) if 'lms_hd' in input else None
        self.trans_m = input['M'].to(self.device) if 'M' in input else None
        self.image_paths = input['im_paths'] if 'im_paths' in input else None
        self.img_name = input['img_name'] if 'img_name' in input else None
        self.face_mask = input['face_mask'].to(self.device) if 'face_mask' in input else None
        self.head_mask = input['head_mask'].to(self.device) if 'head_mask' in input else None
        self.gt_normals = input['normals'].to(self.device) if 'normals' in input else None
        self.input_img_coeff = input['imgs_coeff'].to(self.device) if 'imgs_coeff' in input else None
        self.gt_lm_coeff = input['lms_coeff'].to(self.device) if 'lms_coeff' in input else None

    def set_input_hrn(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['input_img'].to(self.device)
        self.input_img_for_tex = input['input_img_for_tex'].to(self.device)
        self.input_img_hd = input['input_img_hd'].to(self.device)
        self.face_mask = input['face_mask'].to(self.device)
        self.gt_lm = input['gt_lm'].to(self.device)
        self.coeffs = input['coeffs'].to(self.device)
        self.position_map = input['position_map'].to(self.device)
        self.texture_map = input['texture_map'].to(self.device)
        self.tex_valid_mask = input['tex_valid_mask'].to(self.device)
        self.de_retouched_albedo_map = input['de_retouched_albedo_map'].to(self.device)

    def add_input_mvhrn(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img_list.append(input['input_img'].to(self.device))
        self.input_img_for_tex_list.append(input['input_img_for_tex'].to(self.device))
        self.input_img_hd_list.append(input['input_img_hd'].to(self.device))
        self.face_mask_list.append(input['face_mask'].to(self.device))
        self.gt_lm_list.append(input['gt_lm'].to(self.device))
        self.coeffs_list.append(input['coeffs'].to(self.device))
        self.position_map_list.append(input['position_map'].to(self.device))
        self.texture_map_list.append(input['texture_map'].to(self.device))
        self.tex_valid_mask_list.append(input['tex_valid_mask'].to(self.device))
        self.de_retouched_albedo_map_list.append(input['de_retouched_albedo_map'].to(self.device))
        self.deformation_map_list.append(input['deformation_map'].to(self.device))
        self.displacement_map_list.append(input['displacement_map'].to(self.device))

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        pass

    def init_mvhrn_input(self):
        self.input_img_list = []
        self.input_img_for_tex_list = []
        self.input_img_hd_list = []
        self.face_mask_list = []
        self.gt_lm_list = []
        self.coeffs_list = []
        self.position_map_list = []
        self.texture_map_list = []
        self.tex_valid_mask_list = []
        self.de_retouched_albedo_map_list = []
        self.deformation_map_list = []
        self.displacement_map_list = []

    def forward(self):
        output_coeff = self.net_recon(self.input_img)
        self.facemodel_front.to(self.device)
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm = \
            self.facemodel_front.compute_for_render(output_coeff)
        self.pred_mask, _, self.pred_face = self.renderer(self.pred_vertex, self.facemodel_front.face_buf, feat=self.pred_color)
        
        self.pred_coeffs_dict = self.facemodel_front.split_coeff(output_coeff)

    def predict_results_base(self):
        self.facemodel_front.to(self.device)
        # predict low-frequency coefficients
        with torch.no_grad():
            output_coeff = self.net_recon(self.input_img)

        # 3DMM
        face_vertex, face_albedo_map, face_color_map, landmark, face_vertex_noTrans, position_map = self.facemodel_front.compute_for_render(output_coeff)

        # get texture map
        texture_map = self.facemodel_front.get_texture_map(face_vertex, self.input_img_hd)

        # de-retouch
        texture_map_input_high = texture_map.permute(0, 3, 1, 2).detach()  # (1, 3, 256, 256)
        texture_map_input_high = (texture_map_input_high - 0.5) * 2
        de_retouched_face_albedo_map = self.de_retouching_module.run(face_albedo_map, texture_map_input_high)

        # get valid texture mask to deal with occlusion
        valid_mask = self.facemodel_front.get_texture_map(face_vertex, self.face_mask)  # (256, 256, 1)
        valid_mask = valid_mask.permute(0, 3, 1, 2).detach()  # (1, 1, 256, 256)

        # render
        pred_mask, _, pred_face = self.renderer.render_uv_texture(face_vertex, self.facemodel_front.face_buf,
                                                                     self.bfm_UVs.clone(), face_color_map)

        input_img_numpy = 255. * (self.input_img).detach().cpu().permute(0, 2, 3, 1).numpy()
        input_img_numpy = np.squeeze(input_img_numpy)
        output_vis = pred_face * pred_mask + (1 - pred_mask) * self.input_img
        output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
        output_vis_numpy_raw = np.squeeze(output_vis_numpy_raw)
        output_vis_numpy = np.concatenate((input_img_numpy, output_vis_numpy_raw), axis=-2)
        output_vis = np.squeeze(output_vis_numpy)
        output_vis = output_vis[..., ::-1]  # rgb->bgr
        output_face_mask = pred_mask.detach().cpu().permute(0, 2, 3, 1).squeeze().numpy() * 255.0
        output_vis = np.column_stack((output_vis, cv2.cvtColor(output_face_mask, cv2.COLOR_GRAY2BGR)))
        output_input_vis = output_vis[:, :224]
        output_pred_vis = output_vis[:, 224:448]

        input_img_hd = 255. * (self.input_img_hd).detach().cpu().permute(0, 2, 3, 1).numpy()[..., ::-1]
        input_img_hd = np.squeeze(input_img_hd)

        # # get texture map by differentiable rendering
        # UVs_tensor = self.bfm_UVs.clone()[None, ...]
        # target_img = self.input_img_hd
        # target_img = target_img.permute(0, 2, 3, 1)
        # face_buf = self.facemodel_front.face_buf
        # t1 = time.time()
        # pred_mask, _, pred_face, texture_map, texture_mask = self.renderer_high_res.pred_texture(face_vertex, face_buf, UVs_tensor, target_img, tex_size=512)
        # print('get texture map 2', time.time() - t1)

        # from camera space to world space
        recon_vertices = face_vertex  # [1, 35709, 3]
        recon_vertices[..., -1] = 10 - recon_vertices[..., -1]

        recon_shape = face_vertex_noTrans  # [1, 35709, 3]
        recon_shape[..., -1] = 10 - recon_shape[..., -1]

        tri = self.facemodel_front.face_buf

        # output
        output = {}
        output['coeffs'] = output_coeff.detach()  # [B, 257]
        output['vertices'] = recon_vertices.detach()  # [B, 35709, 3]
        output['vertices_noTrans'] = recon_shape.detach()  # [B, 35709, 3]
        output['triangles'] = tri.detach()  # [n_faces, 3], start from 0
        output['UVs'] = self.bfm_UVs.detach()  # [35709, 2]
        output['texture_map'] = texture_map.detach()  # [B, h, w, 3], RGB
        output['albedo_map'] = face_albedo_map.detach()  # [B, 3, h, w]
        output['color_map'] = face_color_map.detach()  # [B, 3, h, w]
        output['position_map'] = position_map.detach()  # [B, 3, h, w]

        output['input_face'] = output_input_vis
        output['pred_face'] = output_pred_vis
        output['input_face_hd'] = input_img_hd

        output['input_img'] = self.input_img
        output['input_img_hd'] = self.input_img_hd
        output['gt_lm'] = self.gt_lm
        output['face_mask'] = self.face_mask
        output['tex_valid_mask'] = valid_mask

        output['de_retouched_albedo_map'] = de_retouched_face_albedo_map.detach()  # [B, 3, h, w]

        return output

    def forward_hrn(self, visualize=False):
        self.facemodel_front.to(self.device)
        self.bfm_UVs = self.bfm_UVs

        # get valid mask to deal with occlusion
        if visualize:
            self.tex_valid_mask = self.smooth_valid_mask(self.tex_valid_mask)
        tex_valid_mask = self.tex_valid_mask  # (B, 1, 256, 256)
        tex_valid_mask_mid = torch.nn.functional.interpolate(tex_valid_mask, (64, 64), mode='bilinear')

        # mid frequency
        texture_map_input = self.texture_map.permute(0, 3, 1, 2).to(self.device)
        texture_map_input = (texture_map_input - 0.5) * 2
        texture_map_input_mid = torch.nn.functional.interpolate(texture_map_input, (64, 64), mode='bilinear')
        position_map_input_mid = torch.nn.functional.interpolate(self.position_map, (64, 64), mode='bilinear')
        input_mid = torch.cat([position_map_input_mid, texture_map_input_mid], dim=1)
        self.deformation_map = self.mid_net(input_mid) * 0.1 * self.alpha  # ori * 0.1 * self.alpha
        self.deformation_map = self.deformation_map * tex_valid_mask_mid
        self.deformation_map = self.deformation_map.permute(0, 2, 3, 1)

        # render mid frequency results
        self.pred_vertex, self.pred_color, self.pred_lm, self.verts_proj, self.face_albedo_map, face_shape_transformed, face_norm_roted, self.extra_results = \
            self.facemodel_front.compute_for_render_hierarchical_mid(self.coeffs, self.deformation_map, self.bfm_UVs, visualize=visualize, de_retouched_albedo_map=self.de_retouched_albedo_map)
        self.pred_mask, _, self.pred_face_mid = self.renderer.render_uv_texture(self.pred_vertex,
                                                                                   self.facemodel_front.face_buf,
                                                                                   self.bfm_UVs.clone(),
                                                                                   self.pred_color)
        self.deformation_map = self.deformation_map.permute(0, 3, 1, 2)

        # get re-aligned texture
        texture_map_input_high = self.facemodel_front.get_texture_map(self.pred_vertex, self.input_img_hd)  # (1, 256, 256, 3)
        texture_map_input_high = texture_map_input_high.permute(0, 3, 1, 2).detach()  # (1, 3, 256, 256)
        texture_map_input_high = (texture_map_input_high - 0.5) * 2

        # high frequency
        position_map_input_high = torch.nn.functional.interpolate(self.position_map, (256, 256), mode='bilinear')
        deformation_map_input_high = torch.nn.functional.interpolate(self.deformation_map, (256, 256), mode='bilinear')
        input_high = torch.cat([position_map_input_high, texture_map_input_high, deformation_map_input_high], dim=1)
        self.displacement_map = self.high_net(input_high) * 0.1 * self.beta  # ori * 0.1 * self.alpha
        self.displacement_map = self.displacement_map * tex_valid_mask

        # render high frequency results
        self.pred_color_high, self.extra_results = self.facemodel_front.compute_for_render_hierarchical_high(self.coeffs, self.displacement_map,
                                                                                         self.de_retouched_albedo_map, face_shape_transformed, face_norm_roted, extra_results=self.extra_results)
        _, _, self.pred_face_high = self.renderer.render_uv_texture(self.pred_vertex,
                                                                       self.facemodel_front.face_buf,
                                                                       self.bfm_UVs.clone(),
                                                                       self.pred_color_high)

        self.pred_coeffs_dict = self.facemodel_front.split_coeff(self.coeffs)

        if visualize:
            # high
            self.extra_results['pred_mask_high'] = self.pred_mask
            self.extra_results['pred_face_high_color'] = self.pred_face_high
            _, _, self.extra_results['pred_face_high_gray'] = self.renderer.render_uv_texture(self.pred_vertex,
                                                                                       self.facemodel_front.face_buf,
                                                                                       self.bfm_UVs.clone(),
                                                                                       self.extra_results['tex_high_gray'])

            # fit texture
            with torch.enable_grad():
                texture_offset = torch.zeros((1, 3, 256, 256), dtype=torch.float32).to(
                    self.device)
                texture_offset.requires_grad = True

                optim = torch.optim.Adam([texture_offset], lr=1e-2)

                n_iters = 100
                for i in range(n_iters):  # 500
                    pred_color_high = self.pred_color_high.detach() + texture_offset
                    _, _, pred_face_high = self.renderer.render_uv_texture(self.pred_vertex.detach(),
                                                                           self.facemodel_front.face_buf,
                                                                           self.bfm_UVs.clone(),
                                                                           pred_color_high)

                    loss_color_high = self.opt.w_color * self.comupte_color_loss(pred_face_high, self.input_img_for_tex, self.pred_mask.detach())
                    loss_smooth = TVLoss()(texture_offset) * 10  # 10000, .permute(0, 3, 1, 2)
                    loss_all = loss_color_high + loss_smooth
                    optim.zero_grad()
                    loss_all.backward()
                    optim.step()

                self.pred_color_high = (self.pred_color_high + texture_offset).detach()

            if 'tex_high_gray_list' in self.extra_results:
                self.extra_results['pred_face_high_gray_list'] = []
                self.extra_results['pred_face_high_color_list'] = []
                for i in range(len(self.extra_results['tex_high_gray_list'])):
                    _, _, pred_face_high_gray_i = self.renderer_high_res.render_uv_texture(self.extra_results['face_vertex_list'][i],
                                                                                                      self.facemodel_front.face_buf,
                                                                                                      self.bfm_UVs.clone(),
                                                                                                      self.extra_results['tex_high_gray_list'][i])
                    self.extra_results['pred_face_high_gray_list'].append(pred_face_high_gray_i)

                    _, _, pred_face_high_color_i = self.renderer_high_res.render_uv_texture(self.extra_results['face_vertex_list'][i],
                                                                                            self.facemodel_front.face_buf,
                                                                                            self.bfm_UVs.clone(),
                                                                                            self.pred_color_high)
                    self.extra_results['pred_face_high_color_list'].append(pred_face_high_color_i)

            # mid
            self.extra_results['pred_mask_mid'] = self.pred_mask
            self.extra_results['pred_face_mid_color'] = self.pred_face_mid
            _, _, self.extra_results['pred_face_mid_gray'] = self.renderer.render_uv_texture(self.pred_vertex,
                                                                                                 self.facemodel_front.face_buf,
                                                                                                 self.bfm_UVs.clone(),
                                                                                                 self.extra_results['tex_mid_gray'])

            # base
            self.extra_results['pred_mask_base'], _, self.extra_results['pred_face_base_color'] = self.renderer.render_uv_texture(self.extra_results['pred_vertex_base'],
                                                                                       self.facemodel_front.face_buf,
                                                                                       self.bfm_UVs.clone(),
                                                                                       self.extra_results['tex_base_color'])
            _, _, self.extra_results['pred_face_base_gray'] = self.renderer.render_uv_texture(self.extra_results['pred_vertex_base'],
                                                                             self.facemodel_front.face_buf,
                                                                             self.bfm_UVs.clone(),
                                                                             self.extra_results['tex_base_gray'])

    def forward_mvhrn(self, visualize=False):
        self.facemodel_front.to(self.device)

        # initialize representations
        self.n_views = len(self.coeffs_list)
        coeff_list = torch.stack(self.coeffs_list, dim=0).detach().clone()
        mean_id = torch.mean(coeff_list[:, :, :80], dim=0, keepdim=False)
        mean_id.requires_grad = True
        output_coeff_list = []
        for coeff in coeff_list:
            output_coeff = coeff
            output_coeff = self.facemodel_front.split_coeff(output_coeff)
            output_coeff['id'] = mean_id
            output_coeff['exp'].requires_grad = True
            output_coeff['tex'].requires_grad = True
            output_coeff['angle'].requires_grad = True
            output_coeff['gamma'].requires_grad = True
            output_coeff['trans'].requires_grad = True
            output_coeff_list.append(output_coeff)

        self.canonical_deformation_map = torch.stack(self.deformation_map_list, dim=0).detach()
        self.canonical_deformation_map = torch.mean(self.canonical_deformation_map, dim=0, keepdim=False)  # (1, 3, 64, 64)
        self.canonical_deformation_map.requires_grad = True

        for i in range(self.n_views):
            self.displacement_map_list[i] = self.displacement_map_list[i].detach()
            self.displacement_map_list[i].requires_grad = True

        # initialize optimizer
        optim = torch.optim.Adam(
            [self.canonical_deformation_map, mean_id] + self.displacement_map_list + [coeff['exp'] for coeff in output_coeff_list] +
            [coeff['tex'] for coeff in output_coeff_list] + [coeff['angle'] for coeff in output_coeff_list] +
            [coeff['gamma'] for coeff in output_coeff_list] + [coeff['trans'] for coeff in output_coeff_list],
            lr=1e-3)

        # fitting
        n_iters = 50  # switch to inference at the last iter
        for i in range(n_iters+1):  # 500
            if i < n_iters:
                cur_grad = torch.enable_grad
            else:
                cur_grad = torch.no_grad
            with cur_grad():
                loss_all = 0
                self.pred_vertex_list, self.pred_color_list, self.pred_lm_list, self.verts_proj_list, self.face_albedo_map_list = [], [], [], [], []
                self.pred_mask_list, self.pred_face_mid_list = [], []
                self.pred_color_high_list, self.pred_face_high_list = [], []
                self.extra_results_list = []
                for j in range(self.n_views):
                    # get valid mask to deal with occlusion
                    if i == n_iters - 1 and visualize:
                        self.tex_valid_mask_list[j] = self.smooth_valid_mask(self.tex_valid_mask_list[j])
                    tex_valid_mask = self.tex_valid_mask_list[j]  # (B, 1, 256, 256)
                    tex_valid_mask_mid = torch.nn.functional.interpolate(tex_valid_mask, (64, 64), mode='bilinear')

                    # mid frequency
                    self.deformation_map = self.canonical_deformation_map * tex_valid_mask_mid
                    self.deformation_map = self.deformation_map.permute(0, 2, 3, 1)

                    cur_visualize = visualize if i == n_iters else False
                    # render
                    self.pred_vertex, self.pred_color, self.pred_lm, self.verts_proj, self.face_albedo_map, face_shape_transformed, face_norm_roted, extra_results = \
                        self.facemodel_front.compute_for_render_hierarchical_mid(output_coeff_list[j], self.deformation_map,
                                                                                 self.bfm_UVs, visualize=cur_visualize,
                                                                                 de_retouched_albedo_map=self.de_retouched_albedo_map_list[j])
                    self.pred_mask, _, self.pred_face_mid = self.renderer.render_uv_texture(self.pred_vertex,
                                                                                            self.facemodel_front.face_buf,
                                                                                            self.bfm_UVs.clone(),
                                                                                            self.pred_color)
                    self.deformation_map = self.deformation_map.permute(0, 3, 1, 2)

                    # high frequency
                    self.displacement_map = self.displacement_map_list[j] * tex_valid_mask

                    # render
                    self.pred_color_high, extra_results = self.facemodel_front.compute_for_render_hierarchical_high(
                        output_coeff_list[j], self.displacement_map,
                        self.de_retouched_albedo_map_list[j], face_shape_transformed, face_norm_roted,
                        extra_results=extra_results)
                    _, _, self.pred_face_high = self.renderer.render_uv_texture(self.pred_vertex,
                                                                                self.facemodel_front.face_buf,
                                                                                self.bfm_UVs.clone(),
                                                                                self.pred_color_high)

                    # change value for computing loss
                    self.input_img = self.input_img_list[j]
                    self.gt_lm = self.gt_lm_list[j]
                    self.left_points = self.left_points_list[j]
                    self.right_points = self.right_points_list[j]
                    self.pred_coeffs_dict = self.facemodel_front.split_coeff(output_coeff_list[j])

                    # record results
                    self.pred_vertex_list.append(self.pred_vertex)
                    self.pred_color_list.append(self.pred_color)
                    self.pred_lm_list.append(self.pred_lm)
                    self.verts_proj_list.append(self.verts_proj)
                    self.face_albedo_map_list.append(self.face_albedo_map)
                    self.pred_mask_list.append(self.pred_mask)
                    self.pred_face_mid_list.append(self.pred_face_mid)
                    self.pred_color_high_list.append(self.pred_color_high)
                    self.pred_face_high_list.append(self.pred_face_high)
                    if extra_results is not None:
                        extra_results['pred_mask_high'] = self.pred_mask
                        extra_results['pred_face_high_color'] = self.pred_face_high
                        extra_results['pred_mask_mid'] = self.pred_mask
                        extra_results['pred_face_mid_color'] = self.pred_face_mid

                        self.extra_results_list.append(extra_results)

                    # compute losses
                    self.compute_losses_for_mvhrn()
                    loss_all += self.loss_all

                    print('{}: lm: {:.6f}, color_mid: {:.6f}, color_high: {:.6f}, deform_tv: {:.6f}, deform_tv_std: {:.6f}, points_horizontal: {:.6f}, displace_tv: {:.6f}, displace_tv_std: {:.6f}'
                          .format(i, self.loss_lm.item(), self.loss_color_mid.item(), self.loss_color_high.item(), self.loss_smooth.item(), self.loss_smooth_std.item(),
                                  self.loss_points_horizontal.item(), self.loss_smooth_displacement.item(), self.loss_smooth_displacement_std.item()))

                print('{}: total loss: {:.6f}'.format(i, loss_all.item()))

                if i < n_iters:
                    optim.zero_grad()
                    loss_all.backward()
                    optim.step()

        # # get valid mask to deal with occlusion
        # if visualize > 0:
        #     self.tex_valid_mask = self.smooth_valid_mask(self.tex_valid_mask)

        if visualize:
            for i in range(self.n_views):
                # high
                _, _, self.extra_results_list[i]['pred_face_high_gray'] = self.renderer.render_uv_texture(self.pred_vertex_list[i],
                                                                                           self.facemodel_front.face_buf,
                                                                                           self.bfm_UVs.clone(),
                                                                                           self.extra_results_list[i]['tex_high_gray'])

                if 'tex_high_gray_list' in self.extra_results_list[i]:
                    self.extra_results_list[i]['pred_face_high_gray_list'] = []
                    self.extra_results_list[i]['pred_face_high_color_list'] = []
                    for j in range(len(self.extra_results_list[i]['tex_high_gray_list'])):
                        _, _, pred_face_high_gray_j = self.renderer_high_res.render_uv_texture(self.extra_results_list[i]['face_vertex_list'][j],
                                                                                                          self.facemodel_front.face_buf,
                                                                                                          self.bfm_UVs.clone(),
                                                                                                          self.extra_results_list[i]['tex_high_gray_list'][j])
                        self.extra_results_list[i]['pred_face_high_gray_list'].append(pred_face_high_gray_j)

                        _, _, pred_face_high_color_j = self.renderer_high_res.render_uv_texture(self.extra_results_list[i]['face_vertex_list'][j],
                                                                                                self.facemodel_front.face_buf,
                                                                                                self.bfm_UVs.clone(),
                                                                                                self.pred_color_high_list[i])
                        self.extra_results_list[i]['pred_face_high_color_list'].append(pred_face_high_color_j)

                # mid
                _, _, self.extra_results_list[i]['pred_face_mid_gray'] = self.renderer.render_uv_texture(self.pred_vertex_list[i],
                                                                                                     self.facemodel_front.face_buf,
                                                                                                     self.bfm_UVs.clone(),
                                                                                                     self.extra_results_list[i]['tex_mid_gray'])

                # base
                self.extra_results_list[i]['pred_mask_base'], _, self.extra_results_list[i]['pred_face_base_color'] = self.renderer.render_uv_texture(self.extra_results_list[i]['pred_vertex_base'],
                                                                                           self.facemodel_front.face_buf,
                                                                                           self.bfm_UVs.clone(),
                                                                                           self.extra_results_list[i]['tex_base_color'])
                _, _, self.extra_results_list[i]['pred_face_base_gray'] = self.renderer.render_uv_texture(self.extra_results_list[i]['pred_vertex_base'],
                                                                                 self.facemodel_front.face_buf,
                                                                                 self.bfm_UVs.clone(),
                                                                                 self.extra_results_list[i]['tex_base_gray'])

    def get_edge_points_horizontal(self):
        left_points_list = []
        right_points_list = []
        for k in range(self.face_mask.shape[0]):
            left_points = []
            right_points = []
            for i in range(self.face_mask.shape[2]):
                inds = torch.where(self.face_mask[k, 0, i, :] > 0.5)  # 0.9
                if len(inds[0]) > 0:  # i > 112 and len(inds[0]) > 0
                    left_points.append(int(inds[0][0]) + 1)
                    right_points.append(int(inds[0][-1]))
                else:
                    left_points.append(0)
                    right_points.append(self.face_mask.shape[3] - 1)
            left_points_list.append(torch.tensor(left_points).long().to(self.device))
            right_points_list.append(torch.tensor(right_points).long().to(self.device))
        self.left_points = torch.stack(left_points_list, dim=0).long().to(self.device)
        self.right_points = torch.stack(right_points_list, dim=0).long().to(self.device)

    def get_edge_points_horizontal_list(self):
        self.left_points_list = []
        self.right_points_list = []
        n_views = len(self.input_img_list)
        for j in range(n_views):
            left_points = []
            right_points = []
            for i in range(self.face_mask_list[j].shape[2]):
                inds = torch.where(self.face_mask_list[j][0, 0, i, :] > 0.5)  # 0.9
                if len(inds[0]) > 0:  # i > 112 and len(inds[0]) > 0
                    left_points.append(int(inds[0][0]) + 1)
                    right_points.append(int(inds[0][-1]))
                else:
                    left_points.append(0)
                    right_points.append(self.face_mask_list[j].shape[3] - 1)
            left_points = torch.tensor(left_points).long().to(self.device)[None, ...]
            right_points = torch.tensor(right_points).long().to(self.device)[None, ...]
            self.left_points_list.append(left_points)
            self.right_points_list.append(right_points)

    def smooth_valid_mask(self, tex_valid_mask):
        """

        :param tex_valid_mask: torch.tensor, (B, 1, 256, 256), value: 0~1
        :return:
        """
        batch_size = tex_valid_mask.shape[0]
        tex_valid_mask_ = tex_valid_mask.detach().cpu().numpy()
        result_list = []
        for i in range(batch_size):
            mask = tex_valid_mask_[i, 0]
            mask = cv2.erode(mask, np.ones(shape=(3, 3), dtype=np.float32))
            # mask = cv2.dilate(mask, np.ones(shape=(7, 7), dtype=np.float32))
            # mask = cv2.erode(mask, np.ones(shape=(17, 17), dtype=np.float32))
            mask = cv2.blur(mask, (11, 11), 0)
            result_list.append(torch.from_numpy(mask)[None].float().to(tex_valid_mask.device))
        smoothed_mask = torch.stack(result_list, dim=0)
        return smoothed_mask

    def compute_losses_for_mvhrn(self):
        face_mask = self.pred_mask

        face_mask = face_mask.detach()
        self.loss_color_high = self.opt.w_color * self.comupte_color_loss(
            self.pred_face_high, self.input_img, face_mask)  # 1.0

        self.loss_color_mid = self.opt.w_color * self.comupte_color_loss(
            self.pred_face_mid, self.input_img, face_mask)  # 1.0

        loss_reg, loss_gamma = self.compute_reg_loss(self.pred_coeffs_dict, self.opt)
        self.loss_reg = self.opt.w_reg * loss_reg  # 1.0
        self.loss_gamma = self.opt.w_gamma * loss_gamma  # 1.0

        self.loss_lm = self.opt.w_lm * self.compute_lm_loss(self.pred_lm, self.gt_lm) * 0.1  # 0.1

        w_offset_tv = 1000
        w_offset_tvStd = 10000
        self.loss_smooth = TVLoss()(self.deformation_map) * w_offset_tv
        self.loss_smooth_std = TVLoss_std()(self.deformation_map) * w_offset_tvStd

        self.loss_reg_displacement = torch.mean(torch.abs(self.displacement_map)) * 1
        self.loss_smooth_displacement = TVLoss()(self.displacement_map) * 10000
        self.loss_smooth_displacement_std = TVLoss_std()(self.displacement_map) * 10000

        self.loss_points_horizontal, self.edge_points_inds = contour_aware_loss(self.verts_proj, self.left_points, self.right_points)
        self.loss_points_horizontal *= 20

        self.loss_all = self.loss_color_high + self.loss_color_mid + self.loss_lm + \
                        self.loss_smooth + self.loss_smooth_std + \
                        self.loss_reg_displacement + self.loss_smooth_displacement + self.loss_smooth_displacement_std + \
                        self.loss_points_horizontal + \
                        self.loss_reg + self.loss_gamma

    def compute_visuals_hrn(self):
        with torch.no_grad():

            input_img_vis = 255. * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
            output_vis_mid = self.pred_face_mid * self.pred_mask + (1 - self.pred_mask) * self.input_img
            output_vis_mid = 255. * output_vis_mid.detach().cpu().permute(0, 2, 3, 1).numpy()
            output_vis_high = self.pred_face_high * self.pred_mask + (1 - self.pred_mask) * self.input_img
            output_vis_high = 255. * output_vis_high.detach().cpu().permute(0, 2, 3, 1).numpy()

            deformation_map_vis = torch.nn.functional.interpolate(self.deformation_map, input_img_vis.shape[1:3], mode='bilinear').permute(0, 2, 3, 1)
            deformation_map_vis = (deformation_map_vis - deformation_map_vis.min()) / (deformation_map_vis.max() - deformation_map_vis.min())
            deformation_map_vis = 255. * deformation_map_vis.detach().cpu().numpy()

            displacement_map_vis = torch.nn.functional.interpolate(self.displacement_map, input_img_vis.shape[1:3], mode='bilinear').permute(0, 2, 3, 1)
            displacement_vis = (displacement_map_vis - displacement_map_vis.min()) / (displacement_map_vis.max() - displacement_map_vis.min())
            displacement_vis = 255. * displacement_vis.detach().cpu().numpy()
            displacement_vis = np.concatenate([displacement_vis, displacement_vis, displacement_vis], axis=-1)

            # displacement_vis2 = displacement_map_vis * 5 + 0.5
            # displacement_vis2 = 255. * displacement_vis2.detach().cpu().numpy()
            # displacement_vis2 = np.concatenate([displacement_vis2, displacement_vis2, displacement_vis2], axis=-1)

            face_albedo_map_vis = torch.nn.functional.interpolate(self.face_albedo_map, input_img_vis.shape[1:3], mode='bilinear').permute(0, 2, 3, 1)
            face_albedo_map_vis = 255. * face_albedo_map_vis.detach().cpu().numpy()

            de_retouched_face_albedo_map_vis = torch.nn.functional.interpolate(self.de_retouched_albedo_map, input_img_vis.shape[1:3], mode='bilinear').permute(0, 2, 3, 1)
            de_retouched_face_albedo_map_vis = 255. * de_retouched_face_albedo_map_vis.detach().cpu().numpy()

            face_mask_vis = torch.nn.functional.interpolate(self.tex_valid_mask, input_img_vis.shape[1:3], mode='bilinear').permute(0, 2, 3, 1)
            face_mask_vis = (face_mask_vis * 255.0).detach().cpu().numpy()
            face_mask_vis = np.concatenate([face_mask_vis, face_mask_vis, face_mask_vis], axis=-1)

            if self.extra_results is not None:
                pred_mask_base = self.extra_results['pred_mask_base']
                output_vis_base = self.extra_results['pred_face_base_color'] * pred_mask_base + (1 - pred_mask_base) * self.input_img
                output_vis_base = 255. * output_vis_base.detach().cpu().permute(0, 2, 3, 1).numpy()

                output_vis_base_gray = self.extra_results['pred_face_base_gray'] * pred_mask_base + (1 - pred_mask_base) * self.input_img
                output_vis_base_gray = 255. * output_vis_base_gray.detach().cpu().permute(0, 2, 3, 1).numpy()
                output_vis_mid_gray = self.extra_results['pred_face_mid_gray'] * self.pred_mask + (1 - self.pred_mask) * self.input_img
                output_vis_mid_gray = 255. * output_vis_mid_gray.detach().cpu().permute(0, 2, 3, 1).numpy()
                output_vis_high_gray = self.extra_results['pred_face_high_gray'] * self.pred_mask + (1 - self.pred_mask) * self.input_img
                output_vis_high_gray = 255. * output_vis_high_gray.detach().cpu().permute(0, 2, 3, 1).numpy()

                output_vis_numpy = np.concatenate((input_img_vis, output_vis_high,
                                                   output_vis_base_gray, output_vis_mid_gray, output_vis_high_gray,
                                                   deformation_map_vis, displacement_vis,
                                                   face_albedo_map_vis, de_retouched_face_albedo_map_vis, face_mask_vis), axis=-2)

            elif self.gt_lm is not None:
                gt_lm_numpy = self.gt_lm.cpu().numpy()
                pred_lm_numpy = self.pred_lm.detach().cpu().numpy()
                output_vis_high_lm = util_.draw_landmarks(output_vis_high, gt_lm_numpy, 'b')
                output_vis_high_lm = util_.draw_landmarks(output_vis_high_lm, pred_lm_numpy, 'r')

                output_vis_numpy = np.concatenate((input_img_vis,
                                                   output_vis_mid, output_vis_high, output_vis_high_lm, deformation_map_vis, displacement_vis), axis=-2)
            else:
                output_vis_numpy = np.concatenate((input_img_vis,
                                                   output_vis_mid, output_vis_high, deformation_map_vis, displacement_vis), axis=-2)

            self.output_vis = torch.tensor(
                output_vis_numpy / 255., dtype=torch.float32
            ).permute(0, 3, 1, 2).to(self.device)

    def save_results(self, out_dir, save_name='test'):
        self.compute_visuals_hrn()
        results = self.get_current_visuals()

        batch_size = results['output_vis'].shape[0]

        hrn_output_vis_batch = (255.0 * results['output_vis']).permute(0, 2, 3, 1).detach().cpu().numpy()[..., ::-1]

        vertices_batch = self.pred_vertex.detach()  # get reconstructed shape, [1, 35709, 3]
        vertices_batch[..., -1] = 10 - vertices_batch[..., -1]  # from camera space to world space
        vertices_batch = vertices_batch.cpu().numpy()

        # dense mesh
        dense_vertices_batch = self.extra_results['dense_mesh']['vertices']
        dense_vertices_batch = dense_vertices_batch.detach().cpu().numpy()
        dense_faces_batch = self.extra_results['dense_mesh']['faces'].detach().cpu().numpy()


        texture_map_batch = (255.0 * self.pred_color_high).permute(0, 2, 3, 1).detach().cpu().numpy()[..., ::-1]

        for i in range(batch_size):
            cv2.imwrite(os.path.join(out_dir, save_name + '_{}_hrn_output.jpg'.format(i)), hrn_output_vis_batch[i])
            # split_vis(os.path.join(out_dir, save_name + '_{}_hrn_output.jpg'.format(i)))

            # export mesh with mid frequency details
            texture_map = texture_map_batch[i]
            vertices = vertices_batch[i]
            normals = estimate_normals(vertices, self.facemodel_front.face_buf.cpu().numpy())
            face_mesh = {
                'vertices': vertices,
                'faces': self.facemodel_front.face_buf.cpu().numpy() + 1,
                'UVs': self.bfm_UVs.detach().cpu().numpy(),
                'faces_uv': self.facemodel_front.face_buf.cpu().numpy() + 1,
                'normals': normals,
                'texture_map': texture_map,
            }
            write_obj2(os.path.join(out_dir, save_name + '_{}_hrn_mid_mesh.obj'.format(i)), face_mesh)
            results['face_mesh'] = face_mesh

            # export mesh with mid and high frequency details
            dense_mesh = {
                'vertices': dense_vertices_batch[i],
                'faces': dense_faces_batch[i],
            }
            vertices_zero = dense_mesh['vertices'] == 0.0
            keep_inds = np.where((vertices_zero[:, 0] * vertices_zero[:, 1] * vertices_zero[:, 2]) == False)[0]
            dense_mesh, _ = crop_mesh(dense_mesh, keep_inds)  # remove the redundant vertices and faces
            write_obj2(os.path.join(out_dir, save_name + '_{}_hrn_high_mesh.obj'.format(i)), dense_mesh)

            pred_face_gray_list = []
            if 'pred_face_high_gray_list' in self.extra_results:
                for j in range(len(self.extra_results['pred_face_high_gray_list'])):
                    pred_face_high_gray_j = self.extra_results['pred_face_high_gray_list'][j][i, ...]
                    pred_face_high_gray_j = 255. * pred_face_high_gray_j.detach().cpu().permute(1, 2, 0).numpy()[..., ::-1]
                    pred_face_gray_list.append(pred_face_high_gray_j.clip(0, 255).astype(np.uint8))
                # video_save_path = os.path.join(out_dir, save_name + '_{}_rotate_gray.mp4'.format(i))
                # write_video(pred_face_gray_list, video_save_path, fps=30)

            pred_face_color_list = []
            if 'pred_face_high_color_list' in self.extra_results:
                for j in range(len(self.extra_results['pred_face_high_color_list'])):
                    pred_face_high_color_j = self.extra_results['pred_face_high_color_list'][j][i, ...]
                    pred_face_high_color_j = 255. * pred_face_high_color_j.detach().cpu().permute(1, 2, 0).numpy()[..., ::-1]
                    pred_face_color_list.append(pred_face_high_color_j.clip(0, 255).astype(np.uint8))
                # video_save_path = os.path.join(out_dir, save_name + '_{}_rotate_color.mp4'.format(i))
                # write_video(pred_face_color_list, video_save_path, fps=30)

            if len(pred_face_color_list) > 0:
                h = hrn_output_vis_batch[i].shape[0]
                static_image = np.concatenate([hrn_output_vis_batch[i][:, :h], hrn_output_vis_batch[i][:, h*2: h*5]], axis=1).clip(0, 255).astype(np.uint8)
                gif_images = []
                for j in range(len(pred_face_color_list)):
                    video_1_i = pred_face_gray_list[j]
                    video_1_i = cv2.resize(video_1_i, (h, h))
                    video_2_i = pred_face_color_list[j]
                    video_2_i = cv2.resize(video_2_i, (h, h))

                    cat_image = np.concatenate([static_image, video_1_i, video_2_i], axis=1)
                    gif_images.append(cat_image[..., ::-1])
                imageio.mimsave(os.path.join(out_dir, save_name + '_{}_hrn_output.gif'.format(i)), gif_images, fps=25)

        return results

    def save_results_mvhrn(self, out_dir, save_name='test'):
        results_list = []
        for i in range(self.n_views):
            # change value for computing visualization results
            self.input_img = self.input_img_list[i]
            self.pred_face_mid = self.pred_face_mid_list[i]
            self.pred_mask = self.pred_mask_list[i]
            self.pred_face_high = self.pred_face_high_list[i]
            self.deformation_map = self.canonical_deformation_map
            self.displacement_map = self.displacement_map_list[i]
            self.face_albedo_map = self.face_albedo_map_list[i]
            self.de_retouched_albedo_map = self.de_retouched_albedo_map_list[i]
            self.tex_valid_mask = self.tex_valid_mask_list[i]
            self.pred_vertex = self.pred_vertex_list[i]
            self.pred_color_high = self.pred_color_high_list[i]

            self.extra_results = self.extra_results_list[i]

            results = self.save_results(out_dir, '{}_view_{}'.format(save_name, i+1))
            results_list.append(results)

        return results_list



