from __future__ import print_function
import numpy as np
import torch
from PIL import Image
import os
import importlib
import argparse
from argparse import Namespace
import torchvision
import cv2
from torchvision import transforms
import time
from util.image_liquify import image_warp_grid1
import torch.nn.functional as F


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf


def genvalconf(train_opt, **kwargs):
    conf = Namespace(**vars(train_opt))
    attr_dict = train_opt.__dict__
    for key, value in attr_dict.items():
        if 'val' in key and key.split('_')[0] in attr_dict:
            setattr(conf, key.split('_')[0], value)

    for key in kwargs:
        setattr(conf, key, kwargs[key])

    return conf
        
def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array, range(0, 1)
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.clamp(0.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def correct_resize_label(t, size):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i, :1]
        one_np = np.transpose(one_t.numpy().astype(np.uint8), (1, 2, 0))
        one_np = one_np[:, :, 0]
        one_image = Image.fromarray(one_np).resize(size, Image.NEAREST)
        resized_t = torch.from_numpy(np.array(one_image)).long()
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


def correct_resize(t, size, mode=Image.BICUBIC):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i:i + 1]
        one_image = Image.fromarray(tensor2im(one_t)).resize(size, Image.BICUBIC)
        resized_t = torchvision.transforms.functional.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)

def draw_landmarks(img, landmark, color='r', step=2):
    """
    Return:
        img              -- numpy.array, (B, H, W, 3) img with landmark, RGB order, range (0, 255)
        

    Parameters:
        img              -- numpy.array, (B, H, W, 3), RGB order, range (0, 255)
        landmark         -- numpy.array, (B, 68, 2), y direction is opposite to v direction
        color            -- str, 'r' or 'b' (red or blue)
    """
    if color =='r':
        c = np.array([255., 0, 0])
    else:
        c = np.array([0, 0, 255.])

    _, H, W, _ = img.shape
    img, landmark = img.copy(), landmark.copy()
    landmark[..., 1] = H - 1 - landmark[..., 1]
    landmark = np.round(landmark).astype(np.int32)
    for i in range(landmark.shape[1]):
        x, y = landmark[:, i, 0], landmark[:, i, 1]
        for j in range(-step, step):
            for k in range(-step, step):
                u = np.clip(x + j, 0, W - 1)
                v = np.clip(y + k, 0, H - 1)
                for m in range(landmark.shape[0]):
                    img[m, v[m], u[m]] = c
    return img

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )[:, None]
    arr /= lens
    return arr


def estimate_normals(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)  # (35709, 3)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]  # (70789,3,3)
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    n[(n[:, 0] == 0) * (n[:, 1] == 0) * (n[:, 2] == 0)] = [0, 0, 1.0]
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    n = normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add thie normals through an indexed view of our (zeroed) per vertex normal array
    for i in range(3):
        for j in range(faces.shape[0]):
            norm[faces[j, i]] += n[j]

    norm[(norm[:, 0]==0) * (norm[:, 1]==0) * (norm[:, 2]==0)] = [0, 0, 1.0]


    # norm[faces[:, 0]] += n
    # norm[faces[:, 1]] += n
    # norm[faces[:, 2]] += n
    result = normalize_v3(norm)
    return result


def spread_flow(length, spread_ratio=2):
    Flow = np.zeros(shape=(length,length,2),dtype=np.float32)
    mag = np.zeros(shape=(length, length), dtype=np.float32)

    radius= length*0.5
    for h in range(Flow.shape[0]):
        for w in range(Flow.shape[1]):

            if (h-length//2) **2+(w -length//2) **2 <= radius**2:
                Flow[h, w, 0] = -(w - length//2)
                Flow[h, w, 1] = -(h - length // 2)

                distance = np.sqrt((w - length // 2) ** 2 + (h - length // 2) ** 2)

                if distance <= radius / 2.0:
                    mag[h, w] = 2.0 / radius * distance
                else:
                    mag[h, w] = - 2.0 / radius * distance + 2.0

    _, ang = cv2.cartToPolar(Flow[..., 0] + 1e-8, Flow[..., 1] + 1e-8)

    mag *= spread_ratio

    x, y = cv2.polarToCart(mag, ang, angleInDegrees=False)
    Flow = np.dstack((x, y))

    return Flow


def viz_flow(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr.astype(np.float)


def blend_textures(mid_img, left_img, right_img):
    out_texture = mid_img.copy()

    start = int(390 * mid_img.shape[1] / 1024)
    stop = int(420 * mid_img.shape[1] / 1024)
    width = stop - start

    fusion_mask = np.stack([np.linspace(0, 1, width)] * mid_img.shape[0], axis=0)[..., None]

    out_texture[:, :start] = left_img[:, :start]
    out_texture[:, start: stop] = out_texture[:, start: stop] * fusion_mask + left_img[:, start: stop] * (1 - fusion_mask)

    out_texture[:, -start:] = right_img[:, -start:]
    out_texture[:, -stop: -start] = out_texture[:, -stop: -start] * (1 - fusion_mask) + right_img[:, -stop: -start] * fusion_mask

    # print(fusion_mask)
    print(fusion_mask.shape)

    # cv2.imwrite(save_path, out_texture.astype(np.uint8))
    return out_texture


def delighting(texture):
    img = texture.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = np.mean(img_gray.astype(np.float))
    thresh = int(thresh + 40)
    _, mask = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.blur(mask, ksize=(100, 100))

    scale = texture.shape[0] / 1024

    # alpha blend
    mean_bg = (np.mean(img[int(422*scale): int((422 + 76)*scale), int(388*scale): int((388 + 76)*scale)], axis=(0, 1), keepdims=True) + np.mean(
        img[int(422*scale): int((422 + 76)*scale), -int((388 + 76)*scale): -int(388*scale)],
        axis=(0, 1),
        keepdims=True)) / 2  # [270: 370, 210: 310]
    mean_bg = np.ones(img.shape, dtype=np.float) * mean_bg
    mask = np.stack([mask, mask, mask], axis=2).astype(np.float) / 255.0
    ind = mask > 0.05
    img[ind] = img[ind] * (1 - 0.7 * mask[ind]) + mean_bg[ind] * 0.7 * mask[ind]

    return img

def calc_BFM_texmap(basis_tex_maps, tex_coeff):
    out_tex = basis_tex_maps[:, :, 0:3].copy()

    for i in range(len(tex_coeff)):
        out_tex += tex_coeff[i] * basis_tex_maps[:, :, (i+1)*3:(i+2)*3]

    return np.clip(out_tex[:,:,::-1]*255,0,255)


def write_obj(save_path, vertices, faces=None, UVs=None, faces_uv=None, normals=None, faces_normal=None, texture_map=None, save_mtl=False, vertices_color=None):
    save_dir = os.path.dirname(save_path)
    save_name = os.path.splitext(os.path.basename(save_path))[0]

    if save_mtl or texture_map is not None:
        if texture_map is not None:
            cv2.imwrite(os.path.join(save_dir, save_name + '.jpg'), texture_map)
        with open(os.path.join(save_dir, save_name + '.mtl'), 'w') as wf:
            wf.write('# Created by HRN\n')
            wf.write('newmtl material_0\n')
            wf.write('Ka 1.000000 0.000000 0.000000\n')
            wf.write('Kd 1.000000 1.000000 1.000000\n')
            wf.write('Ks 0.000000 0.000000 0.000000\n')
            wf.write('Tr 0.000000\n')
            wf.write('illum 0\n')
            wf.write('Ns 0.000000\n')
            wf.write('map_Kd {}\n'.format(save_name + '.jpg'))

    with open(save_path, 'w') as wf:
        if save_mtl or texture_map is not None:
            wf.write("# Create by HRN\n")
            wf.write("mtllib ./{}.mtl\n".format(save_name))

        if vertices_color is not None:
            for i, v in enumerate(vertices):
                wf.write('v {} {} {} {} {} {}\n'.format(v[0], v[1], v[2], vertices_color[i][0], vertices_color[i][1], vertices_color[i][2]))
        else:
            for v in vertices:
                wf.write('v {} {} {}\n'.format(v[0], v[1], v[2]))

        if UVs is not None:
            for uv in UVs:
                wf.write('vt {} {}\n'.format(uv[0], uv[1]))

        if normals is not None:
            for vn in normals:
                wf.write('vn {} {} {}\n'.format(vn[0], vn[1], vn[2]))

        if faces is not None:
            for ind, face in enumerate(faces):
                if faces_uv is not None or faces_normal is not None:
                    if faces_uv is not None:
                        face_uv = faces_uv[ind]
                    else:
                        face_uv = face
                    if faces_normal is not None:
                        face_normal = faces_normal[ind]
                    else:
                        face_normal = face
                    row = 'f ' + ' '.join(['{}/{}/{}'.format(face[i], face_uv[i], face_normal[i]) for i in range(len(face))]) + '\n'
                else:
                    row = 'f ' + ' '.join(['{}'.format(face[i]) for i in range(len(face))]) + '\n'
                wf.write(row)


def read_obj(obj_path, print_shape=False):
    with open(obj_path, 'r') as f:
        bfm_lines = f.readlines()

    vertices = []
    faces = []
    uvs = []
    vns = []
    faces_uv = []
    faces_normal = []
    max_face_length = 0
    for line in bfm_lines:
        if line[:2] == 'v ':
            vertex = [float(a) for a in line.strip().split(' ')[1:] if len(a)>0]
            vertices.append(vertex)

        if line[:2] == 'f ':
            items = line.strip().split(' ')[1:]
            face = [int(a.split('/')[0]) for a in items if len(a)>0]
            max_face_length = max(max_face_length, len(face))
            # if len(faces) > 0 and len(face) != len(faces[0]):
            #     continue
            faces.append(face)

            if '/' in items[0] and len(items[0].split('/')[1])>0:
                face_uv = [int(a.split('/')[1]) for a in items if len(a)>0]
                faces_uv.append(face_uv)

            if '/' in items[0] and len(items[0].split('/')) >= 3 and len(items[0].split('/')[2])>0:
                face_normal = [int(a.split('/')[2]) for a in items if len(a)>0]
                faces_normal.append(face_normal)

        if line[:3] == 'vt ':
            items = line.strip().split(' ')[1:]
            uv = [float(a) for a in items if len(a)>0]
            uvs.append(uv)

        if line[:3] == 'vn ':
            items = line.strip().split(' ')[1:]
            vn = [float(a) for a in items if len(a)>0]
            vns.append(vn)

    vertices = np.array(vertices).astype(np.float32)
    if max_face_length <= 3:
        faces = np.array(faces).astype(np.int32)
    else:
        print('not a triangle face mesh!')

    if vertices.shape[1] == 3:
        mesh = {
            'vertices': vertices,
            'faces': faces,
        }
    else:
        mesh = {
            'vertices': vertices[:, :3],
            'colors': vertices[:, 3:],
            'faces': faces,
        }

    if len(uvs) > 0:
        uvs = np.array(uvs).astype(np.float32)
        mesh['UVs'] = uvs

    if len(vns) > 0:
        vns = np.array(vns).astype(np.float32)
        mesh['normals'] = vns

    if len(faces_uv) > 0:
        if max_face_length <= 3:
            faces_uv = np.array(faces_uv).astype(np.int32)
        mesh['faces_uv'] = faces_uv

    if len(faces_normal) > 0:
        if max_face_length <= 3:
            faces_normal = np.array(faces_normal).astype(np.int32)
        mesh['faces_normal'] = faces_normal

    if print_shape:
        print('num of vertices', len(vertices))
        print('num of faces', len(faces))
    return mesh


def write_obj2(save_path, mesh):
    save_dir = os.path.dirname(save_path)
    save_name = os.path.splitext(os.path.basename(save_path))[0]

    if 'texture_map' in mesh:
        cv2.imwrite(
            os.path.join(save_dir, save_name + '.jpg'), mesh['texture_map'])

        with open(os.path.join(save_dir, save_name + '.mtl'), 'w') as wf:
            wf.write('# Created by HRN\n')
            wf.write('newmtl material_0\n')
            wf.write('Ka 1.000000 0.000000 0.000000\n')
            wf.write('Kd 1.000000 1.000000 1.000000\n')
            wf.write('Ks 0.000000 0.000000 0.000000\n')
            wf.write('Tr 0.000000\n')
            wf.write('illum 0\n')
            wf.write('Ns 0.000000\n')
            wf.write('map_Kd {}\n'.format(save_name + '.jpg'))

    with open(save_path, 'w') as wf:
        if 'texture_map' in mesh:
            wf.write('# Create by HRN\n')
            wf.write('mtllib ./{}.mtl\n'.format(save_name))

        if 'colors' in mesh:
            for i, v in enumerate(mesh['vertices']):
                wf.write('v {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    v[0], v[1], v[2], mesh['colors'][i][0],
                    mesh['colors'][i][1], mesh['colors'][i][2]))
        else:
            for v in mesh['vertices']:
                wf.write('v {:.6f} {:.6f} {:.6f}\n'.format(v[0], v[1], v[2]))

        if 'UVs' in mesh:
            for uv in mesh['UVs']:
                wf.write('vt {:.6f} {:.6f}\n'.format(uv[0], uv[1]))

        if 'normals' in mesh:
            for vn in mesh['normals']:
                wf.write('vn {:.6f} {:.6f} {:.6f}\n'.format(vn[0], vn[1], vn[2]))

        if 'faces' in mesh:
            for ind, face in enumerate(mesh['faces']):
                if 'faces_uv' in mesh or 'faces_normal' in mesh or 'UVs' in mesh:
                    if 'faces_uv' in mesh:
                        face_uv = mesh['faces_uv'][ind]
                    else:
                        face_uv = face
                    if 'faces_normal' in mesh:
                        face_normal = mesh['faces_normal'][ind]
                    else:
                        face_normal = face
                    row = 'f ' + ' '.join([
                        '{}/{}/{}'.format(face[i], face_uv[i], face_normal[i])
                        for i in range(len(face))
                    ]) + '\n'
                else:
                    row = 'f ' + ' '.join(
                        ['{}'.format(face[i])
                         for i in range(len(face))]) + '\n'
                wf.write(row)


def concat_mesh(mesh1, mesh2):
    vertices1 = mesh1['vertices']
    vertices2 = mesh2['vertices']

    faces1 = mesh1['faces']
    faces2 = mesh2['faces']

    new_vertices = np.concatenate([vertices1, vertices2], axis=0)
    new_faces = np.concatenate([faces1, faces2 + len(vertices1)], axis=0)

    new_mesh = {
        'vertices': new_vertices,
        'faces': new_faces,
    }

    if 'colors' in mesh1 and 'colors' in mesh2:
        new_mesh['colors'] = np.concatenate([mesh1['colors'], mesh2['colors']], axis=0)

    return new_mesh


def crop_mesh(mesh, keep_vert_inds):
    vertices = mesh['vertices'].copy()
    faces = mesh['faces']
    new_vertices = vertices[keep_vert_inds].copy()
    faces -= 1

    inds_mapping = dict()
    for i in range(len(keep_vert_inds)):
        inds_mapping[keep_vert_inds[i]] = i

    new_faces = []
    keep_face_inds = []
    for ind, face in enumerate(faces):
        if face[0] in inds_mapping and face[1] in inds_mapping and face[2] in inds_mapping:
            new_face = [inds_mapping[face[0]], inds_mapping[face[1]], inds_mapping[face[2]]]
            new_faces.append(new_face)
            keep_face_inds.append(ind)
    new_faces = np.array(new_faces)
    new_faces += 1
    keep_face_inds = np.array(keep_face_inds)

    new_mesh = mesh.copy()
    new_mesh['vertices'] = new_vertices
    new_mesh['faces'] = new_faces
    if 'colors' in new_mesh:
        new_mesh['colors'] = new_mesh['colors'][keep_vert_inds]
    if 'faces_uv' in new_mesh:
        new_mesh['faces_uv'] = new_mesh['faces_uv'][keep_face_inds]
    if 'faces_normal' in new_mesh:
        new_mesh['faces_normal'] = new_mesh['faces_normal'][keep_face_inds]

    return new_mesh, keep_face_inds


def remove_isolate_vertices(mesh):
    keep_vertices_inds = set()

    for face in mesh['faces']:
        for a in face:
            keep_vertices_inds.add(a - 1)

    keep_vertices_inds = sorted(list(keep_vertices_inds))
    keep_vertices_inds = np.array(keep_vertices_inds)

    new_mesh, _ = crop_mesh(mesh, keep_vertices_inds)

    return new_mesh


def get_colored_mesh_from_textured(mesh, texture):
    colors = np.zeros((len(mesh['vertices']), 3), dtype=np.float32)
    texture = texture.astype(np.float32)
    h, w = texture.shape[:2]

    for ind in range(len(mesh['faces_uv'])):
        for i in range(3):
            vert_ind = mesh['faces'][ind][i] - 1
            uv_ind = mesh['faces_uv'][ind][i] - 1
            u = mesh['UVs'][uv_ind][0]
            v = mesh['UVs'][uv_ind][1]

            x = min(int(u * w), w-1)
            y = min(int((1 - v) * h), h-1)

            colors[vert_ind] = texture[y, x]

    mesh['colors'] = colors[..., ::-1].copy() / 255.0

    return mesh

def img_value_rescale(img, old_range: list, new_range: list):
    assert len(old_range) == 2
    assert len(new_range) == 2
    img = (img - old_range[0]) / (old_range[1] - old_range[0]) * (new_range[1] - new_range[0]) + new_range[0]
    return img


def resize_on_long_side(img, long_side=800):
    src_height = img.shape[0]
    src_width = img.shape[1]

    if src_height > src_width:
        scale = long_side * 1.0 / src_height
        _img = cv2.resize(img, (int(src_width * scale), long_side), interpolation=cv2.INTER_CUBIC)


    else:
        scale = long_side * 1.0 / src_width
        _img = cv2.resize(img, (long_side, int(src_height * scale)), interpolation=cv2.INTER_CUBIC)

    return _img, scale

def get_mg_layer(src, gt, skin_mask=None):
    """
    src, gt shape: [h, w, 3] value: [0, 1]
    return: mg, shape: [h, w, 1] value: [0, 1]
    """
    mg = (src * src - gt + 1e-10) / (2 * src * src - 2 * src + 2e-10)
    mg[mg < 0] = 0.5
    mg[mg > 1] = 0.5

    diff_abs = np.abs(gt - src)
    mg[diff_abs < (1/255.0)] = 0.5

    if skin_mask is not None:
        t1 = time.time()
        mg[skin_mask==0] = 0.5
        print('use mask', time.time() - t1)

    return mg

def vis_landmarks(img, kpts, write_path, scale=1.0):
    img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
    kpts = kpts.copy() * scale
    for ind, pt in enumerate(kpts):
        img = cv2.circle(img, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
        img = cv2.putText(img, '{}'.format(ind), (int(pt[0]), int(pt[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    cv2.imwrite(write_path, img)


def warp(x, flow, mode='bilinear', padding_mode='zeros', coff=0.1):
    """

    Args:
        x: [n, c, h, w]
        flow: [n, h, w, 2]
        mode:
        padding_mode:
        coff:

    Returns:

    """
    n, c, h, w = x.size()
    yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
    xv = xv.float() / (w - 1) * 2.0 - 1
    yv = yv.float() / (h - 1) * 2.0 - 1

    '''
    grid[0,:,:,0] =
    -1, .....1
    -1, .....1
    -1, .....1

    grid[0,:,:,1] =
    -1,  -1, -1
     ;        ;
     1,   1,  1

    '''

    if torch.cuda.is_available():
        grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0).cuda()
    else:
        grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0)
    grid_x = grid + 2 * flow * coff
    warp_x = F.grid_sample(x, grid_x, mode=mode, padding_mode=padding_mode)
    return warp_x


def sharpen_img(img, degree=1.5):
    blur_img = cv2.GaussianBlur(img, (0, 0), 5)
    usm = cv2.addWeighted(img, degree, blur_img, 1-degree, 0)
    return usm


def split_vis(img_path, target_dir=None):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    n_split = w // h
    if target_dir is None:
        target_dir = os.path.dirname(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    for i in range(n_split):
        img_i = img[:, i*h: (i+1)*h, :]
        cv2.imwrite(os.path.join(target_dir, '{}_{:0>2d}.jpg'.format(base_name, i+1)), img_i)


def write_video(image_list, save_path, fps=20.0):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # avi格式

    h, w = image_list[0].shape[:2]

    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h), True)

    for frame in image_list:
        out.write(frame)

    out.release()
