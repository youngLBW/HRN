import numpy as np
import cv2
import math
import numba
import time
import torch
import torch.nn.functional as F


def viz_flow(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


@numba.jit(nopython=True, parallel=True)
def bilinear_interp(x, y, v11, v12, v21, v22):
    t = 0.2

    if x < t and y < t:
        return v11
    elif x < t and y > 1 - t:
        return v12
    elif x > 1 - t and y < t:
        return v21
    elif x > 1 - t and y > 1 - t:
        return v22
    else:
        result = (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x
        if result < 0:
            result = 0

        if result > 255:
            result = 255
        return result


def image_warp_cuda(flow, oriImg):
    x = torch.from_numpy(oriImg.astype(np.float32)).cuda()
    x = x.permute((2, 0, 1)).unsqueeze(0)

    flow_tensor = torch.from_numpy(flow).unsqueeze(0).cuda()

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

    flow_tensor[:, :, :, 0] /= flow.shape[1]
    flow_tensor[:, :, :, 1] /= flow.shape[0]

    grid_x = grid + 2 * flow_tensor

    mode = 'bilinear'
    padding_mode = 'zeros'

    warp_x = F.grid_sample(x, grid_x, mode=mode, padding_mode=padding_mode)

    warp_x = warp_x.squeeze().permute((1, 2, 0)).cpu().numpy()

    return warp_x


@numba.jit(nopython=True, parallel=True)
def image_warp_grid1(rDx, rDy, oriImg, transRatio, pads):
    # assert oriImg.dtype == np.uint8
    grid_size = 1
    srcW = oriImg.shape[1]
    srcH = oriImg.shape[0]

    padTop, padBottom, padLeft, padRight = pads

    left_bound = padLeft + 1
    right_bound = srcW - padRight
    bottom_bound = srcH - padBottom
    top_bound = padTop + 1

    newImg = oriImg.copy()

    for i in range(srcH):
        for j in range(srcW):
            _i = i
            _j = j

            deltaX = rDx[_i, _j]
            deltaY = rDy[_i, _j]

            if abs(deltaX) < 0.2 and abs(deltaY) < 0.2:
                continue

            nx = _j + deltaX * transRatio
            ny = _i + deltaY * transRatio

            if nx >= srcW - padRight:
                if nx > srcW - 1:
                    nx = srcW - 1

                if _j < right_bound:
                    right_bound = _j

            if ny >= srcH - padBottom:
                if ny > srcH - 1:
                    ny = srcH - 1

                if _i < bottom_bound:
                    bottom_bound = _i

            if nx < padLeft:
                if nx < 0:
                    nx = 0

                if _j + 1 > left_bound:
                    left_bound = _j + 1

            if ny < padTop:
                if ny < 0:
                    ny = 0

                if _i + 1 > top_bound:
                    top_bound = _i + 1

            nxi = int(math.floor(nx))
            nyi = int(math.floor(ny))
            nxi1 = int(math.ceil(nx))
            nyi1 = int(math.ceil(ny))

            # if nxi < 0 or nyi < 0 or nxi1 <0 or nyi1 < 0:
            #     print('nxi:{}, nyi:{}, nxi1:{}, nyi1:{}'.format(nxi, nyi, nxi1, nyi1))

            if nxi < 0:
                nxi =0
            if nxi > oriImg.shape[1]-1:
                nxi = oriImg.shape[1]-1

            if nxi1 < 0:
                nxi1 =0
            if nxi1 > oriImg.shape[1]-1:
                nxi1 = oriImg.shape[1]-1

            if nyi < 0:
                nyi = 0
            if nyi > oriImg.shape[0] - 1:
                nyi = oriImg.shape[0] - 1

            if nyi1 < 0:
                nyi1 = 0
            if nyi1 > oriImg.shape[0] - 1:
                nyi1 = oriImg.shape[0] - 1

            for ll in range(3):
                newImg[_i, _j, ll] = bilinear_interp(
                    ny - nyi, nx - nxi,
                    oriImg[nyi, nxi, ll],
                    oriImg[nyi, nxi1, ll],
                    oriImg[nyi1, nxi, ll],
                    oriImg[nyi1, nxi1, ll])

    return newImg, top_bound, bottom_bound, left_bound, right_bound


@numba.jit(nopython=True)
def image_warp(srcW, srcH, rDx, rDy, oriImg, transRatio, newImg):
    grid_size = 3
    tarW = srcW
    tarH = srcH
    i_range = int(math.floor(tarH / grid_size))
    j_range = int(math.floor(tarW / grid_size))
    for i in range(i_range):
        for j in range(j_range):
            _i = i * grid_size
            _j = j * grid_size
            ni = _i + grid_size
            nj = _j + grid_size
            w = h = grid_size
            if ni >= tarH:
                ni = tarH - 1
                h = ni - _i + 1
            if nj >= tarW:
                nj = tarW - 1
                w = nj - _j + 1

            for di in range(h):
                for dj in range(w):

                    deltaX = bilinear_interp(di * 1.0 / h, dj * 1.0 / w, rDx[_i, _j], rDx[_i, nj],
                                             rDx[ni, _j], rDx[ni, nj])
                    deltaY = bilinear_interp(di * 1.0 / h, dj * 1.0 / w, rDy[_i, _j], rDy[_i, nj],
                                             rDy[ni, _j], rDy[ni, nj])

                    nx = _j + dj + deltaX * transRatio
                    ny = _i + di + deltaY * transRatio
                    if nx > srcW - 1:
                        nx = srcW - 1
                    if ny > srcH - 1:
                        ny = srcH - 1
                    if nx < 0:
                        nx = 0
                    if ny < 0:
                        ny = 0

                    nxi = int(math.floor(nx))
                    nyi = int(math.floor(ny))
                    nxi1 = int(math.ceil(nx))
                    nyi1 = int(math.ceil(ny))

                    for ll in range(3):
                        newImg[_i + di, _j + dj, ll] = bilinear_interp(
                            ny - nyi, nx - nxi,
                            oriImg[nyi, nxi, ll],
                            oriImg[nyi, nxi1, ll],
                            oriImg[nyi1, nxi, ll],
                            oriImg[nyi1, nxi1, ll])

    return newImg


@numba.jit(nopython=True)
def getDist_Point2Line(pointP, pointA, pointB):
    A = pointA[1] - pointB[1]
    B = pointB[0] - pointA[0]
    C = pointA[0] * pointB[1] - pointA[1] * pointB[0]
    distance = (math.fabs(A * pointP[0] + B * pointP[1] + C)) / (math.sqrt(A * A + B * B))
    return distance


def calc_angle(vec_1, vec_2):
    inner_prod = vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]
    inner_prod = inner_prod / (math.sqrt(vec_1[0] ** 2 + vec_1[1] ** 2)) / (math.sqrt(vec_2[0] ** 2 + vec_2[1] ** 2))
    return math.acos(inner_prod) / math.pi * 180


def calc_distance(pt_1, pt_2):
    dis = pt_1 - pt_2
    return math.sqrt(dis[0] ** 2 + dis[1] ** 2)


def liquify(brush_radius, brush_center_start, brush_center_dst, pressure, rDx, rDy, uniform_brush=False):
    assert rDx.dtype == np.float
    assert rDy.dtype == np.float
    assert rDy.shape[0] == rDx.shape[0]
    assert rDy.shape[1] == rDx.shape[1]

    if np.linalg.norm(brush_center_start - brush_center_dst) < 1:
        return rDx, rDy

    img_width = rDx.shape[1]
    img_height = rDx.shape[0]

    center_dist = np.linalg.norm(brush_center_dst - brush_center_start)

    mask = np.zeros((rDx.shape[0], rDx.shape[1]), dtype=np.float)

    trace_mask = np.zeros((2 * brush_radius, int(center_dist) + 2 * brush_radius), dtype=np.float)

    cv2.circle(trace_mask, (brush_radius, brush_radius), brush_radius, 255, -1)
    cv2.circle(trace_mask, (brush_radius + int(center_dist), brush_radius), brush_radius, 255, -1)
    cv2.rectangle(trace_mask, (brush_radius, 0), (brush_radius + int(center_dist), brush_radius * 2), 255, -1)

    degree = calc_angle(brush_center_dst - brush_center_start, [1, 0])
    clockwise = np.cross(np.array([1, 0]), brush_center_dst - brush_center_start)

    degree = -np.sign(clockwise) * degree

    # print('clockwise:{}'.format(clockwise))
    # print('degree:{}'.format(degree))

    heightNew = int(trace_mask.shape[1] * math.fabs(math.sin(math.radians(degree))) + trace_mask.shape[0] * math.fabs(
        math.cos(math.radians(degree))))
    widthNew = int(trace_mask.shape[0] * math.fabs(math.sin(math.radians(degree))) + trace_mask.shape[1] * math.fabs(
        math.cos(math.radians(degree))))
    #
    matRotation = cv2.getRotationMatrix2D((trace_mask.shape[1] / 2, trace_mask.shape[0] / 2), degree, 1)
    #
    matRotation[0, 2] += (widthNew - trace_mask.shape[1]) / 2
    matRotation[1, 2] += (heightNew - trace_mask.shape[0]) / 2

    trace_mask = cv2.warpAffine(trace_mask, matRotation, (widthNew, heightNew), borderValue=0)

    # cv2.circle(img, tuple(brush_center_start), brush_radius, (255,0,0), 2)
    # cv2.circle(img, tuple(brush_center_dst), brush_radius, (0,0,255), 2)
    # cv2.imwrite('img_vis.jpg', img)

    location = (brush_center_start + brush_center_dst) / 2
    # print('location:{}'.format(location))
    # print('trace_mask:{}'.format(trace_mask.shape))
    #
    # print('mask shape:{}'.format(mask.shape))
    mask = cv2.copyMakeBorder(mask, trace_mask.shape[0], trace_mask.shape[0], trace_mask.shape[1], trace_mask.shape[1],
                              cv2.BORDER_CONSTANT, value=0)
    mask[int(trace_mask.shape[0] + location[1] - trace_mask.shape[0] / 2):int(
        trace_mask.shape[0] + location[1] + trace_mask.shape[0] / 2),
    int(trace_mask.shape[1] + location[0] - trace_mask.shape[1] / 2):int(
        trace_mask.shape[1] + location[0] + trace_mask.shape[1] / 2)] = trace_mask

    mask = mask[trace_mask.shape[0]:-trace_mask.shape[0], trace_mask.shape[1]:-trace_mask.shape[1]]
    cv2.imwrite('mask.jpg', mask)
    # print('mask shape2:{}'.format(mask.shape))

    unit_direction = (brush_center_dst - brush_center_start) / np.linalg.norm(brush_center_dst - brush_center_start)

    start_pt_a = brush_center_start - unit_direction * brush_radius
    start_pt_b = brush_center_start + np.array([-unit_direction[1], unit_direction[0]],
                                               dtype=np.float) - unit_direction * brush_radius

    if center_dist >= brush_radius:
        move_distance = brush_radius
    else:
        move_distance = center_dist

    rDx, rDy = calc_rDx_rDy(img_width, img_height, mask, start_pt_a, start_pt_b, brush_radius, brush_center_start,
                            brush_center_dst, pressure, move_distance, unit_direction, rDx, rDy, uniform=uniform_brush)

    return rDx, rDy


@numba.jit(nopython=True)
def guassian_function(sigma, x):
    y = 1.0 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-x ** 2 / (2 * (sigma ** 2)))
    return y


@numba.jit(nopython=True)
def calc_rDx_rDy(img_width, img_height, mask, start_pt_a, start_pt_b, brush_radius, brush_center_start,
                 brush_center_dst, pressure, move_distance, unit_direction, rDx, rDy, uniform=False):
    for x in range(img_width):
        for y in range(img_height):
            if mask[y, x] > 128:
                dist_to_start_line = getDist_Point2Line([x, y], start_pt_a, start_pt_b)

                if dist_to_start_line > brush_radius:
                    dist_to_line = getDist_Point2Line([x, y], brush_center_start, brush_center_dst) / brush_radius
                    # normalized_moment = 1- pow(dist_to_line,2)

                    if not uniform:
                        sigma = 0.5
                        normalized_moment = guassian_function(sigma, dist_to_line)
                        normalized_moment -= guassian_function(sigma, 1)
                        normalized_moment = normalized_moment / (
                                    guassian_function(sigma, 0) - guassian_function(sigma, 1) + 1e-7)
                    else:
                        normalized_moment = 1

                    new_rDx_value = - pressure * move_distance * normalized_moment * unit_direction[0]
                    new_rDy_value = - pressure * move_distance * normalized_moment * unit_direction[1]
                    rDx[y, x] = rDx[y, x] if math.fabs(rDx[y, x]) > math.fabs(new_rDx_value) else new_rDx_value
                    rDy[y, x] = rDy[y, x] if math.fabs(rDy[y, x]) > math.fabs(new_rDy_value) else new_rDy_value

                else:
                    dist_to_line = getDist_Point2Line([x, y], brush_center_start, brush_center_dst) / brush_radius
                    # normalized_moment = 1- pow(dist_to_line,2)

                    if not uniform:
                        sigma = 0.5
                        normalized_moment = guassian_function(sigma, dist_to_line)
                        cut_x = math.sqrt(brush_radius ** 2 - (brush_radius - dist_to_start_line) ** 2) / brush_radius
                        normalized_moment -= guassian_function(sigma, cut_x)
                        normalized_moment = normalized_moment / (
                                    guassian_function(sigma, 0) - guassian_function(sigma, cut_x) + 1e-7)
                    else:
                        normalized_moment = 1

                    new_rDx_value = -dist_to_start_line / brush_radius * pressure * move_distance * normalized_moment * \
                                    unit_direction[0]

                    new_rDy_value = -dist_to_start_line / brush_radius * pressure * move_distance * normalized_moment * \
                                    unit_direction[1]

                    rDx[y, x] = rDx[y, x] if math.fabs(rDx[y, x]) > math.fabs(new_rDx_value) else new_rDx_value
                    rDy[y, x] = rDy[y, x] if math.fabs(rDy[y, x]) > math.fabs(new_rDy_value) else new_rDy_value

    return rDx, rDy


if __name__ == "__main__":
    img = cv2.imread('../test_data/d11cdb418a74a79ca2c1b18e50ee685c6104-photo.jpg')
    print('img shape:{}'.format(img.shape))
    import random

    grid_size = 10
    x = 0
    while x < img.shape[1]:
        cv2.line(img, (int(x), 0), (int(x), img.shape[0] - 1), (0, 180, 0), 1)
        x += grid_size

    y = 0
    while y < img.shape[0]:
        cv2.line(img, (0, int(y)), (img.shape[1] - 1, int(y)), (0, 180, 0), 1)

        y += grid_size

    rDx = np.zeros((img.shape[0], img.shape[1]), dtype=np.float)
    rDy = np.zeros((img.shape[0], img.shape[1]), dtype=np.float)

    t1 = time.time()
    for i in range(1):
        rDx, rDy = liquify(random.randint(60, 160), np.array([random.randint(10, 800), random.randint(10, 800)]),
                           np.array([random.randint(10, 800), random.randint(10, 1000)]), 0.5, rDx, rDy,
                           uniform_brush=False)
    print(' linquify: {}ms'.format(int((time.time() - t1) * 1000)))

    vis_rDx = (rDx - np.min(rDx)) / (np.max(rDx) - np.min(rDx) + 1e-10) * 255
    vis_rDy = (rDy - np.min(rDy)) / (np.max(rDy) - np.min(rDy) + 1e-10) * 255

    cv2.imwrite('rdx.jpg', vis_rDx)
    cv2.imwrite('rdy.jpg', vis_rDy)

    oriImg = img.astype(np.float)
    newImg = np.zeros((oriImg.shape[0], oriImg.shape[1], oriImg.shape[2]), dtype=np.float)

    rdx_vis = np.zeros((oriImg.shape[0], oriImg.shape[1]), dtype=np.float)
    rdy_vis = np.zeros((oriImg.shape[0], oriImg.shape[1]), dtype=np.float)
    liquified_img = image_warp(oriImg.shape[1], oriImg.shape[0], rDx, rDy, oriImg, 0.8, newImg)
    cv2.imwrite('liquified.jpg', liquified_img)

