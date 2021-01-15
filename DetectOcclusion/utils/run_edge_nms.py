# run nms on edge map

import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # use matplotlib without gui support
from PIL import Image
from edge_nms import *


def edge_nms(edge):
    """
    :param edge: edge probability, HxW, value: 0~1
    """
    dy, dx = np.gradient(conv_tri(edge, radius=4))  # edge gradients
    _, dxx = np.gradient(dx)
    dyy, dxy = np.gradient(dy)
    orientation = np.arctan2(dyy * np.sign(-dxy) + 1e-5, dxx)  # orientation by dyy, dxx
    orientation[orientation < 0] += np.pi

    edge_thin = non_maximum_supr(edge, orientation, r=1, s=5, m=1.01)

    return edge_thin


# def non_maximum_supr(edge, ori, r, s, m):
#     """
#     Non-Maximum Suppression (from https://github.com/ArtanisCV/StructuredForests/ cython version)
#     :param edge: original edge map HxW
#     :param ori: orientation map HxW
#     :param r: radius for nms suppression
#     :param s: radius for suppress boundaries
#     :param m: multiplier for conservative suppression
#     :return: suppressed edge map
#     """
#
#     h = edge.shape[0]
#     w = edge.shape[1]
#     E_arr = np.zeros((h, w), dtype=np.float64)
#     E = E_arr
#     C = np.cos(ori)
#     S = np.sin(ori)
#
#     # suppress edges where edge is stronger in orthogonal direction
#     for y in range(0, h):
#         for x in range(0, w):
#             e = E[y, x] = edge[y, x]
#             if e == 0:
#                 continue
#
#             e *= m
#             co = C[y, x]
#             si = S[y, x]
#
#             for d in range(-r, r + 1):
#                 if d != 0:
#                     e0 = bilinear_interp(edge, x + d * co, y + d * si)
#                     if e < e0:
#                         E[y, x] = 0
#                         break
#
#     # suppress noisy edge estimates near boundaries
#     s = w / 2 if s > w / 2 else s
#     s = h / 2 if s > h / 2 else s
#
#     for x in range(0, s):
#         for y in range(0, h):
#             E[y, x] *= x / s
#             E[y, w - 1 - x] *= x / s
#
#     for x in range(0, w):
#         for y in range(0, s):
#             E[y, x] *= y / s
#             E[h - 1 - y, x] *= y / s
#
#     return E_arr
#
#
# def bilinear_interp(img, x, y):
#     """
#     Return img[y, x] via bilinear interpolation
#     """
#     h, w = img.shape[0:2]
#
#     if x < 0:
#         x = 0
#     elif x > w - 1.001:
#         x = w - 1.001
#
#     if y < 0:
#         y = 0
#     elif y > h - 1.001:
#         y = h - 1.001
#
#     x0 = int(x)
#     y0 = int(y)
#     x1 = x0 + 1
#     y1 = y0 + 1
#     dx0 = x - x0
#     dy0 = y - y0
#     dx1 = 1 - dx0
#     dy1 = 1 - dy0
#
#     return img[y0, x0]*dx1*dy1 + img[y0, x1]*dx0*dy1 + img[y1, x0]*dx1*dy0 + img[y1, x1]*dx0*dy0


def conv_tri(src, radius):
    """
    Image convolution with a triangle filter along row and column separately.
    :param src: input image
    :param radius: gradient normalization radius
    :return: convolution result
    """

    if radius == 0:
        return src
    elif radius <= 1:
        p = 12.0 / radius / (radius + 2) - 2
        kernel = np.asarray([1, p, 1], dtype=np.float64) / (p + 2)
        return cv2.sepFilter2D(src, ddepth=-1, kernelX=kernel, kernelY=kernel, borderType=cv2.BORDER_REFLECT)
    else:
        radius = int(radius)
        kernel = list(range(1, radius + 1)) + [radius + 1] + list(range(radius, 0, -1))  # [1,...,r+1,...,1]
        kernel = np.asarray(kernel, dtype=np.float64) / (radius + 1)**2
        return cv2.sepFilter2D(src, ddepth=-1, kernelX=kernel, kernelY=kernel, borderType=cv2.BORDER_REFLECT)


def thin_p2orm_w_edge(edge_prob, im_name):
    """
    thin p2orm, keep pairwise occlusion if at least one pixel is on the occlusion boundary.
    visually may result in holes along SE, NE inclinations.
    """
    order_in_path = "{}-order-pix.npy".format(im_name)
    occ_order_pix = np.load(order_in_path, allow_pickle=True)  # occ_edge_prob + occ_order ;H,W,9; {-1,0,1}
    H, W = occ_order_pix.shape[:2]

    occ_order_pair = np.ones((H, W, 4))  # p2orm
    occ_order_pair[:, :, 0] = occ_order_pix[:, :, 5]
    occ_order_pair[:, :, 1] = occ_order_pix[:, :, 7]
    occ_order_pair[:, :, 2] = occ_order_pix[:, :, 8]
    occ_order_pair[:, :, 3] = occ_order_pix[:, :, 3]

    # filter out no occ regions
    no_occ_mask = np.zeros((H, W), dtype=np.uint8)  # pixels not on occ edge
    no_occ_mask[edge_prob < 0.5] = 1
    no_occ_E  = np.zeros((H, W), dtype=np.uint8)  # no pairwise occ along E
    no_occ_S  = np.zeros((H, W), dtype=np.uint8)  # no pairwise occ along S
    no_occ_SE = np.zeros((H, W), dtype=np.uint8)  # no pairwise occ along SE
    no_occ_NE = np.zeros((H, W), dtype=np.uint8)  # no pairwise occ along NE
    no_occ_E[:, :W-1] = no_occ_mask[:, :W-1] * no_occ_mask[:, 1:]  # H,W-1
    no_occ_S[:H-1, :] = no_occ_mask[:H-1, :] * no_occ_mask[1:, :]  # H-1,W
    no_occ_SE[:H-1, :W-1] = no_occ_mask[:H-1, :W-1] * no_occ_mask[1:, 1:]  # H-1,W-1
    no_occ_NE[1:, :W-1]   = no_occ_mask[:H-1, 1:] * no_occ_mask[1:, :W-1]  # H-1,W-1

    occ_order_pair[no_occ_E == 1, 0] = 0
    occ_order_pair[no_occ_S == 1, 1] = 0
    occ_order_pair[no_occ_SE == 1, 2] = 0
    occ_order_pair[no_occ_NE == 1, 3] = 0

    # save res
    vis_p2orm(occ_order_pair, im_name)


def vis_p2orm(occ_order, name):
    """
    vis p2orm and save
    :param occ_order: HxWx4, value [-1, 0, 1]
    :param name: image name
    :return:
    """
    cm = plt.get_cmap('bwr')
    occ_order = (occ_order + 1.) / 2  # => [0, 0.5, 1]

    colored_image_E = cm(occ_order[:, :, 0])
    colored_image_S = cm(occ_order[:, :, 1])
    colored_image_SE = cm(occ_order[:, :, 2])
    colored_image_NE = cm(occ_order[:, :, 3])

    viz_out_path = os.path.join('{}-order-viz.png'.format(name))
    Image.fromarray((colored_image_E[:, :, :3] * 255).astype(np.uint8)).save(viz_out_path.replace('-viz', '-viz-E'))
    Image.fromarray((colored_image_S[:, :, :3] * 255).astype(np.uint8)).save(viz_out_path.replace('-viz', '-viz-S'))
    Image.fromarray((colored_image_SE[:, :, :3] * 255).astype(np.uint8)).save(viz_out_path.replace('-viz', '-viz-SE'))
    Image.fromarray((colored_image_NE[:, :, :3] * 255).astype(np.uint8)).save(viz_out_path.replace('-viz', '-viz-NE'))


if __name__ == '__main__':
    im_name = 'corridor_02-rgb'
    f_name = '{}_lab_v_g.png'.format(im_name)
    out_name = '{}_lab_v_g_nms.png'.format(im_name)
    edge_prob = cv2.imread(f_name, -1) / 255.  # 0~1
    tic = time.time()
    edge_prob_nms = edge_nms(edge_prob)
    used_t = time.time() - tic
    print('used time is {:.4f} s'.format(used_t))

    cv2.imwrite(out_name, (edge_prob_nms * 255).astype(np.uint8))

    edge_nms = np.zeros(edge_prob_nms.shape, dtype=np.uint8)
    edge_nms[edge_prob_nms > 0.5] = 1
    cv2.imwrite('{}_edge_nms.png'.format(im_name),  edge_nms * 255)

    # thin p2orm w/ edge_prob after nms
    thin_p2orm_w_edge(edge_prob_nms, im_name)


