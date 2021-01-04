# run nms on edge map

import cv2
import numpy as np


def edge_nms(edge):
    """
    :param edge: edge probability, 0~1
    """
    dy, dx = np.gradient(conv_tri(edge, 4))
    _, dxx = np.gradient(dx)
    dyy, dxy = np.gradient(dy)
    orientation = np.arctan2(dyy * np.sign(-dxy) + 1e-5, dxx)
    orientation[orientation < 0] += np.pi

    edge_thin = non_maximum_supr(edge, orientation, 1, 5, 1.02)

    return edge_thin


def non_maximum_supr(edge, ori, r, s, m):
    """
    Non-Maximum Suppression (from https://github.com/ArtanisCV/StructuredForests/ cython version)
    :param edge: original edge map HxW
    :param ori: orientation map HxW
    :param r: radius for nms suppression
    :param s: radius for suppress boundaries
    :param m: multiplier for conservative suppression
    :return: suppressed edge map
    """

    h = edge.shape[0]
    w = edge.shape[1]
    E_arr = np.zeros((h, w), dtype=np.float64)
    E = E_arr
    C = np.cos(ori)
    S = np.sin(ori)

    # suppress edges where edge is stronger in orthogonal direction
    for y in range(0, h):
        for x in range(0, w):
            e = E[y, x] = edge[y, x]
            if e == 0:
                continue

            e *= m
            co = C[y, x]
            si = S[y, x]

            for d in range(-h, h + 1):
                if d != 0:
                    e0 = bilinear_interp(edge, x + d * co, y + d * si)
                    if e < e0:
                        E[y, x] = 0
                        break

    # suppress noisy edge estimates near boundaries
    s = w / 2 if s > w / 2 else s
    s = h / 2 if s > h / 2 else s

    for x in range(0, s):
        for y in range(0, h):
            E[y, x] *= x / s
            E[y, w - 1 - x] *= x / s

    for x in range(0, h):
        for y in range(0, s):
            E[y, x] *= y / s
            E[h - 1 - y, x] *= y / s

    return E_arr


def conv_tri(src, radius):
    """
    Image convolution with a triangle filter.
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
        kernel = list(range(1, radius + 1)) + [radius + 1] + list(range(radius, 0, -1))
        kernel = np.asarray(kernel, dtype=np.float64) / (radius + 1) ** 2
        return cv2.sepFilter2D(src, ddepth=-1, kernelX=kernel, kernelY=kernel, borderType=cv2.BORDER_REFLECT)


def bilinear_interp(img, x, y):
    """
    Return img[y, x] via bilinear interpolation
    """
    h, w = img.shape[0:2]

    if x < 0:
        x = 0
    elif x > w - 1.001:
        x = w - 1.001

    if y < 0:
        y = 0
    elif y > h - 1.001:
        y = h - 1.001

    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1
    dx0 = x - x0
    dy0 = y - y0
    dx1 = 1 - dx0
    dy1 = 1 - dy0

    return img[y0, x0]*dx1*dy1 + img[y0, x1]*dx0*dy1 + img[y1, x0]*dx1*dy0 + img[y1, x1]*dx0*dy0


if __name__ == '__main__':
    f_name = 'corridor_02-rgb_lab_v_g.png'
    out_name = 'corridor_02-rgb_lab_v_g_nms.png'
    edge_prob = cv2.imread(f_name, -1) / 255.  # 0~1
    edge_prob_nms = edge_nms(edge_prob) * 255
    cv2.imwrite(out_name, edge_prob_nms.astype(np.uint8))


