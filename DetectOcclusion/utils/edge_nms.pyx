# edge nms code using cython to boost the speed

import math
import numpy as N
cimport numpy as N

ctypedef N.int32_t C_INT32
ctypedef N.float64_t C_FLOAT64


def non_maximum_supr(double[:, :] edge, double[:, :] ori, int r, int s, double m):
    """
    Non-Maximum Suppression
    :param edge: original edge map
    :param ori: orientation map
    :param r: radius for nms suppression
    :param s: radius for suppress boundaries
    :param m: multiplier for conservative suppression
    :return: suppressed edge map
    """

    cdef int h = edge.shape[0], w = edge.shape[1], x, y, d
    cdef double e, e0, co, si
    cdef N.ndarray[C_FLOAT64, ndim=2] E_arr = N.zeros((h, w), dtype=N.float64)
    cdef double[:, :] E = E_arr
    cdef double[:, :] C = N.cos(ori), S = N.sin(ori)

    with nogil:
        # suppress edges where edge is stronger in orthogonal direction
        for y from 0 <= y < h:
            for x from 0 <= x < w:
                e = E[y, x] = edge[y, x]
                if e == 0:
                    continue

                e *= m
                co = C[y, x]
                si = S[y, x]

                for d from -r <= d <= r:
                    if d != 0:
                        e0 = bilinear_interp(edge, x + d * co, y + d * si)
                        if e < e0:
                            E[y, x] = 0
                            break

        # suppress noisy edge estimates near boundaries
        s = w / 2 if s > w / 2 else s
        s = h / 2 if s > h / 2 else s

        for x from 0 <= x < s:
            for y from 0 <= y < h:
                edge[y, x] *= x / <double>s
                edge[y, w - 1 - x] *= x / <double>s

        for x from 0 <= x < w:
            for y from 0 <= y < s:
                edge[y, x] *= y / <double>s
                edge[h - 1 - y, x] *= y / <double>s

    return E_arr


cdef inline float bilinear_interp(double[:, :] img, float x, float y) nogil:
    """
    Return img[y, x] via bilinear interpolation
    """

    cdef int h = img.shape[0], w = img.shape[1]

    if x < 0:
        x = 0
    elif x > w - 1.001:
        x = w - 1.001

    if y < 0:
        y = 0
    elif y > h - 1.001:
        y = h - 1.001

    cdef int x0 = int(x), y0 = int(y), x1 = x0 + 1, y1 = y0 + 1
    cdef double dx0 = x - x0, dy0 = y - y0, dx1 = 1 - dx0, dy1 = 1 - dy0

    return img[y0, x0] * dx1 * dy1 + img[y0, x1] * dx0 * dy1 + \
           img[y1, x0] * dx1 * dy0 + img[y1, x1] * dx0 * dy0