import random

import numba as nb
import numpy as np
import math


@nb.njit(nogil=True)
def cross(a, b):
    norm = np.zeros_like(a)
    norm[0] = a[1] * b[2] - a[2] * b[1]
    norm[1] = a[2] * b[0] - a[0] * b[2]
    norm[2] = a[0] * b[1] - a[1] * b[0]
    return norm


@nb.njit(nogil=True)
def normalize(a):
    dist = np.sqrt(np.sum(a**2))
    a /= dist
    return a


@nb.njit(nogil=True)
def shader(vertices):
    x, y, z = vertices
    normal = cross(y - x, z - x)
    normalize(normal)
    return [normal]


@nb.njit(nogil=True)
def dither(color):
    for i in nb.prange(len(color)):
        j = color[i]
        color[i] = math.floor(j) if (np.random.random() > j - math.floor(j)) else math.ceil(j)
    return color


@nb.njit(nogil=True)
def fragment(VData, zv):
    normal = VData[0] + 1
    normal /= 2
    normal[2] = 1 - normal[2]
    normal *= 255 * zv
    color = dither(normal)
    return color
