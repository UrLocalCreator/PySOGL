
import numba as nb
import numpy as np
import math


@nb.njit(nogil=True)
def cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])


@nb.njit(nogil=True)
def normalize(a):
    dist = np.sqrt(np.sum(a**2))
    return a / dist


@nb.njit(nogil=True)
def shader(vertices):
    x, y, z = vertices
    normal = cross(y - x, z - x)
    normal = normalize(normal)
    return [normal]


@nb.njit(nogil=True)
def dither(x, y, color, colors, dith):
    colors = 256 / colors
    color /= colors
    for i in nb.prange(len(color)):
        if dith:
            j = color[i]
            color[i] = np.floor(j) if (np.random.random() > j - np.floor(j)) else np.ceil(j)
        color[i] = np.round(color[i])
    color *= colors
    return color


@nb.njit(nogil=True)
def fragment(x, y, VData, zv):
    normal = (VData[0] + 1) / 2
    normal[2] = 1 - normal[2]
    normal *= 255 * zv
    color = dither(x, y, normal, 256, True)
    return color
