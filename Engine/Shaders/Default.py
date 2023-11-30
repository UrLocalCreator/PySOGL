import random
import numba as nb
import numpy as np
import math
from Engine.Noise import *


@nb.njit(nogil=True)
def cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])


@nb.njit(nogil=True)
def normalize(a):
    dist = np.sqrt(np.sum(a**2))
    return a / dist


@nb.njit(nogil=True)
def shader(tri):
    normal = normalize(cross(tri[1] - tri[0], tri[2] - tri[0]))
    return [normal]


@nb.njit(nogil=True)
def pixel_filter(xyz, color, colors, dith):
    colors = 255 / colors
    color /= colors
    for i in nb.prange(len(color)):
        if dith:
            j = color[i]
            c = noise(xyz, 9834579) / 999999999999999
            #c = random.random()
            color[i] = np.floor(j) if (c > j - np.floor(j)) else np.ceil(j)
        color[i] = np.round(color[i]) * colors
        if color[i] > 255:
            color[i] = 255
    return color


@nb.njit(nogil=True)
def fragment(xyz, VData, tri):

    normal = (VData[0] + 1) / 2
    normal[2] = 1 - normal[2]
    normal *= 255 / xyz[2]
    color = normal
    color = pixel_filter(xyz, color, 32, True)
    return color
