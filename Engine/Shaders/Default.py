import random
import numba as nb
import numpy as np
import math

@nb.njit(nogil=True, fastmath=True)
def cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])


@nb.njit(nogil=True, fastmath=True)
def normalize(a):
    return a / (np.sqrt(np.sum(a**2)))


@nb.njit(nogil=True, fastmath=True)
def shader(tri):
    return [normalize(cross(tri[1] - tri[0], tri[2] - tri[0]))]


@nb.njit(nogil=True, fastmath=True)
def pixel_filter(xyz, color, colors, dith):
    colors = 255 / colors
    for i in nb.prange(len(color)):
        color[i] /= colors
        if dith:
            j = color[i]
            seed = 9834579
            x, y, z = xyz
            seed = seed * 315325887822453 + 141861939145533
            x = x * 315325887822453 + 141861939145533
            y = y * 315325887822453 + 141861939145533
            z = z * 315325887822453 + 141861939145533
            c = ((seed * x * y * z) % 999999999999999) / 999999999999999
            color[i] = np.floor(j) if (c > j - np.floor(j)) else np.ceil(j)
        color[i] = np.round(color[i]) * colors
        if color[i] > 255:
            color[i] = 255
    return color


@nb.njit(nogil=True, fastmath=True)
def fragment(xyz, VData, tri, lights):
    color = np.array([255.0, 255.0, 255.0])
    lcolor = np.copy(color)
    color = np.zeros_like(lcolor)
    n = VData[0]
    for i in nb.prange(len(lights)):
        i = lights[i]
        d = i[0] - xyz
        dist = np.sqrt(np.sum((d) ** 2))
        light = i[1] / dist
        check = np.copy(lcolor)
        d = normalize(d)
        d = d[0] * n[0] + d[1] * n[1] + d[2] * n[2]
        for j in nb.prange(len(lcolor)):
            #shader
            diff = light[j] * ((d + 1) / 2)
            spec = 0
            lightr = diff + spec
            check[j] *= lightr
            color[j] += check[j]
    color = pixel_filter(xyz, color, 255, True)
    return color
