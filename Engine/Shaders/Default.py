import random
import numba as nb
import numpy as np
import math

@nb.njit(nogil=True, fastmath=True)
def cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

@nb.njit(nogil=True, fastmath=True)
def normalize(a):
    norm_a = np.sqrt(np.sum(a**2))
    return a / norm_a if norm_a != 0 else a

@nb.njit(nogil=True, fastmath=True)
def shader(tri):
    return [normalize(cross(tri[1] - tri[0], tri[2] - tri[0]))]


@nb.njit(nogil=True, fastmath=True)
def pixel_filter(xyz, color, colors, dith):
    colors = 255 / colors
    if colors > 1:
        color /= colors
    if dith:
        for i in nb.prange(len(color)):
            j = color[i]
            seed = 9834579
            x, y, z = xyz
            seed = seed * 315325887822453 + 141861939145533
            x = x * 315325887822453 + 141861939145533
            y = y * 315325887822453 + 141861939145533
            z = z * 315325887822453 + 141861939145533
            c = ((seed * x * y * z) % 999999999999999) / 999999999999999
            color[i] = np.floor(j) if (c > j - np.floor(j)) else np.ceil(j)

    color = np.round(color)

    if colors > 1:
        color = color * colors

    color = np.minimum(255, color)
    return color


@nb.njit(nogil=True, fastmath=True)
def fragment(xyz, VData, lights, colors, dith):
    color = np.array([255.0, 255.0, 255.0])
    lcolor = np.copy(color)
    color = np.zeros_like(lcolor)
    n = VData[0]

    for i in nb.prange(len(lights)):
        light = lights[i]
        d = light[0] - xyz
        dist = np.sqrt(np.sum(d ** 2))
        light_intensity = light[1] / dist
        check = lcolor.copy()

        d = normalize(d)
        d = d[0] * n[0] + d[1] * n[1] + d[2] * n[2]

        for j in nb.prange(len(lcolor)):
            # shader
            diff = light_intensity[j] * ((d + 1) / 2)
            spec = 0
            lightr = diff + spec
            check[j] *= lightr
            color[j] += check[j]

    # Dithering
    colors = 255 / colors
    if colors > 1:
        color /= colors
    if dith:
        for i in nb.prange(len(color)):
            j = color[i]
            seed = 9834579
            x, y, z = xyz
            seed = seed * 315325887822453 + 141861939145533
            x = x * 315325887822453 + 141861939145533
            y = y * 315325887822453 + 141861939145533
            z = z * 315325887822453 + 141861939145533
            c = ((seed * x * y * z) % 999999999999999) / 999999999999999
            color[i] = np.floor(j) if (c > j - np.floor(j)) else np.ceil(j)
    color = np.round(color)
    if colors > 1:
        color = color * colors

    color = np.minimum(255, color)
    return color
