import numba as nb


@nb.njit(nogil=True)
def noise(xyz, seed):
    x, y, z = xyz
    seed = seed * 315325887822453 + 141861939145533
    x = x * 315325887822453 + 141861939145533
    y = y * 315325887822453 + 141861939145533
    z = z * 315325887822453 + 141861939145533
    return (seed * x * y * z) % 999999999999999

