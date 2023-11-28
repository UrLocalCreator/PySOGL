import numba as nb
import numpy as np


@nb.njit
def shader(vertices):
    
    return(vertices)


@nb.njit
def fragment(U, V, texture):
    
    return(texture)