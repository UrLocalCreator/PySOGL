import pygame
from Engine.EngineCPU import *


def resize_window(RRResolutionX, RRResolutionY):
    info = pygame.display.Info()
    ResolutionX, ResolutionY = info.current_w, info.current_h
    Res = RRResolutionX / ResolutionX
    if Res > RRResolutionY / ResolutionY:
        Res = RRResolutionY / ResolutionY
    RResolutionX, RResolutionY = int(ResolutionX * Res), int(ResolutionY * Res)
    return ResolutionX, ResolutionY, RResolutionX, RResolutionY


def render(objectn, position, camera, surface, zbuffer, Res, Objects, ObjectData, light):
    Objects, ObjectData, surface = renderCPU(objectn, position, camera, surface, zbuffer, np.array(Res), Objects, ObjectData, light)
    return Objects, ObjectData, surface


@nb.njit(nogil=True)
def init_surface(Res, scale):

    Res = np.asarray(Res) * scale
    screen_buffer = np.zeros((Res[0], Res[1], 3), dtype=np.int32)
    zbuffer = np.zeros((Res[0], Res[1])) + 1e32
    return screen_buffer, zbuffer


@nb.njit
def anti_aliasing(surface):
    return surface

