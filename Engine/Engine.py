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


def render(objectn, position, camera, surface, zbuffer, Res, Objects, ObjectData):
    Objects, ObjectData, surface = renderCPU(objectn, position, camera, surface, zbuffer, np.array(Res), Objects, ObjectData)
    return Objects, ObjectData, surface


@nb.njit
def init_surface(Res, scale):

    Res = np.asarray(Res) * scale
    screen_buffer = np.zeros((Res[0], Res[1], 3), dtype=np.int32)
    zbuffer = np.zeros((Res[0], Res[1])) + 1e32
    return screen_buffer, zbuffer


def FXAA(surface):
    modified_surface = surface
    lum = surface
    # for y in nb.prange(len(surface)):
    #     for x in nb.prange(len(surface[y])):
    #         lum[y][x][0] = ((np.dot(surface[y][x], (0.299, 0.587, 0.114)) + 1) / 2) * 255
    #         modified_surface[y][x][0:3] = lum[y][x][0]
            
    return modified_surface