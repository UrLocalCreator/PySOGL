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


def name(n):
    pygame.display.set_caption(n)


def icon(n):
    pygame.display.set_icon(pygame.image.load(n))


def FPS(otime):
    time = pygame.time.get_ticks()
    dtime = time - otime
    fps = 1 / (dtime / 1000 + 1e-32)
    otime += dtime
    return fps, otime


def check(running):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    pygame.display.flip()
    return running


def render(scene, camera, surface, zbuffer, Res, Objects, ObjectData, light, screen_buffer, SRes, screen):
    screen.fill((0, 0, 0))
    renderCPU(scene, camera, surface, zbuffer, np.array(Res), Objects, ObjectData, light)
    screen_surface = pygame.surfarray.make_surface(screen_buffer)
    screen_surface = pygame.transform.scale(screen_surface, SRes)
    screen.blit(screen_surface, (0, 0))

@nb.njit(nogil=True)
def init_surface(Res, scale):
    Res = np.asarray(Res) * scale
    screen_buffer = np.zeros((Res[0], Res[1], 3), dtype=np.int32)
    zbuffer = np.zeros((Res[0], Res[1])) + 1e32
    return screen_buffer, zbuffer
