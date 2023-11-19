import sys

import pygame

from Engine.Engine import *
from Engine.Loader import *
from Engine.Process import *

lowpriority()

pygame.init()
img = pygame.image.load('Icon.png')
pygame.display.set_caption("Loading...")
pygame.display.set_icon(img)
screen = pygame.display.set_mode((480, 360), pygame.RESIZABLE)
Objects, ObjectData = [], []
running = True
time = pygame.time.get_ticks()
otime = time
print(pygame.display.Info())
while running:
    Camera = [0, 0, 0, 0, 0, 0, 60]
    RRResolutionX, RRResolutionY = 1920, 1080
    ResolutionX, ResolutionY, RResolutionX, RResolutionY = resize_window(RRResolutionX, RRResolutionY)
    screen.fill((0, 0, 0))

    screen_buffer, zbuffer = init_surface(np.array([RResolutionX, RResolutionY]), 1)
    
    #Objects, ObjectData, screen_buffer = render("Engine/Objects/Ayaka/Ayaka.obj", [0, -10, 20, time / 100, -90, 90, 1], Camera, screen_buffer, zbuffer, [RResolutionX, RResolutionY], Objects, ObjectData)
    Objects, ObjectData, screen_buffer = render("Engine/Objects/Suzanne/Suzanne.obj", [0, 0, 3, time / 100, 90, 90, 1], Camera, screen_buffer, zbuffer, [RResolutionX, RResolutionY], Objects, ObjectData)

    #screen_buffer = FXAA(screen_buffer)
    screen_surface = pygame.surfarray.make_surface(screen_buffer)
    screen_surface = pygame.transform.scale(screen_surface, (ResolutionX, ResolutionY))
    screen.blit(screen_surface, (0, 0))
    pygame.display.flip()
    time = pygame.time.get_ticks()
    fps = 1 / ((time - otime) / 1000 + 1e-32)
    otime = time
    pygame.display.set_caption("SOGL - FPS: " + str(round(fps)))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
sys.exit()
