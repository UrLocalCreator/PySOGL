import sys

from Engine.Engine import *
from Engine.Loader import *
from Engine.Process import *

priority()

pygame.init()
name("Loading...")
icon("Icon.png")

screen = pygame.display.set_mode((640, 360), pygame.RESIZABLE)

Objects, ObjectData = [], []
running = True
time = pygame.time.get_ticks()
otime = time

while running:

    lights = []
    scene = []
    RRResolutionX, RRResolutionY = 640, 360

    ResolutionX, ResolutionY, RResolutionX, RResolutionY = resize_window(RRResolutionX, RRResolutionY)
    screen_buffer, zbuffer = init_surface(np.array([RResolutionX, RResolutionY]), 1)
    rad = math.pi/180
    rad *= otime/10

    mx, my = pygame.mouse.get_pos()
    mx = (0.5 - mx / ResolutionX) * 640
    my = (0.5 - my / ResolutionY) * 360

    Camera = [0, 0, 0, 0, 0, 0, 60]
    sun = 1000000/4

    lights.append([[100000, 100000, 0], [sun, sun, sun], [0, 0, 0]])
    lights.append([[-4, 0, -7], [2, 1, 0], [0, 0, 0]])
    lights.append([[10, 0, 3], [0, 1, 2], [0, 0, 0]])

    # scene.append(["Engine/Objects/Ayaka/Ayaka.obj", [0, -1.7, 0.5, mx + 180, -90, 0, 0.1]])
    scene.append(["Engine/Objects/Suzanne/Suzanne.obj", [0, 0, 3, mx + 180,  my + 90, 0, 1]])
    # scene.append(["Engine/Objects/Ayaka/Ayaka.obj", [0, -1, 2.5, mx + 180, -90, 0, 0.1]])
    # scene.append(["Engine/Objects/Suzanne/Suzanne.obj", [0, 0, 3, time / 100, 90, 90, 1]])
    # scene.append(["Engine/Objects/Ayaka/Ayaka.obj", [0, -1, 2.5, 180, -90, 0, 0.1]])

    render(scene, Camera, screen_buffer, zbuffer, [RResolutionX, RResolutionY], Objects, ObjectData, lights, screen_buffer, [ResolutionX, ResolutionY], screen)
    fps, otime = FPS(otime)
    name("SOGL - FPS: " + str(round(fps)))
    running = check(running)

pygame.quit()
sys.exit()
