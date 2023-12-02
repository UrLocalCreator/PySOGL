import sys

from Engine.Engine import *
from Engine.Loader import *
from Engine.Process import *

priority()

pygame.init()
img = pygame.image.load('Icon.png')
pygame.display.set_caption("Loading...")
pygame.display.set_icon(img)
screen = pygame.display.set_mode((640, 360), pygame.RESIZABLE)
Objects, ObjectData = [], []
running = True
time = pygame.time.get_ticks()
otime = time
print(pygame.display.Info())
while running:
    lights = []
    Camera = [0, 0, 0, 0, 0, 0, 60]
    RRResolutionX, RRResolutionY = 640, 360
    #RRResolutionX, RRResolutionY = np.array([160, 90])
    ResolutionX, ResolutionY, RResolutionX, RResolutionY = resize_window(RRResolutionX, RRResolutionY)
    screen.fill((0, 0, 0))

    screen_buffer, zbuffer = init_surface(np.array([RResolutionX, RResolutionY]), 1)
    rad = math.pi/180
    rad *= otime/10
    lights.append([[-4, 0, -7 + 3 * math.cos(rad)], [2, 1, 0]])
    lights.append([[10 + 3 * math.sin(rad), 0, 3], [0, 1, 2]])

    mx, my = pygame.mouse.get_pos()
    mx = (0.5 - mx / ResolutionX) * 640
    my = (0.5 - my / ResolutionY) * 360
    #mx, my = 0, 0
    Objects, ObjectData, screen_buffer = render("Engine/Objects/Suzanne/Suzanne.obj", [0, 0, 3, mx + 180,  my + 90, 0, 1], Camera, screen_buffer, zbuffer, [RResolutionX, RResolutionY], Objects, ObjectData, lights)
    #Objects, ObjectData, screen_buffer = render("Engine/Objects/Ayaka/Ayaka.obj", [0, -10, 20, mx + 180, -90, 0, 1], Camera, screen_buffer, zbuffer, [RResolutionX, RResolutionY], Objects, ObjectData, lights)
    #Objects, ObjectData, screen_buffer = render("Engine/Objects/Suzanne/Suzanne.obj", [0, 0, 3, time / 100, 90, 90, 1], Camera, screen_buffer, zbuffer, [RResolutionX, RResolutionY], Objects, ObjectData, lights)
    #Objects, ObjectData, screen_buffer = render("Engine/Objects/Ayaka/Ayaka.obj", [0, -10, 20, time / 100, -90, 90, 1], Camera, screen_buffer, zbuffer, [RResolutionX, RResolutionY], Objects, ObjectData, lights)

    screen_buffer = anti_aliasing(screen_buffer)
    screen_surface = pygame.surfarray.make_surface(screen_buffer)
    screen_surface = pygame.transform.scale(screen_surface, (ResolutionX, ResolutionY))
    screen.blit(screen_surface, (0, 0))

    fps, otime = FPS(otime)
    name("SOGL - FPS: " + str(round(fps)))
    running = check(running)

pygame.quit()
sys.exit()
