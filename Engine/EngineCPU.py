from Engine.Loader import *

from Engine.Shaders.Default import *


@nb.njit(nogil=True, parallel=True, fastmath=True)
def fill_object(faces, vertices, tris, cam, uvs, surface, zbuffer, Res, lights):
    for i in nb.prange(len(faces)):
        j = faces[i]
        vert = vertices[j]
        min_x = max(min(vert[:, 0]), 0)
        max_x = min(max(vert[:, 0]), Res[0] / 2 - 1)
        min_y = max(min(vert[:, 1]), 0)
        max_y = min(max(vert[:, 1]), Res[1] / 2 - 1)
        if max_x >= 0 and min_x < Res[0] and max_y >= 0 and min_y < Res[1]:
            vert1 = vert[0]
            vert2 = vert[1]
            vert3 = vert[2]

            x1, y1, z1 = vert1
            x2, y2, z2 = vert2
            x3, y3, z3 = vert3

            if z1 > 0 and z2 > 0 and z3 > 0:

                z = np.array([z1, z2, z3])

                # Simplified condition for triangle orientation check
                orientation = (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) >= 0

                if orientation:

                    vy = vert[vert[:, 1].argsort()]
                    tri1 = tris[j[0]]
                    tri2 = tris[j[1]]
                    tri3 = tris[j[2]]

                    epsilon = 1e-32
                    z = 1 / z

                    d1 = y2 - y3
                    d2 = x1 - x3
                    d3 = x3 - x2
                    d4 = y3 - y1
                    d5 = d1 * d2 - d3 * d4 + epsilon

                    slope1 = (vy[2, 0] - vy[0, 0]) / (vy[2, 1] - vy[0, 1] + epsilon)
                    slope2 = (vy[1, 0] - vy[0, 0]) / (vy[1, 1] - vy[0, 1] + epsilon)
                    slope3 = (vy[2, 0] - vy[1, 0]) / (vy[2, 1] - vy[1, 1] + epsilon)

                    VData = shader([tri1, tri2, tri3])

                    y1 = max(math.ceil(vy[0, 1]), 0)
                    y2 = min(math.ceil(vy[2, 1]), Res[1] - 1)

                    for y in nb.prange(y1, y2):
                        dy = y - vy[0, 1]
                        x1 = vy[0, 0] + slope1 * dy
                        if y < vy[1, 1]:
                            x2 = vy[0, 0] + slope2 * dy
                        else:
                            x2 = vy[1, 0] + slope3 * (y - vy[1, 1])

                        if x1 > x2:
                            x1, x2 = x2, x1

                        x1 = max(math.ceil(x1), 0)
                        x2 = min(math.ceil(x2), Res[0] - 1)

                        for x in nb.prange(x1, x2):
                            d6 = x - x3
                            d7 = y - y3
                            u = (d1 * d6 + d3 * d7) / d5
                            v = (d4 * d6 + d2 * d7) / d5
                            w = 1.0 - (u + v)

                            zv = z[0] * u + z[1] * v + z[2] * w
                            zv = 1 / zv

                            if zv < zbuffer[x, y]:
                                zbuffer[x, y] = zv

                                xyz = tri1 * u + tri2 * v + tri3 * w
                                surface[x, y] = fragment(xyz, cam, [u, v, w], VData, lights, 255, True, 0.1)

    return surface


@nb.njit(nogil=True, fastmath=True)
def translate(vertices, position):
    rad = math.pi / 180
    cosX, sinX = math.cos(position[3] * rad), math.sin(position[3] * rad)
    cosY, sinY = math.cos(position[4] * rad), math.sin(position[4] * rad)
    cosZ, sinZ = math.cos(position[5] * rad), math.sin(position[5] * rad)

    scale_factor = position[6]
    scaled_vertices = vertices * scale_factor

    x, y, z = scaled_vertices.T

    y, z = z * sinY + y * cosY, z * cosY - y * sinY
    x, z = z * sinX + x * cosX, z * cosX - x * sinX
    x, y = y * sinZ + x * cosZ, y * cosZ - x * sinZ

    translated = np.column_stack((x, y, z))

    translated += position[:3]

    return translated


@nb.njit(nogil=True, fastmath=True)
def project(vertices, FOV, ResF, Res):
    FOV = (ResF / 2) / np.tan((FOV / 2) * (180 / np.pi))

    # Extracting x, y, z columns for better cache efficiency
    x, y, z = vertices.T

    # Vectorized computation
    s = FOV / z
    x_proj = Res[0] / 2 + x * s
    y_proj = Res[1] / 2 - y * s

    return np.column_stack((x_proj, y_proj, z))


def renderCPU(scene, camera, surface, zbuffer, Res, Objects, ObjectData, lights):
    lights = np.asarray(lights, dtype=np.float32)
    camera = np.asarray(camera, dtype=np.float32, order='C')
    for i in nb.prange(len(scene)):
        objectn = scene[i][0]
        position = scene[i][1]

        if objectn in Objects:
            vertices, uvs, faces = ObjectData[Objects.index(objectn)]
        else:
            vertices, uvs, faces = load_obj(objectn)
            Objects.append(objectn)
            ObjectData.append([vertices, uvs, faces])

        vertices = translate(vertices, np.asarray(position))
        tris = vertices
        fov = camera[6]
        cam = camera[0:3]
        c = -camera
        c[6] = 1
        vertices = translate(vertices, c)
        ResF = min(Res[0], Res[1])
        projected = project(vertices, fov, ResF, np.asarray(Res))
        fill_object(np.asarray(faces), projected, tris, cam, uvs, surface, zbuffer, np.asarray(Res), np.asarray(lights))
