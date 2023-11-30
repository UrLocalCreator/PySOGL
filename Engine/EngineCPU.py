import math
from Engine.Loader import *

from Engine.Shaders.Default import *


@nb.njit(nogil=True)
def fill_triangle(surface, zbuffer, vertices, tri, z, Res):
    A, B, C = vertices
    if (A[0] * B[1] - B[0] * A[1]) + (B[0] * C[1] - C[0] * B[1]) + (C[0] * A[1] - A[0] * C[1]) >= 0:
        epsilon = 1e-32
        z = 1 / z
        VData = shader(tri)
        d1 = B[1] - C[1]
        d2 = A[0] - C[0]
        d3 = C[0] - B[0]
        d4 = C[1] - A[1]
        d5 = d1 * d2 - d3 * d4 + epsilon
        vt = vertices[vertices[:, 1].argsort()]
        slope1 = (vt[2][0] - vt[0][0]) / (vt[2][1] - vt[0][1] + epsilon)
        slope2 = (vt[1][0] - vt[0][0]) / (vt[1][1] - vt[0][1] + epsilon)
        slope3 = (vt[2][0] - vt[1][0]) / (vt[2][1] - vt[1][1] + epsilon)

        y1 = max(math.ceil(vt[0][1]), 0)
        y2 = min(math.ceil(vt[2][1]), Res[1] - 1)
        for y in nb.prange(y1, y2):
            dy = y - vt[0][1]
            x1 = vt[0][0] + slope1 * dy
            if y < vt[1][1]:
                x2 = vt[0][0] + slope2 * dy
            else:
                x2 = vt[1][0] + slope3 * (y - vt[1][1])
            if x1 > x2:
                x1, x2 = x2, x1
            x1 = max(math.ceil(x1), 0)
            x2 = min(math.ceil(x2), Res[0] - 1)
            for x in nb.prange(x1, x2):
                d6 = x - C[0]
                d7 = y - C[1]
                u = (d1 * d6 + d3 * d7) / d5
                v = (d4 * d6 + d2 * d7) / d5
                w = 1.0 - (u + v)

                zv = z[0] * u + z[1] * v + z[2] * w
                zv = 1 / zv
                if zv < zbuffer[x][y]:
                    zbuffer[x][y] = zv
                    zv = 1 / zv
                    surface[x, y] = fragment(x, y, VData, zv)
    return surface


@nb.njit(nogil=True)
def fill_object(faces, vertices, tris, uvs, surface, zbuffer, Res):
    for i in nb.prange(len(faces)):
        j = faces[i]

        vert1 = vertices[j[0]]
        vert2 = vertices[j[1]]
        vert3 = vertices[j[2]]
        vert = np.zeros((3, 2), dtype=float)
        vert[0] = vert1[0:2]
        vert[1] = vert2[0:2]
        vert[2] = vert3[0:2]
        tri = np.zeros((3, 3), dtype=float)
        tri[0] = tris[j[0]]
        tri[1] = tris[j[1]]
        tri[2] = tris[j[2]]

        z = np.array([vert1[2], vert2[2], vert3[2]])
        surface = fill_triangle(surface, zbuffer, vert, tri, z, Res)
    return surface


@nb.njit(nogil=True)
def translate(vertices, position):
    rad = math.pi / 180
    vertices *= position[6]
    rotated = position[3:6]
    position = position[0:3]
    translated = np.zeros_like(vertices)

    cosX = rotated[0] * rad
    sinX = math.sin(cosX)
    cosX = math.cos(cosX)
    cosY = rotated[1] * rad
    sinY = math.sin(cosY)
    cosY = math.cos(cosY)
    cosZ = rotated[2] * rad
    sinZ = math.sin(cosZ)
    cosZ = math.cos(cosZ)

    for i in nb.prange(len(vertices)):
        x, y, z = vertices[i]
        s = x
        x = y * sinZ + s * cosZ
        y = y * cosZ - s * sinZ
        s = y
        y = z * sinY + s * cosY
        z = z * cosY - s * sinY
        s = x
        x = z * sinX + s * cosX
        z = z * cosX - s * sinX
        x += position[0]
        y += position[1]
        z += position[2]
        translated[i] = np.array([x, y, z])

    return translated


@nb.njit(nogil=True)
def project(vertices, FOV, ResF, Res):
    rad = 180 / math.pi
    FOV = (ResF / 2) / math.tan((FOV / 2) * rad)
    projected = np.empty_like(vertices)
    for i in nb.prange(len(vertices)):
        vertex = vertices[i]
        s = FOV / vertex[2]
        vertex[0] = Res[0] / 2 + vertex[0] * s
        vertex[1] = Res[1] / 2 - vertex[1] * s
        projected[i] = vertex[0:3]

    return projected


def renderCPU(objectn, position, camera, surface, zbuffer, Res, Objects, ObjectData):
    camera = np.asarray(camera)
    
    if objectn in Objects:
        vertices, uvs, faces = ObjectData[Objects.index(objectn)]
    else:
        vertices, uvs, faces = unload_object(load_obj(objectn))
        Objects.append(objectn)
        ObjectData.append([vertices, uvs, faces])
    vertices = translate(vertices, np.asarray(position))
    tris = vertices
    fov = camera[6]
    c = -camera
    c[6] = 1
    vertices = translate(vertices, np.asarray(c))
    ResF = min(Res[0], Res[1])
    projected = project(vertices, fov, ResF, np.asarray(Res))
    surface = fill_object(np.asarray(faces), projected, tris, uvs, surface, zbuffer, np.asarray(Res))
    return Objects, ObjectData, surface
