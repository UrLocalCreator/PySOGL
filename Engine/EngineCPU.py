from Engine.Loader import *


@nb.njit(nogil=True, parallel=True, fastmath=True)
def fill_object(faces, vertices, tris, cam, uvs, surface, zbuffer, Res, lights, shader, ShaderS):
    R = Res - 1
    # for i in nb.prange(len(faces)):
    #     vnormals = 0
    for i in nb.prange(len(faces)):
        j = faces[i]
        vert = vertices[j]
        min_x = max(min(vert[:, 0]), 0)
        max_x = min(max(vert[:, 0]), R[0])
        min_y = max(min(vert[:, 1]), 0)
        max_y = min(max(vert[:, 1]), R[1])
        if max_x >= 0 and min_x < R[0] and max_y >= 0 and min_y < R[1]:
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

                    #Shader Code num1
                    a = tri2 - tri1
                    b = tri3 - tri1
                    cr = np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])
                    VData = [(cr / np.sqrt(np.sum(cr ** 2)))]

                    y1 = max(np.ceil(vy[0, 1]), 0)
                    y2 = min(np.ceil(vy[2, 1]), Res[1] - 1)

                    for y in nb.prange(y1, y2):
                        dy = y - vy[0, 1]
                        x1 = vy[0, 0] + slope1 * dy
                        if y < vy[1, 1]:
                            x2 = vy[0, 0] + slope2 * dy
                        else:
                            x2 = vy[1, 0] + slope3 * (y - vy[1, 1])

                        if x1 > x2:
                            x1, x2 = x2, x1

                        x1 = max(np.ceil(x1), 0)
                        x2 = min(np.ceil(x2), Res[0] - 1)

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
                                #Shader Code num2
                                if shader == "Default":
                                    xyz = tri1 * u + tri2 * v + tri3 * w

                                    colors = ShaderS[0]
                                    dith = ShaderS[1] == 1
                                    diffuse = ShaderS[2]
                                    lcolor = np.asarray([255.0, 255.0, 255.0])
                                    n = VData[0]
                                    color = np.zeros_like(lcolor)
                                    cr = cam - xyz
                                    d_norm = cr / np.sqrt(np.sum(cr ** 2))
                                    for j in nb.prange(len(lights)):
                                        light = lights[j]
                                        d = light[0] - xyz
                                        ds = np.sqrt(np.sum(d ** 2))
                                        dist = (ds / light[3][0]) ** (1 + light[3][1])

                                        light_intensity = light[1] / dist
                                        light_intensity *= lcolor
                                        if round(max(light_intensity)) > 0:
                                            d = d / ds
                                            diff = ((d[0] * n[0] + d[1] * n[1] + d[2] * n[2] + 1) / 2)
                                            cr = d_norm + d
                                            v = cr / np.sqrt(np.sum(cr ** 2))
                                            spec = max(0, (v[0] * n[0] + v[1] * n[1] + v[2] * n[2])) ** (1 / (diffuse ** 2))

                                            light_intensity *= diff + spec
                                            color += light_intensity

                                    # Dithering
                                    colors = 255 / colors
                                    if colors > 1:
                                        color /= colors
                                    if dith:
                                        for k in nb.prange(len(color)):
                                            o = color[k]
                                            seed = 9834579
                                            seed = seed * 315325887822453 + 141861939145533
                                            xyz = xyz * 315325887822453 + 141861939145533
                                            c = ((seed * xyz[0] * xyz[1] * xyz[2]) % 999999999999999) / 999999999999999
                                            color[k] = np.floor(o) if (c > o - np.floor(o)) else np.ceil(o)
                                    else:
                                        color = np.round(color)
                                    if colors > 1:
                                        color = color * colors

                                    color = np.minimum(255, color)
                                    surface[x, y] = color
                                else:
                                    surface[x, y] = (u * 255, v * 255, w * 255)

    return surface


@nb.njit(nogil=True, fastmath=True)
def translate(vertices, position):
    pos = position[0]
    rot = position[1]
    scale_factor = position[2][0]
    rad = np.pi / 180
    cosX, sinX = np.cos(rot[0] * rad), np.sin(rot[0] * rad)
    cosY, sinY = np.cos(rot[1] * rad), np.sin(rot[1] * rad)
    cosZ, sinZ = np.cos(rot[2] * rad), np.sin(rot[2] * rad)

    scaled_vertices = vertices * scale_factor

    x, y, z = scaled_vertices.T

    y, z = z * sinY + y * cosY, z * cosY - y * sinY
    x, z = z * sinX + x * cosX, z * cosX - x * sinX
    x, y = y * sinZ + x * cosZ, y * cosZ - x * sinZ

    translated = np.column_stack((x, y, z))

    translated += pos

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
        fov = camera[2][0]
        cam = camera[0]
        c = -camera
        c[2][0] = 1
        vertices = translate(vertices, c)
        ResF = min(Res[0], Res[1])
        projected = project(vertices, fov, ResF, np.asarray(Res))
        fill_object(np.asarray(faces), projected, tris, cam, uvs, surface, zbuffer, np.asarray(Res), np.asarray(lights),"Default", np.array([255, 1, 0.2]))
