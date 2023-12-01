import numpy as np
import numba as nb


def load_obj(file_name):
    data = []

    with open(file_name, 'r') as file:
        vertices = []
        uvs = []
        faces = []

        for line in file:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append(list(map(float, parts[1:])))
            elif parts[0] == 'vt':
                uvs.append(list(map(float, parts[1:])))
            elif parts[0] == 'f':
                face_data = []
                for vert in parts[1:]:
                    vertex_info = vert.split('/')
                    vertex_indices = [int(index) if index != '' else None for index in vertex_info]
                    face_data.append(vertex_indices)
                faces.append(face_data)

        if vertices:
            data.append({'vertices': vertices})
        if uvs:
            data.append({'uvs': uvs})
        if faces:
            data.append({'faces': faces})

    return data


def unload_object(Object):
    vertices = np.empty((0, 3), dtype=float)
    uvs = np.empty((0, 2), dtype=float)
    faces = []
    for section in Object:
        if "vertices" in section:
            vertices = np.vstack((vertices, np.array(section["vertices"])))
        if "uvs" in section:
            uvs = np.vstack((uvs, np.array(section["uvs"])))
        if "faces" in section:
            for face in section["faces"]:
                face_array = []
                for index_info in face:
                    if index_info is not None:
                        vertex_index = int(index_info[0]) - 1
                    else:
                        vertex_index = None
                    face_array.append(vertex_index)
                if len(face_array) > 3:
                    newarr = np.zeros(3)
                    for i in nb.prange(len(face_array)):
                        if i > 2:
                            faces.append(np.array(newarr, dtype=int))
                            newarr[1] = newarr[2]
                            newarr[2] = face_array[i]

                        else:
                            newarr[i] = face_array[i]
                    faces.append(np.array(newarr, dtype=int))
                else:
                    faces.append(np.array(face_array, dtype=int))
    return vertices, uvs, faces
