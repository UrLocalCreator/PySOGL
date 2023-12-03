import numpy as np
import numba as nb

@nb.njit
def load_obj(file_name):
    vertices = []
    uvs = []
    faces = []

    with open(file_name, 'r') as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertex = np.array(parts[1:], dtype=float)
                vertices.append(vertex)
            elif parts[0] == 'vt':
                uv = np.array(parts[1:], dtype=float)
                uvs.append(uv)
            elif parts[0] == 'f':
                face_data = []
                for vert in parts[1:]:
                    vertex_info = vert.split('/')
                    vertex_indices = [int(index) if index != '' else None for index in vertex_info]
                    face_data.append(vertex_indices)

                if len(face_data) > 3:
                    newarr = np.zeros(3, dtype=int)
                    for i in range(len(face_data)):
                        if i > 2:
                            faces.append(newarr.copy())
                            newarr[1] = newarr[2]
                            newarr[2] = face_data[i][0] - 1 if face_data[i] is not None else -1
                        else:
                            newarr[i] = face_data[i][0] - 1 if face_data[i] is not None else -1
                    faces.append(newarr.copy())
                else:
                    faces.append(
                        np.array([index[0] - 1 if index is not None else -1 for index in face_data], dtype=int))

    return np.array(vertices), np.array(uvs), np.array(faces)

