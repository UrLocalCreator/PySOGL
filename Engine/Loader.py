import numpy as np
import numba as nb

# @nb.njit
def load_obj(file_name):
    vertices = np.empty((0, 3), dtype=np.float64)
    uvs = np.empty((0, 2), dtype=np.float64)
    faces = np.empty((0, 3), dtype=np.int64)

    with open(file_name, 'r') as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertex = np.array(parts[1:], dtype=np.float64)
                vertices = np.vstack((vertices, vertex))
            elif parts[0] == 'vt':
                uv = np.array(parts[1:], dtype=np.float64)
                uvs = np.vstack((uvs, uv))
            elif parts[0] == 'f':
                face_data = []
                for vert in parts[1:]:
                    vertex_info = vert.split('/')
                    vertex_indices = [int(index) if index != '' else -1 for index in vertex_info]
                    face_data.append(vertex_indices)

                if len(face_data) > 3:
                    newarr = np.zeros((1, 3), dtype=np.int64)
                    for i in range(len(face_data)):
                        if i > 2:
                            faces = np.vstack((faces, newarr))
                            newarr[0, 1] = newarr[0, 2]
                            newarr[0, 2] = face_data[i][0] - 1 if face_data[i][0] != -1 else -1
                        else:
                            newarr[0, i] = face_data[i][0] - 1 if face_data[i][0] != -1 else -1
                    faces = np.vstack((faces, newarr))
                else:
                    faces = np.vstack((faces, np.array([index[0] - 1 if index[0] != -1 else -1 for index in face_data], dtype=np.int64)))

    return vertices, uvs, faces
