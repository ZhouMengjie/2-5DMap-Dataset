import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import open3d as o3d
from experiments.visualizer import visualize_pcd

def resample_mesh(mesh_cad, density=1):
    '''
    https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
    Samples point cloud on the surface of the model defined as vectices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.

    param mesh_cad: low-polygon triangle mesh in o3d.geometry.TriangleMesh
    param density: density of the point cloud per unit area
    param return_numpy: return numpy format or open3d pointcloud format
    return resampled point cloud

    Reference :
      [1] Barycentric coordinate system
      \begin{align}
        P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
      \end{align}
    '''
    faces = np.array(mesh_cad.triangles).astype(int)
    vertices = np.array(mesh_cad.vertices)

    vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                         vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
    face_areas = np.sqrt(np.sum(vec_cross**2, 1))

    n_samples = (np.sum(face_areas) * density).astype(int)

    n_samples_per_face = np.ceil(density * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc:acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]

    P = (1 - np.sqrt(r[:, 0:1])) * A + \
        np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + \
        np.sqrt(r[:, 0:1]) * r[:, 1:] * C

    return P

if __name__ == "__main__":
    area = 'manhattan.osm_water'
    path = os.path.join('datasets/manhattan/manhattan_cropped', area+'.obj')
    mesh = o3d.io.read_triangle_mesh(path)
    coords = resample_mesh(mesh, density=0.1)
    visualize_pcd(coords, None, type='npy')

    path = os.path.join('datasets/manhattan/manhattan_pcd', area+'.pcd')
    pcd = o3d.io.read_point_cloud(path)
    visualize_pcd(pcd, None, type='pcd')

