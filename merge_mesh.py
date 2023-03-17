import os
import sys
sys.path.append(os.getcwd())
import yaml
import numpy as np
import open3d as o3d


if __name__ == "__main__":
    # assign path 
    city = 'pittsburgh'
    data_path = os.path.join(os.getcwd(), 'datasets', city)
    obj_files= os.listdir(os.path.join(data_path, (city + '_cropped')))

    # load color map
    with open('color_map.yaml','r') as f:
        color = yaml.load(f, Loader=yaml.FullLoader)

    mesh = o3d.io.read_triangle_mesh(os.path.join(data_path, city+'_cropped', obj_files[0]))
    triangles = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)
    vertices = np.asarray(mesh.vertices)
    class_name = obj_files[0].split('.')[1]
    class_name = class_name.replace('osm_','')
    class_id = color['labels'][class_name]
    mesh.paint_uniform_color(np.divide(color['color_map'][class_id], 255))
    colors = np.asarray(mesh.vertex_colors)
    shift = vertices.shape[0]
   
    for i  in range(1, len(obj_files)):
        mesh = o3d.io.read_triangle_mesh(os.path.join(data_path, city+'_cropped', obj_files[i]))
        triangles_ = np.asarray(mesh.triangles) + shift
        normals_ = np.asarray(mesh.vertex_normals)
        vertices_ = np.asarray(mesh.vertices)
        class_name = obj_files[i].split('.')[1]
        class_name = class_name.replace('osm_','')
        class_id = color['labels'][class_name]
        mesh.paint_uniform_color(np.divide(color['color_map'][class_id], 255))
        colors_ = np.asarray(mesh.vertex_colors)
       
        triangles = np.concatenate((triangles,triangles_), axis=0)
        normals = np.concatenate((normals,normals_), axis=0)
        vertices = np.concatenate((vertices,vertices_), axis=0)
        colors = np.concatenate((colors,colors_), axis=0)
        shift = vertices.shape[0]

    mesh_ = o3d.geometry.TriangleMesh()
    mesh_.triangles = o3d.utility.Vector3iVector(triangles)
    mesh_.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh_.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(os.path.join(data_path, (city + '.obj')), mesh_)
    o3d.visualization.draw_geometries([mesh_], window_name="mesh")
    

