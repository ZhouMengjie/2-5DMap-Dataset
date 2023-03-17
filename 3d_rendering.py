import os
import sys
sys.path.append(os.getcwd())
import yaml
import pandas as pd
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from openstreetmap import cropping
import copy
from PIL import Image
from experiments.visualizer import visualize_pcd



if __name__ == "__main__":
    # # assign path 
    # city = 'pittsburgh'
    # data_path = os.path.join(os.getcwd(), 'datasets', city)
    # obj_files= os.listdir(os.path.join(data_path, (city + '_cropped')))

    # # load color map
    # with open('color_map.yaml','r') as f:
    #     color = yaml.load(f, Loader=yaml.FullLoader)
   
    # render = rendering.OffscreenRenderer(256, 256)
    # for i  in range(len(obj_files)):
    #     mesh = o3d.io.read_triangle_mesh(os.path.join(data_path, city+'_cropped', obj_files[i]))
    #     class_name = obj_files[i].split('.')[1]
    #     class_name = class_name.replace('osm_','')
    #     class_id = color['labels'][class_name]
    #     c = np.divide(color['color_map'][class_id], 255)
    #     mcolor = rendering.MaterialRecord()
    #     mcolor.base_color = np.concatenate((c,[1]), axis=0)
    #     mcolor.shader = 'defaultLit'
    #     render.scene.add_geometry(class_name, mesh, mcolor)

   
    # crop mesh
    data_path = 'datasets'
    area = 'unionsquare5kU'
    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_xy.csv')), sep=',', header=None)
    info = data.values
    idx = 1
    panoid = info[idx][0]
    center_xy = [info[idx][1], info[idx][2]]
    city = info[idx][4]
    
    mesh = o3d.io.read_triangle_mesh(os.path.join(data_path, city, (city + '.obj')))
    (mesh_indices, vertex_indices) = cropping.crop_mesh_v2(center_xy, mesh)  
    meshc = copy.deepcopy(mesh)
    meshc.triangles = o3d.utility.Vector3iVector(
        np.asarray(mesh.triangles)[mesh_indices, :]) 
    # normalize
    pc = np.array(meshc.vertices)
    pc = pc - [center_xy[0],0,center_xy[1]]
    meshc.vertices = o3d.utility.Vector3dVector(
        np.asarray(pc)) 
    meshc.compute_vertex_normals()
    visualize_pcd(meshc.vertices, None, 'npy')

    # o3d.visualization.draw_geometries([meshc], window_name="mesh")
    # lookat：相机的主视方向向量
    # up：相机的俯视方向向量
    # front：相机的前视方向向量
    # zoom：相机的焦距
    org = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15,origin=[0,0,0])
    o3d.visualization.draw_geometries([meshc,org],zoom=0.03412,
                                  front=[0, 1, 0],
                                  lookat=[0, 0, 0],
                                  up=[1, 0, 0])
  
    # 2D map
    data = pd.read_csv(os.path.join('datasets', 'csv', ( area+ '_set.csv')), sep=',', header=None)
    info = data.values   
    global_idx = info[idx][1] # for train only
    city = info[idx][0]
    tile_path = os.path.join('datasets', 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
    # tile = Image.open(tile_path)
    # tile.show()

    # rendering
    render = rendering.OffscreenRenderer(256, 256)
    white = rendering.MaterialRecord()
    white.base_color = [1.0, 1.0, 1.0, 1.0]
    white.shader = "defaultLit"
    render.scene.add_geometry('mesh',meshc,white)
    # setup_camera(vertical_field_of_view, center, eye, up)
    # center：相机所指向的中心位置
    # eye：相机的位置
    # up：规定上方的向量
    render.setup_camera(100, [0, 0, 0], [0, 100, 0], [1, 0, 0])
    # render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
    #                                  75000)
    # render.scene.scene.enable_sun_light(True)
    # render.scene.show_axes(True)
    img = render.render_to_image()
    # img = render.render_to_depth_image()
    print("Saving image at test.png")
    o3d.io.write_image("test.png", img, 9)


