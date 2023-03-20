import os
import sys
sys.path.append(os.getcwd())
import copy
import yaml
import argparse
import numpy as np
import open3d as o3d
from utils import cropping
from utils import sampling

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate .pcd files')
    parser.add_argument('--dataroot', type=str, required=False, default='datasets', help='dataset root folder')
    parser.add_argument('--city', type=str, required=False, default='manhattan', help='city name')
    parser.add_argument('--radius', type=int, required=False, default=76, help='height/width of the local region')
    parser.add_argument('--density', type=float, required=False, default=0.1, help='sampling density')
    args = parser.parse_args()

    dataroot = args.dataroot
    city = args.city
    radius = args.radius
    density = args.density
    data_path = os.path.join(os.getcwd(), dataroot, city)
    obj_files= os.listdir(os.path.join(data_path, (city + '_obj')))
    
    cropped_name = city + '_cropped'
    pcd_name = city + 'U_pcd'
    if not os.path.isdir(os.path.join(data_path, cropped_name)):
        os.makedirs(os.path.join(data_path, cropped_name))
    if not os.path.isdir(os.path.join(data_path, pcd_name)):
        os.makedirs(os.path.join(data_path, pcd_name))

    # calculating bbox
    origin = os.path.join(data_path, (city + '.txt'))
    if city == 'manhattan':
        boundary = [40.695, -74.028, 40.788, -73.940]
    elif city == 'pittsburgh':
        boundary = [40.425, -80.035, 40.460, -79.930]
    else:
        raise NotImplementedError('Please manually set the bounding box!')
    bbox = cropping.get_bbox_v2(boundary, origin, radius)

    # load color map
    with open('color_map.yaml','r') as f:
        color = yaml.load(f, Loader=yaml.FullLoader)

    # labels
    labels = []
    
    print("Loading mesh file ...")
    for i, obj_file in enumerate(obj_files):
        obj_name = os.path.splitext(obj_file)[0]
        class_name = obj_name.split('.')[1]
        class_name = class_name.replace('osm_','')
        print(class_name)

        if os.path.isfile(os.path.join(data_path, cropped_name, (obj_name +'.obj'))):
            meshc = o3d.io.read_triangle_mesh(os.path.join(data_path, cropped_name, (obj_name +'.obj')))
        else:
            mesh = o3d.io.read_triangle_mesh(os.path.join(data_path, (city + '_obj'), (obj_name + '.obj'))) # try to read object class
            print("Cropping mesh file ...")
            (mesh_indices, vertex_indices) = cropping.crop_mesh(bbox, mesh)  
            print("The number of vertices is " + str(len(vertex_indices))) 

            meshc = copy.deepcopy(mesh)
            meshc.triangles = o3d.utility.Vector3iVector(
                np.asarray(mesh.triangles)[mesh_indices, :])                  
            o3d.io.write_triangle_mesh(os.path.join(data_path, cropped_name, (obj_name +'.obj')), meshc)
        
        print("Uniformly sampling points from meshes ...")
        if len(np.asarray(meshc.triangles)) > 0:
            if os.path.isfile(os.path.join(data_path, pcd_name, (obj_name + '.pcd'))):
                pcd = o3d.io.read_point_cloud(os.path.join(data_path, pcd_name, (obj_name + '.pcd')))
            else:
                # in this way, normal is lacked
                coords = sampling.resample_mesh(meshc, density)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(coords)
                class_id = color['labels'][class_name]
                assert class_name is not color['categories'][class_id], 'Class name and id is not aligned at {}'.format(class_id)
                pcd.paint_uniform_color(np.divide(color['color_map'][class_id], 255))
                label = np.array([class_id] * len(coords))
                labels = np.concatenate((labels,label), axis=0)
                o3d.io.write_point_cloud(os.path.join(data_path, pcd_name, (obj_name + '.pcd')), pcd)

    np.savetxt(os.path.join(data_path, (city + 'U.csv')), labels, delimiter = ',', fmt = '%s')                



