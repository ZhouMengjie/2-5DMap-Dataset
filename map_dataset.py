import os
import sys
sys.path.append(os.getcwd())
import copy
import tqdm
import numpy as np
import pandas as pd
import open3d as o3d
from geographiclib.geodesic import Geodesic
from openstreetmap.transverse_mercator import TransverseMercator
from openstreetmap import cropping

if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), 'datasets')
    # load pcd
    pcd_mh = o3d.io.read_point_cloud(os.path.join(data_path, 'manhattan', 'manhattan.pcd'))
    origin_mh = os.path.join(data_path, 'manhattan', ('manhattan.txt'))
                    
    pcd_pt = o3d.io.read_point_cloud(os.path.join(data_path, 'pittsburgh', 'pittsburgh.pcd'))
    origin_pt = os.path.join(data_path, 'pittsburgh', ('pittsburgh.txt'))

    # read csv file
    # argv = sys.argv[sys.argv.index("--") + 1:]
    # assert(len(argv) == 1)
    # area = argv[0]
    area = 'wallstreet5k'
    if not os.path.isdir(os.path.join(data_path, area)):
        os.makedirs(os.path.join(data_path, area))

    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '.csv')), sep=',', header=None)
    info = data.values
    radius = 76

    # area = area+'_small'
    for j in tqdm.tqdm(range(info.shape[0])):
        panoid = info[j][0]
        lat = info[j][1]
        lon = info[j][2]
        heading = info[j][3] / 180 * np.pi
        city = info[j][4]

        if os.path.exists(os.path.join(os.path.join(data_path, area), (panoid + '.npy'))):
            print(panoid+' exists')
        else:
            # plan A: first rotate, then crop
            if city == 'manhattan':
                center_xy = cropping.get_center([lat, lon], origin_mh)
                center_xy = [center_xy[0], -center_xy[1]]
                pcdr = copy.deepcopy(pcd_mh)
                R = pcd_mh.get_rotation_matrix_from_xyz((0, heading, 0))
            elif city == 'pittsburgh':
                center_xy = cropping.get_center([lat, lon], origin_pt)
                center_xy = [center_xy[0], -center_xy[1]]
                pcdr = copy.deepcopy(pcd_pt)
                R = pcd_pt.get_rotation_matrix_from_xyz((0, heading, 0))
            else:
                center_xy = []
                R = []
                assert len(center_xy)==0 or len(R) == 0, f"Cannot access find correspondent points in both datasets!"
            pcdr.rotate(R, center=(center_xy[0], 0, center_xy[1])) 
            vertex_indices = cropping.crop_pcd(center_xy, pcdr, radius)
            if len(vertex_indices) > 0:
                # print("Patitioning the point cloud ...")
                # Note: the indexes are from pcdr, not original pcd
                np.save(os.path.join(os.path.join(data_path, area), (panoid + '_idx.npy')), vertex_indices)
                pcdi = pcdr.select_by_index(vertex_indices)
                points = np.asarray(pcdi.points)
                np.save(os.path.join(os.path.join(data_path, area), (panoid + '.npy')), points)
                o3d.visualization.draw_geometries([pcdi], window_name="cropped_pcd") 

            # plan B: first crop, then rotate      
            # if city == 'manhattan':
            #     center_xy = cropping.get_center([lat, lon], origin_mh)
            #     center_xy = [center_xy[0], -center_xy[1]]
            #     vertex_indices = cropping.crop_pcd(center_xy, pcd_mh, radius)
            #     if len(vertex_indices) > 0:
            #         pcdi = pcd_mh.select_by_index(vertex_indices)
            #         pcdr = copy.deepcopy(pcdi)
            #         R = pcdi.get_rotation_matrix_from_xyz((0, heading, 0))
            #         pcdr.rotate(R, center=(center_xy[0], 0, center_xy[1])) 
            #         o3d.visualization.draw_geometries([pcdr], window_name="rotated_pcd")
            # elif city == 'pittsburgh':
            #     center_xy = cropping.get_center([lat, lon], origin_pt)
            #     center_xy = [center_xy[0], -center_xy[1]]
            #     vertex_indices = cropping.crop_pcd(center_xy, pcd_pt, radius)
            #     if len(vertex_indices) > 0:
            #         pcdi = pcd_pt.select_by_index(vertex_indices)
            #         pcdr = copy.deepcopy(pcdi)
            #         R = pcdi.get_rotation_matrix_from_xyz((0, heading, 0))
            #         pcdr.rotate(R, center=(center_xy[0], 0, center_xy[1])) 
            #         # o3d.visualization.draw_geometries([pcdr], window_name="rotated_pcd")  
            # else:
            #     center_xy = []
            #     R = []
            #     assert len(center_xy)==0 or len(R) == 0, f"Cannot access find correspondent points in both datasets!"     
            
            # plan c: only crop
            if city == 'manhattan':
                center_xy = cropping.get_center([lat, lon], origin_mh)
                center_xy = [center_xy[0], -center_xy[1]]
                pcd = pcd_mh
            elif city == 'pittsburgh':
                center_xy = cropping.get_center([lat, lon], origin_pt)
                center_xy = [center_xy[0], -center_xy[1]]
                pcd = pcd_pt
            else:
                center_xy = []
                R = []
                assert len(center_xy)==0 or len(R) == 0, f"Cannot access find correspondent points in both datasets!"

            vertex_indices = cropping.crop_pcd(center_xy, pcd, radius)
            if len(vertex_indices) > 0:
                np.save(os.path.join(os.path.join(data_path, area), (panoid + '_114.npy')), vertex_indices)                      

            







    
        
        

            






