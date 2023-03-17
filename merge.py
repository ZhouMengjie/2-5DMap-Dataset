import os
import numpy as np
import open3d as o3d

if __name__ == "__main__":
    city = 'pittsburgh'
    data_path = os.path.join(os.getcwd(), 'datasets', city)
    pcd_files= os.listdir(os.path.join(data_path, (city + '_pcd')))

    pcd = o3d.io.read_point_cloud(os.path.join(data_path, (city + '_pcd'), pcd_files[0]))
    pcd_points = np.asarray(pcd.points)
    pcd_normals = np.asarray(pcd.normals)
    pcd_colors = np.asarray(pcd.colors)
    
    for i  in range(1, len(pcd_files)):
        pcdt = o3d.io.read_point_cloud(os.path.join(data_path, (city + '_pcd'), pcd_files[i]))
        pcdt_points = np.asarray(pcdt.points)
        pcdt_normals = np.asarray(pcdt.normals)
        pcdt_colors = np.asarray(pcdt.colors)
        print(len(pcdt_colors))

        pcd_points = np.concatenate((pcd_points,pcdt_points), axis=0)
        pcd_normals = np.concatenate((pcd_normals,pcdt_normals), axis=0)
        pcd_colors = np.concatenate((pcd_colors,pcdt_colors), axis=0)

    pcdn = o3d.geometry.PointCloud()
    pcdn.points = o3d.utility.Vector3dVector(pcd_points)
    pcdn.normals = o3d.utility.Vector3dVector(pcd_normals)
    pcdn.colors = o3d.utility.Vector3dVector(pcd_colors)

    o3d.visualization.draw_geometries([pcdn], window_name="sampled_point")
    o3d.io.write_point_cloud(os.path.join(data_path, (city + '.pcd')), pcdn)
    np.save(os.path.join(data_path, (city + '.npy')), pcd_points)







