import os
import sys
sys.path.append(os.getcwd())
import torch
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import yaml
from experiments.visualizer import visualize_pcd  
from data import augmentation_pc
import cv2
import random
from torch.nn.functional import grid_sample

def tensor2img(x):
    t = transforms.Compose([transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                            transforms.ToPILImage()])
    return t(x)


def farthest_point_sample(point, feature, npoint):
        """
        Input:
            xyz: pointcloud data, [N, D]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [npoint, D]
        """
        N, D = point.shape
        xyz = point[:,:3]
        centroids = np.zeros((npoint,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N) # same as shuffle
        xyz2 = np.sum(xyz ** 2, -1) 
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            # dist = np.sum((xyz - centroid) ** 2, -1)
            dist = -2 * np.matmul(xyz, centroid)
            dist += xyz2
            dist +=  np.sum(centroid ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        pt = point[centroids.astype(np.int32)]
        ft = feature[centroids.astype(np.int32)]
        return pt, ft

mesh_grid_cache = {}
def mesh_grid(n, h, w, device, channel_first=True):
    global mesh_grid_cache
    str_id = '%d,%d,%d,%s,%s' % (n, h, w, device, channel_first)
    if str_id not in mesh_grid_cache:
        x_base = torch.arange(0, w, dtype=torch.float32, device=device)[None, None, :].expand(n, h, w)
        y_base = torch.arange(0, h, dtype=torch.float32, device=device)[None, None, :].expand(n, w, h)  # NWH
        grid = torch.stack([x_base, y_base.transpose(1, 2)], 1)  # B2HW
        if not channel_first:
            grid = grid.permute(0, 2, 3, 1)  # BHW2
        mesh_grid_cache[str_id] = grid
    return mesh_grid_cache[str_id]


def squared_distance(xyz1: torch.Tensor, xyz2: torch.Tensor):
    """
    Calculate the Euclidean squared distance between every two points.
    :param xyz1: the 1st set of points, [batch_size, n_points_1, 3]
    :param xyz2: the 2nd set of points, [batch_size, n_points_2, 3]
    :return: squared distance between every two points, [batch_size, n_points_1, n_points_2]
    """
    assert xyz1.shape[-1] == xyz2.shape[-1] and xyz1.shape[-1] <= 3  # assert channel_last
    batch_size, n_points1, n_points2 = xyz1.shape[0], xyz1.shape[1], xyz2.shape[1]
    dist = -2 * torch.matmul(xyz1, xyz2.permute(0, 2, 1))
    dist += torch.sum(xyz1 ** 2, -1).view(batch_size, n_points1, 1)
    dist += torch.sum(xyz2 ** 2, -1).view(batch_size, 1, n_points2)
    return dist


def k_nearest_neighbor(input_xyz: torch.Tensor, query_xyz: torch.Tensor, k: int):
    """
    Calculate k-nearest neighbor for each query.
    :param input_xyz: a set of points, [batch_size, n_points, 3] or [batch_size, 3, n_points]
    :param query_xyz: a set of centroids, [batch_size, n_queries, 3] or [batch_size, 3, n_queries]
    :param k: int
    :param cpp_impl: whether to use the CUDA C++ implementation of k-nearest-neighbor
    :return: indices of k-nearest neighbors, [batch_size, n_queries, k]
    """
    if input_xyz.shape[1] <= 3:  # channel_first to channel_last
        assert query_xyz.shape[1] == input_xyz.shape[1]
        input_xyz = input_xyz.transpose(1, 2).contiguous()
        query_xyz = query_xyz.transpose(1, 2).contiguous()
    
    dists = squared_distance(query_xyz, input_xyz)
    return dists.topk(k, dim=2, largest=False).indices.to(torch.long)


def grid_sample_wrapper(feat_2d, xy):
    image_h, image_w = feat_2d.shape[2:]
    new_x = 2.0 * xy[:, 0] / (image_w - 1) - 1.0  # [bs, n_points] [-1,1]
    new_y = 2.0 * xy[:, 1] / (image_h - 1) - 1.0  # [bs, n_points]
    new_xy = torch.cat([new_x[:, :, None, None], new_y[:, :, None, None]], dim=-1)  # [bs, n_points, 1, 2]
    result = grid_sample(feat_2d, new_xy, 'bilinear', align_corners=True)  # [bs, n_channels, n_points, 1]
    return result[..., 0]

def batch_indexing_channel_first(batched_data: torch.Tensor, batched_indices: torch.Tensor):
    """
    :param batched_data: [batch_size, C, N]
    :param batched_indices: [batch_size, I1, I2, ..., Im]
    :return: indexed data: [batch_size, C, I1, I2, ..., Im]
    """
    def product(arr):
        p = 1
        for i in arr:
            p *= i
        return p
    assert batched_data.shape[0] == batched_indices.shape[0]
    batch_size, n_channels = batched_data.shape[:2]
    indices_shape = list(batched_indices.shape[1:])
    batched_indices = batched_indices.reshape([batch_size, 1, -1])
    batched_indices = batched_indices.expand([batch_size, n_channels, product(indices_shape)])
    result = torch.gather(batched_data, dim=2, index=batched_indices.to(torch.int64))
    result = result.view([batch_size, n_channels] + indices_shape)
    return result

if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), 'datasets')
    area = 'unionsquare5kU'

    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_xy.csv')), sep=',', header=None)
    info = data.values
    idx = 200
    idx = random.randint(0, 5000)
    # seed = 3
    print(idx)
    panoid = info[idx][0]
    center_x = info[idx][1]
    center_y = info[idx][2]
    heading = info[idx][3]
    city = info[idx][4]

    # load point cloud
    points = np.load(os.path.join(data_path, city, city+'U.npy'))
    # load semantic id 
    data = pd.read_csv(os.path.join(data_path, city, city+'U.csv'), sep=',', header=None)
    semantic_ids = data.values
    # load color map
    with open('color_map.yaml','r') as f:
        color = yaml.load(f, Loader=yaml.FullLoader)

    vertex_indices = np.load(os.path.join(data_path, (area+'_idx'), (panoid + '.npy')))
    coords = points[vertex_indices][:]
    feats = semantic_ids[vertex_indices]
    colors = []
    for i in range(len(feats)):
        class_id = feats[i][0]
        colors.append(np.divide(color['color_map'][class_id], 255))
    colors = np.asarray(colors)
    
    pc = torch.tensor(coords, dtype=torch.float)
    center = [center_x, 0, center_y]
    result = {'heading': heading, 'center':center}
    result['cloud'] = torch.tensor(coords, dtype=torch.float)
    result['cloud_ft'] = torch.tensor(colors, dtype=torch.float)
    t = [augmentation_pc.RandomRotation(max_theta=0, axis=np.asarray([0,1,0])),
        augmentation_pc.RandomCenterCrop(radius=76, rnd=0)]
    transform = transforms.Compose(t)    
    result = transform(result)
    coords = result['cloud'].numpy()
    feats = result['cloud_ft'].numpy()
    coords, feats = farthest_point_sample(coords, feats, 1024)
    result['cloud'] = torch.tensor(coords, dtype=torch.float)
    result['cloud_ft'] = torch.tensor(feats, dtype=torch.float)
    new_coords = result['cloud'].numpy()
    new_colors = result['cloud_ft'].numpy()
    # visualize_pcd(new_coords, new_colors, 'npy')

    # projection
    center_x = torch.tensor(center_x)
    center_y = torch.tensor(center_y)
    image_w, image_h = 256, 256
    sensor_w, sensor_h = 152, 152
    pc = result['cloud']   # N x C
    # pc -> b, c, n
    pc = pc.view(1,3,-1)

    cx = 0.5*sensor_w - center_x
    cy = 0.5*sensor_h - center_y
    image_x = (pc[:,0,:] + cx)*(image_w - 1) / (sensor_w - 1)
    image_y = (pc[:,2,:] + cy)*(image_h - 1) / (sensor_h - 1)
    image = torch.cat([
        image_x[:,None,:],
        image_y[:,None,:],
    ], dim=1)
    # image = image.type(torch.int)

    # pre-compute knn indices
    batch_size = image.shape[0]
    grid = mesh_grid(batch_size, image_h, image_w, image.device)  # [B, 2, H, W]
    grid = grid.reshape([batch_size, 2, -1])  # [B, 2, HW]
    nn_proj1 = k_nearest_neighbor(image, grid, k=1)  # [B, HW, k]
    nn_indices = nn_proj1

    # pwc fusion
    feat_2d = torch.rand(batch_size,128,image_h,image_w)
    feat_3d = torch.rand(batch_size,128,1024)

    batch_size, _, image_h, image_w = feat_2d.shape
    grid = mesh_grid(batch_size, image_h, image_w, image.device)  # [B, 2, H, W]
    grid = grid.reshape([batch_size, 2, -1])  # [B, 2, HW]
    if nn_indices is None:
        nn_indices = k_nearest_neighbor(image, grid, k=1)[..., 0]  # [B, HW]
    else:
        nn_indices = nn_indices[...,0]
        assert nn_indices.shape == (batch_size, image_h * image_w)

    nn_feat2d = batch_indexing_channel_first(grid_sample_wrapper(feat_2d, image), nn_indices)  # [B, n_channels_2d, HW]
    nn_feat3d = batch_indexing_channel_first(feat_3d, nn_indices)  # [B, n_channels_3d, HW]
    nn_offset = batch_indexing_channel_first(image, nn_indices) - grid  # [B, 2, HW]
    nn_corr = torch.mean(nn_feat2d * feat_2d.reshape(batch_size, -1, image_h * image_w), dim=1, keepdim=True)  # [B, 1, HW]

    final = torch.cat([nn_offset, nn_corr, nn_feat3d], dim=1)  # [B, n_channels_3d + 3, HW]
    final = final.reshape([batch_size, -1, image_h, image_w])

   

    # tile
    data = pd.read_csv(os.path.join('datasets', 'csv', ( area+ '_set.csv')), sep=',', header=None)
    # info = data.values   
    # global_idx = info[idx][1] # for train only
    # city = info[idx][0]
    # tile_path = os.path.join('datasets', 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
    # # tile = Image.open(tile_path)
    # # tile.show()
    # tile = cv2.imread(tile_path)
    # point_list = image.numpy()
    # for point in point_list:
    #     cv2.circle(tile,point,1,(255,0,0),1)
    # cv2.imshow('projection',tile)
    # cv2.waitKey(0)
    # cv2.imwrite('projection.png',tile)




   