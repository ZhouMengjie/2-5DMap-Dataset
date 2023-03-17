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


def projection(center, pc, image_h, image_w, sensor_h, sensor_w, npoints, batch_size):
        center_x = center[:,0]
        center_y = center[:,2]
        cx = 0.5*sensor_w - center_x
        cy = 0.5*sensor_h - center_y
        cx = cx.view(batch_size, 1)
        cy = cy.view(batch_size, 1)
        image_x = (pc[:,0,:] + cx)*(image_w - 1) / (sensor_w - 1)
        image_y = (pc[:,2,:] + cy)*(image_h - 1) / (sensor_h - 1)
        return torch.cat([
            image_x[:,None,:],
            image_y[:,None,:],
        ], dim=1)


if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), 'datasets')
    area = 'unionsquare5kU'

    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_xy.csv')), sep=',', header=None)
    info = data.values
    idx = 1534
    # idx = random.randint(0, 5000)
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
    # center_x = torch.tensor(center_x)
    # center_y = torch.tensor(center_y)
    image_w, image_h = 256, 256
    sensor_w, sensor_h = 152, 152
    pc = result['cloud']   # N x C
    # cx = 0.5*sensor_w - center_x + 1
    # cy = 0.5*sensor_h - center_y + 1
    # image_x = torch.round((pc[:,0] + cx)*(image_w - 1) / (sensor_w))
    # image_y = torch.round((pc[:,2] + cy)*(image_h - 1) / (sensor_h))
    # image = torch.cat([
    #     image_x[:,None],
    #     image_y[:,None],
    # ], dim=1)
    # image = image.type(torch.int)
    batch_size = 1
    center_x = torch.tensor(center_x).view(1,-1)
    center_y = torch.tensor(center_y).view(1,-1)
    pc = pc.transpose(1,0)
    pc = pc.view(1,3,-1)
    cx = 0.5*sensor_w - center_x
    cy = 0.5*sensor_h - center_y
    cx = cx.view(batch_size, 1)
    cy = cy.view(batch_size, 1)
    image_x = (pc[:,0,:] + cx)*(image_w - 1) / (sensor_w - 1)
    image_y = (pc[:,2,:] + cy)*(image_h - 1) / (sensor_h - 1)
    image =  torch.cat([
        image_x[:,None,:],
        image_y[:,None,:],
    ], dim=1)
    image = image.type(torch.int)

    # tile
    data = pd.read_csv(os.path.join('datasets', 'csv', ( area+ '_set.csv')), sep=',', header=None)
    info = data.values   
    global_idx = info[idx][1] # for train only
    city = info[idx][0]
    tile_path = os.path.join('datasets', 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
    # tile = Image.open(tile_path)
    # tile.show()
    tile = cv2.imread(tile_path)
    point_list = image[0].numpy().T
    for point in point_list:
        cv2.circle(tile,point,1,(255,0,0),1)
    cv2.imshow('projection',tile)
    cv2.waitKey(0)
    cv2.imwrite('projection.png',tile)




   