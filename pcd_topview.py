import os
import sys
sys.path.append(os.getcwd())
import random
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from experiments.visualizer import visualize_pcd  

if __name__ == "__main__":
    # load pcd of the whole map
    data_path = 'datasets'
    area = 'hudsonriver5kU'

    # load csv file including pano_id, center (x,y), and heading angle
    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_xy.csv')), sep=',', header=None)
    info = data.values
    # idx = random.randint(0, info.shape[0]-1)
    idx = 1200
    panoid = info[idx][0]
    center_x = info[idx][1]
    center_y = info[idx][2]
    heading = info[idx][3] # in degree
    city = info[idx][4]

    # load point cloud
    points = np.load(os.path.join(data_path, city, city+'U.npy'))
    # load semantic id 
    data = pd.read_csv(os.path.join(data_path, city, city+'U.csv'), sep=',', header=None)
    semantic_ids = data.values
    # load color map
    with open('color_map.yaml','r') as f:
        color = yaml.load(f, Loader=yaml.FullLoader)

    # original area
    vertex_indices = np.load(os.path.join(data_path, (area+'_idx'), (panoid + '.npy')))
    coords = points[vertex_indices][:] # x,z,y
    feats = semantic_ids[vertex_indices]
    colors = []
    for i in range(len(feats)):
        class_id = feats[i][0]
        colors.append(np.divide(color['color_map'][class_id], 255))
    colors = np.asarray(colors)
    # coords = np.concatenate((coords, colors),axis=1)
    visualize_pcd(coords, colors, 'npy')

    # pixelation
    coords[:,[1,2]] = coords[:,[2,1]] # x,y,z
    # ratio = int(coords.shape[0] ** 0.5)
    ratio = 224
    min_ = np.min(coords[:, :2], axis=0) - 1e-8
    max_ = np.max(coords[:, :2], axis=0) + 1e-8
    size2d = np.array([ratio, ratio])
    shape = (max_ - min_) / size2d
    image = np.repeat(np.zeros(size2d)[:, :, None], 3, axis=-1)

    # assign each point to each pixel
    # for i in range(size2d[0]):
    #     for j in range(size2d[1]):
    #         lu = min_ + shape * np.array([i, j])  # left up
    #         rd = min_ + shape * np.array([i + 1, j + 1])  # right down

    #         filter_x = np.bitwise_and(coords[:, 0] <= rd[0], coords[:, 0] >= lu[0])
    #         filter_y = np.bitwise_and(coords[:, 1] <= rd[1], coords[:, 1] >= lu[1])

    #         flags = np.bitwise_and(filter_x, filter_y)
    #         scene_in_block = coords[flags, :]

    #         if scene_in_block.shape[0] > 0:
    #             index = np.argmax(scene_in_block[:, 2])
    #             image[i, j, :] = scene_in_block[index, 3:6]

    # assign each point to each pixel
    coords[:, :2] = coords[:, :2] - min_
    Xindex = coords[:, 0] // shape[0]
    Yindex = coords[:, 1] // shape[1]
    df = pd.DataFrame({"Xindex": Xindex, "Yindex": Yindex, "Z": coords[:, 2]})
    df = df.groupby(["Xindex", "Yindex"]).idxmax().reset_index()  #搜索最大的z的索引
    index = np.array(df).astype(np.int32)
    image[index[:, 1], index[:, 0]] = coords[index[:, 2], 3:6]
    
    plt.imshow(image)
    plt.savefig('topview.png')



    