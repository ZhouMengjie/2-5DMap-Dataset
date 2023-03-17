from math import ceil
import os
import sys
import torch
sys.path.append(os.getcwd())
import yaml
from scipy.linalg import expm, norm
import numpy as np
import open3d.visualization.rendering as rendering
from experiments.visualizer import visualize_pcd 
import torchvision.transforms as transforms 

def tensor2img(x):
    t = transforms.Compose([transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                            transforms.ToPILImage()])
    return t(x)

if __name__ == "__main__":
    data_folder = 'train_batch'
    batch_id = 1
    seed = 20
    device = '3'
    coords = np.load(os.path.join(data_folder, 'coords_'+str(batch_id)+'_'+device+'.npy'))
    features = np.load(os.path.join(data_folder, 'features_'+str(batch_id)+'_'+device+'.npy'))
    images = np.load(os.path.join(data_folder, 'images_'+str(batch_id)+'_'+device+'.npy'))
    labels = np.load(os.path.join(data_folder, 'labels_'+str(batch_id)+'_'+device+'.npy'))
    
    with open('color_map.yaml','r') as f:
        colors = yaml.load(f, Loader=yaml.FullLoader)
    
    coord = []
    color = []
    # mink version
    # for i in range(len(coords)):
    #     if coords[i][0] == seed:
    #         coord.append(coords[i][1:])
    #         class_id = np.argmax(features[i])
    #         color.append(np.divide(colors['color_map'][class_id], 255))
    # coord = np.asarray(coord)
    # color = np.asarray(color)

    # torch version
    pcd = coords[seed].T
    for i in range(len(pcd)):
        coord.append(pcd[i][:3])
        class_id = np.argmax(pcd[i][3:])
        color.append(np.divide(colors['color_map'][class_id], 255))
    coord = np.asarray(coord)
    color = np.asarray(color)
    visualize_pcd(coord, color, 'npy')

    # camera to pixel
    img_sz = 224
    img = np.ones((3,img_sz,img_sz))
    min_x = min(coord[:,0])
    max_x = max(coord[:,0])
    min_y = min(coord[:,2])
    max_y = max(coord[:,2])   
    for i in range(len(coord)):
        x = round((coord[i,0]-min_x)/(max_x-min_x)*(img_sz-1))
        y = round((coord[i,2]-min_y)/(max_y-min_y)*(img_sz-1))
        img[:,y,x] = color[i]

    img_tensor = torch.tensor(img)
    x = tensor2img(img_tensor)
    x.show()
    x.save('test.png')


