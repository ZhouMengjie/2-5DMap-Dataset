import os
import yaml
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import numpy
plt.rc('font',family='Times New Roman') 

def visualize_pcd(coords, colors, type='pcd'):
    if type != 'pcd':
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        if colors is None:
            pcd.paint_uniform_color([1, 0.706, 0])
        else:
            pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd = coords

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="pcl")
    vis.get_render_option().point_size = 2
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def class_statistic(data, file_name):
    labels = data.values
    semantic_ids, class_nums = np.unique(labels, return_counts=True)
    height = numpy.log10(class_nums)
    with open('color_map.yaml','r') as f:
        color = yaml.load(f, Loader=yaml.FullLoader)
    
    categories = []
    colormaps = [] 
    for id in semantic_ids:
        categories.append(color['categories'][id])
        colormaps.append(np.divide(color['color_map'][id], 255))

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(20, 7))
    # plt.rcParams.update({"font.size":15})
    for i in range(len(height)):
        ax.bar(
        x = str(int(semantic_ids[i])),  
        height = height[i], 
        width = 0.6, 
        align = "center",
        color = colormaps[i],
        label = categories[i]
        )

    xticks = ax.get_xticks()
    for i in range(len(class_nums)):
        xy = (xticks[i], height[i] * 1.03)
        s = str(class_nums[i])
        ax.annotate(
            text=s,  
            xy=xy,  
            fontsize=14,  
            fontweight='bold',
            ha="center", 
            va="baseline"  
        )
    font={'size':20}
    ax.set_ylim(0,15)
    # ax.set_title("Semantic Classes", fontdict=font)  
    ax.set_xlabel("categories",fontdict=font)
    ax.set_ylabel("number of points (log)",fontdict=font)
    ax.legend(loc=1,ncol=4,fontsize=18)
    # plt.xticks(fontproperties='Times New Roman',size=15,weight='bold')
    # plt.yticks(fontproperties='Times New Roman',size=15,weight='bold')
    plt.savefig(os.path.join("results", (file_name + ".jpg")), bbox_inches='tight')

if __name__ == "__main__":
    city = 'manhattan'
    data_path = os.path.join(os.getcwd(), 'datasets', city)
    print('Displaying the completed point cloud ...')
    pcd = o3d.io.read_point_cloud(os.path.join(data_path, (city + 'U.pcd')))
    visualize_pcd(pcd, None, type='pcd')

    print('Displaying the semantic classes statistic ...')
    data = pd.read_csv(os.path.join(data_path, (city + 'U.csv')),sep=',', header=None)
    class_statistic(data, city)
    




