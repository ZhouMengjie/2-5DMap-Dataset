## A Large Scale Ground-to-2.5D Map Geolocalization Dataset
This repository contains datasets and codes related to our latest work:
- Image-based Geolocalization by Ground-to-2.5D Map Matching (to be published)
- The task is done by querying the ground-view image with respect to the cartographic map.
- The large-scale and georeferenced map consists of 2.5D structure models and 2D aerial-view map images.

2D map            |  2.5D map
:-------------------------:|:-------------------------:
<img src="datasets/examples/2Dmap.png" width="435" height="267">  |  <img src="datasets/examples/2_5Dmap.png" width="500" height="300">


### Datasets
Here we present two ready-made datasets repsectively including:
- Ground-view panoramic images (To request, please visit the project of [StreetLearn Dataset](https://sites.google.com/view/streetlearn/dataset "StreetLearn Dataset")). 
- 2D maps (To request, please visit our previous project: [You Are Here: Geolocation by Embedding Maps and Images](https://github.com/ZhouMengjie/Image-Map-Embeddings "You Are Here: Geolocation by Embedding Maps and Images")).


### Codes
Here we provide python implementation to generate 2.5D map dataset. The prerequisite and data procesing pipeline are shown below.

#### Prerequisite
You can use conda and pip to configure your own environment with the following commands:
 ```
conda create -n env_name python=3.8
pip3 install pyyaml
pip3 install pandas
pip3 install tqdm
pip3 install open3d==0.15.1 
```

#### Data Processing Pipeline
Step 1. Obtain the meta data (.osm file) from the [OpenStreetMap](https://www.openstreetmap.org "OpenStreetMap")
```
python osm_loarder.py --dataroot 'datasets' --city 'manhattan'
```
- We have set the bounding box [minlon, minlat, maxlon, maxlat] for Manhattan and Pittsburgh, you can automatically get their .osm files. 
- If the network connection is not stable, you can also choose to manually download the required files from the official OSM website.
- The data will be stored following this directory structure: \datasets\manhattan\manhattan.osm

Step 2. Get the 2.5D models of each semantic categroy represented in mesh structure
- Preparation: install the [Blender 3.1.0](https://www.blender.org "Blender 3.1.0") and [blender-osm addon](https://github.com/vvoovv/blender-osm/wiki/Documentation "blender-osm addon").
```
blender --background --python blender_osm.py --dataroot 'datasets' --city 'manhattan'
```
- The data will be stored following this directory structure: 
```
|–– datasets
|   |––manhattan
|   |   |––manhattan.txt
|   |   |––manhattan_obj
|   |   |   |––manhattan.osm_areas_footway.obj
|   |   |   |––manhattan.osm_buildings.obj
|   |   |   |––manhattan.osm_roads_primary.obj
```
- The [lat,lon] saved in .txt file aligns with the the origin of the Blender global coordinate system.
- There are 25 .obj files named with specific semantic category.
- The .obj file mainly includes: o-object name, v-geometric vertices, vn-vertex normals, vt-texture vertices, mtllib-material library, usemtl-material name, f-face, l-line.

Step 3. Generate point cloud of each semantic category by adopting uniform sampling on mesh
```
python pcd_generate.py --dataroot 'datasets' --city 'manhattan' --radius 76 --density 0.1
```
- The data will be stored following this directory structure: 
```
|–– datasets
|   |––manhattan
|   |   |––manhattan_cropped
|   |   |––manhattanU_pcd
|   |   |   |––manhattan.osm_areas_footway.pcd
|   |   |   |––manhattan.osm_buildings.pcd
|   |   |   |––manhattan.osm_roads_primary.pcd
|   |   |––manhattanU.csv
```
- Due to the connectivity of regions (such as water), the actual downloaded 2.5D model will be much larger than the required area. 
- Therefore, we first cut the original data based on the bounding box, and saved them in \manhattan_cropped\.
- The [barycentric coordinate](https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/ "barycentric coordinate") is used to achieve       the uniform sampling.
- The semantic label of each point is saved in .csv file. 
- Please refer to the "color_map.yaml" for each category's name, label and encoded color.

Step 4. Merge point clouds of each semantic category into a completed point cloud representing the whole area
```
python pcd_merge.py --dataroot 'datasets' --city 'manhattan'
```
- The data will be stored following this directory structure: \datasets\manhattan\manhattanU.pcd or manhattanU.npy.
- The .pcd file contains the coordinate and color of each point, while .npy file only contains the coordinate information.

Step 5. Generate map subsets for training, validation and testing with multiprocessing
```
python lat2xy.py --dataroot 'datasets' --area 'unionsquare5kU'
python merge_csv.py
python map_dataset_mp.py --dataroot 'datasets' --area 'unionsquare5kU' --radius 114 --num_threads 16
```
- First, the center [lat,lon] of each local region provided in .csv files should be converted to [x,y].
- There are five areas named as 'hudsonriver5kU', 'wallstreet5kU', 'unionsquare5kU', 'trainstreetlearnU' and 'cmu5kU'.
- The .csv files of 'trainstreetlearnU' and 'cmu5kU' are merged for training. 
- The radius is defined as original_radius * sqrt(2), which can ensure that there is no loss of information after rotating the magnified area and restoring it to its original size.

Finally, all the data will be stored following this directory structure: 
```
|–– datasets
|   |––csv
|   |––manhattan
|   |––pittsburgh
|   |––jpegs_manhattan_2019
|   |––jpegs_pittsburgh_2019
|   |––tiles_manhattan_2019
|   |––tiles_pittsburgh_2019
|   |––trainstreetlearnU_cmu5kU_idx
|   |––hudsonriver5kU_idx
|   |––wallstreet5kU_idx
|   |––unionsquare5kU_idx
```
- The ground-view image and 2.5D map (point cloud) are named with unique string identifier. The 2D map is named with global index.
- The folders of "manhattan" and "pittsburgh" can be obtained with the extraction code [data](https://pan.baidu.com/s/1XTy4qbMVDXHIjPJi2JZVqw "data").
- We provide a pair of data in \datasets\examples\ and "visualizer.py" to help users do a visualization check. 

Panoram                                |  2D map                | 2.5D map         
:-------------------------:|:-------------------------:|:-------------------------:
<img src="datasets/examples/_dlEF8O77LTsFm2G9m7EiA.jpg" width="448" height="224">  |  <img src="datasets/examples/46265.png" > | <img src="datasets/examples/map1.png" >


### FAQ
1. To install the Open3D on Linux, you may encounter that the GLIBC version is incompatible.
- The compatible GLIBC version of open3d-0.15.0 is 2.18 or 2.27. You can choose to upgrade it if you have root authority.
- We have verified that it is much easier to install Open3D on MacOS or Windows system. 
2. In the Step 3, you may be prompted that [Open3D INFO] Skipping non-triangle primitive geometry of type: 6.
- You should check your .obj files and delete "l" (line) manually. Otherwise, the .obj file won't be processed by the subsequent steps.
3. If you have any other questions, please feel free to leave a message or contact me via "mengjie.zhou@bristol.ac.uk".


### To release
- codes for learning embedding space
- codes for route based geolocalization
