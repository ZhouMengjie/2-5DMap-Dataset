## A Large Scale Ground-to-2.5D Map Geolocalization Dataset
This repository contains datasets and codes related to our latest work:
- Image-based Geolocalization by Ground-to-2.5D Map Matching (to be published)


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
1. Obtain the meta data (.osm file) from the [OpenStreetMap](https://www.openstreetmap.org "OpenStreetMap")
```
python osm_loarder.py --dataroot 'datasets'  --city 'manhattan'
```
- We have set the bounding box [minlon, minlat, maxlon, maxlat] for Manhattan and Pittsburgh, you can automatically get their .osm files. 
- If the network connection is not stable, you can also choose to manually download the required files from the official OSM website.
- The data will be stored following this directory structure: \datasets\manhattan\manhattan.osm

2. Get the 2.5D models of each semantic categroy represented in mesh structure
- Preparation: install the [Blender](https://www.blender.org "Blender") 3.1.0 and [blender-osm](https://github.com/vvoovv/blender-osm/wiki/Documentation "blender-osm") addon.
```
blender --background --python blender_osm.py -- manhattan
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
- There are 25 .obj files named with specific semantic category. 
- Please refer to the "color_map.yaml" for each category's name, label and encoded color.


3. pcd_process.py / pcd_processU.py
4. mergeU.py
5. map_dataset_mp.py

