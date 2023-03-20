import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import xml.etree.ElementTree as ET
from functools import reduce
from utils.transverse_mercator import TransverseMercator

'''
Some instructions:
    the get_bbox and get_bbox_v2 are all used to transform the geographical boundary to the global coordinate
    if your input is .osm file including boundary information, you can choose get_bbox()
    if your input is already the geographical bbox, you can choose get_bbox_v2()

    the crop_pcd and crop_pcd_v2 are all used to get the point cloud representing a local aera
    if your input is .pcd file (o3d.utility.Vector3dVector), you can choose crop_pcd()
    if your input is .npy file (np.array), you can choose crop_pcd_v2()
'''

def get_boundary(children):
    boundary_coordinates = []
    for child in children:
        if child.tag == 'bounds':
            boundary_coordinates.append(child.attrib['minlat'])
            boundary_coordinates.append(child.attrib['minlon'])
            boundary_coordinates.append(child.attrib['maxlat'])
            boundary_coordinates.append(child.attrib['maxlon'])

    return boundary_coordinates


def get_bbox(map, origin, input_type='file', radius=76):
    # parse osm file to get the boundary
    if input_type == 'file':
        tree = ET.parse(map)
        root = tree.getroot()
        children = list(root)
    elif input_type == 'str':
        tree = ET.fromstring(map)
        children = tree.getchildren()
    boundary = np.array(get_boundary(children),dtype=float)

    # get the origin
    f = open(origin,'r')
    ls=[]
    for line in f:
        line=line.strip('\n')   
        ls.append(line.split(' '))
    f.close()
    geo_origin=np.array(ls,dtype=float) 

    # transform the geographical boundary to the global coordinate
    projection = TransverseMercator(lat=geo_origin[0], lon=geo_origin[1])
    (min_x, min_y) = projection.fromGeographic(boundary[0], boundary[1])
    (max_x, max_y) = projection.fromGeographic(boundary[2], boundary[3])

    return [min_x-radius, max_x+radius, min_y-radius, max_y+radius]


def get_bbox_v2(boundary, origin, radius=76):
    # get the origin
    f = open(origin,'r')
    ls=[]
    for line in f:
        line=line.strip('\n')   
        ls.append(line.split(' '))
    f.close()
    geo_origin=np.array(ls,dtype=float) 

    # transform the geographical boundary to the global coordinate
    projection = TransverseMercator(lat=geo_origin[0], lon=geo_origin[1])
    (min_x, min_y) = projection.fromGeographic(boundary[0], boundary[1])
    (max_x, max_y) = projection.fromGeographic(boundary[2], boundary[3])

    return [min_x-radius, max_x+radius, min_y-radius, max_y+radius]


def crop_mesh(bbox, mesh):
    vertex_x = np.asarray(mesh.vertices)[:,0]
    vertex_y = np.asarray(mesh.vertices)[:,2]

    indices_x1 = np.where(vertex_x > bbox[0])
    indices_x2 = np.where(vertex_x < bbox[1])
    indices_y1 = np.where(vertex_y > bbox[2])
    indices_y2 = np.where(vertex_y < bbox[3])
    vertex_indices = reduce(np.intersect1d,[indices_x1,indices_x2,indices_y1,indices_y2])

    triangles = np.asarray(mesh.triangles)
    mesh_bools = np.isin(triangles, vertex_indices)
    mesh_indices = []
    for i, mesh_bool in enumerate(mesh_bools):
        if (mesh_bool == True).any():
            mesh_indices.append(i)

    return mesh_indices, vertex_indices


def get_center(center, origin):
    # get the origin
    f = open(origin,'r')
    ls=[]
    for line in f:
        line=line.strip('\n')   
        ls.append(line.split(' '))
    f.close()
    geo_origin=np.array(ls,dtype=float) 

    # transform the geographical boundary to the global coordinate
    projection = TransverseMercator(lat=geo_origin[0], lon=geo_origin[1])
    (x, y) = projection.fromGeographic(center[0], center[1])

    return [x, y]


def crop_pcd(center, pcd, radius=114): 
    min_x = center[0] - radius
    min_y = center[1] - radius
    max_x = center[0] + radius
    max_y = center[1] + radius

    vertex_x = np.asarray(pcd.points)[:,0]
    vertex_y = np.asarray(pcd.points)[:,2]

    indices_x1 = np.where(vertex_x > min_x)
    indices_x2 = np.where(vertex_x < max_x)
    indices_y1 = np.where(vertex_y > min_y)
    indices_y2 = np.where(vertex_y < max_y)
    vertex_indices = reduce(np.intersect1d,[indices_x1,indices_x2,indices_y1,indices_y2])

    return vertex_indices


def crop_pcd_v2(center, points, radius=114): 
    min_x = center[0] - radius
    min_y = center[1] - radius
    max_x = center[0] + radius
    max_y = center[1] + radius

    vertex_x = points[:,0]
    vertex_y = points[:,2]

    indices_x1 = np.where(vertex_x > min_x)
    indices_x2 = np.where(vertex_x < max_x)
    indices_y1 = np.where(vertex_y > min_y)
    indices_y2 = np.where(vertex_y < max_y)
    vertex_indices = reduce(np.intersect1d,[indices_x1,indices_x2,indices_y1,indices_y2])

    return vertex_indices