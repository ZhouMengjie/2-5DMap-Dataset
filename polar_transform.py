import os
from matplotlib.pyplot import polar
import numpy as np
import pandas as pd
from PIL import Image

def sample_within_bounds(signal, x, y, bounds):
    xmin, xmax, ymin, ymax = bounds
    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)       
    sample = np.zeros((x.shape[0], x.shape[1], signal.shape[-1]))
    sample[idxs, :] = signal[x[idxs], y[idxs], :]
    return sample

def sample_bilinear(signal, rx, ry):
    signal_dim_x = signal.shape[0]
    signal_dim_y = signal.shape[1]
    # obtain four sample coordinates
    ix0 = rx.astype(int)
    iy0 = ry.astype(int)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    bounds = (0, signal_dim_x, 0, signal_dim_y)
    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)
    na = np.newaxis
    # linear interpolation in x-direction
    fx1 = (ix1-rx)[...,na] * signal_00 + (rx-ix0)[...,na] * signal_10
    fx2 = (ix1-rx)[...,na] * signal_01 + (rx-ix0)[...,na] * signal_11
    # linear interpolation in y-direction
    return (iy1 - ry)[...,na] * fx1 + (ry - iy0)[...,na] * fx2

def get_polar(tile, height=224, width=448):
    S = 224 # tile_size
    i = np.arange(0, height)
    j = np.arange(0, width)
    jj, ii = np.meshgrid(j, i)
    y = S/2. - S/2./height*(height-1-ii)*np.sin(2*np.pi*jj/width)
    x = S/2. + S/2./height*(height-1-ii)*np.cos(2*np.pi*jj/width)
    polar_tile = sample_bilinear(tile, x, y)
    return polar_tile

if __name__ == '__main__':
    dataset_path = 'datasets'
    query_filename = 'hudsonriver5kU'

    query_filepath = os.path.join(dataset_path, 'csv', query_filename+'_xy.csv')
    queries = (pd.read_csv(query_filepath, sep=',', header=None)).values

    set_filepath = os.path.join(dataset_path, 'csv', query_filename+ '_set.csv')
    global_queries= (pd.read_csv(set_filepath, sep=',', header=None)).values

    ndx = 1200
    panoid = queries[ndx][0] # panoid
    city = queries[ndx][4]
    global_idx = global_queries[ndx][1]

    pano_pathname = os.path.join(dataset_path, 'jpegs_'+city+'_2019', panoid+'.jpg')
    pano = Image.open(pano_pathname).convert('RGB')
    pano.save('pano.png')

    tile_pathname = os.path.join(dataset_path, 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
    assert os.path.isfile(tile_pathname), "Image {} not found".format(tile_pathname)
    tile = Image.open(tile_pathname).convert('RGB')
    tile.save('bev.png')
    tile = np.array(tile) # 256x256x3
    polar_tile = get_polar(tile) # 224x448x3
    polar_tile = Image.fromarray(np.uint8(polar_tile))
    polar_tile.show()
    polar_tile.save('polar.png')


