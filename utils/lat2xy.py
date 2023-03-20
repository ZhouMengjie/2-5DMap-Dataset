import os
import sys
sys.path.append(os.getcwd())
import tqdm
import argparse
import numpy as np
import pandas as pd
from utils import cropping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert lat/lon to x/y')
    parser.add_argument('--dataroot', type=str, required=False, default='datasets', help='dataset root folder')
    parser.add_argument('--area', type=str, required=False, default='unionsquare5kU', help='subset area')
    args = parser.parse_args()

    dataroot = args.dataroot
    area = args.area

    data_path = os.path.join(os.getcwd(), dataroot)
    csv_file = os.path.join(data_path, 'csv', (area + '.csv'))
    assert os.path.exists(csv_file), 'Cannot open file: {}'.format(csv_file)
    data = pd.read_csv(csv_file, sep=',', header=None)
    info = data.values

    for j in tqdm.tqdm(range(info.shape[0])):
        lat = info[j][1]
        lon = info[j][2]
        city = info[j][4]

        origin = os.path.join(data_path, city, (city+'.txt'))
        assert os.path.exists(origin), 'Cannot open file: {}'.format(origin)
        center_xy = cropping.get_center([lat, lon], origin)
        center_xy = [center_xy[0], -center_xy[1]]

        info[j][1] = center_xy[0]
        info[j][2] = center_xy[1]

    np.savetxt(os.path.join(data_path, 'csv', (area+'_xy.csv')), info, delimiter = ',', fmt='%s')