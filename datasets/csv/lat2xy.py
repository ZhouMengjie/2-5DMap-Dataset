import os
import sys
sys.path.append(os.getcwd())
import tqdm
import numpy as np
import pandas as pd
from openstreetmap import cropping

if __name__ == "__main__":
    area = 'cmu5k'
    data_path = os.path.join(os.getcwd(), 'datasets')
    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '.csv')), sep=',', header=None)
    info = data.values

    for j in tqdm.tqdm(range(info.shape[0])):
        lat = info[j][1]
        lon = info[j][2]
        city = info[j][4]

        origin= os.path.join(data_path, city, (city+'.txt'))
        center_xy = cropping.get_center([lat, lon], origin)
        center_xy = [center_xy[0], -center_xy[1]]

        info[j][1] = center_xy[0]
        info[j][2] = center_xy[1]

    np.savetxt(os.path.join(data_path, 'csv', (area+'_xy.csv')), info, delimiter = ',', fmt='%s')