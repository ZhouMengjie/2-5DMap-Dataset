import os
import tqdm
import numpy as np
import pandas as pd

if __name__ == "__main__":
    area = 'trainstreetlearn'
    data_path = os.path.join(os.getcwd(), 'datasets')
    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '.csv')), sep=',', header=None)
    info = data.values

    for j in tqdm.tqdm(range(info.shape[0])):
        city = info[j][4]
        panoid = info[j][0]

        img_path = os.path.join(data_path, 'jpegs_manhattan_2019', panoid+'.jpg')

        if not os.path.exists(img_path):
            img_path = os.path.join(data_path, 'jpegs_pittsburgh_2019', panoid+'.jpg')
            city = 'pittsburgh'

        info[j][4] = city

    np.savetxt(os.path.join(data_path, 'csv', (area+'.csv')), info, delimiter = ',', fmt='%s')