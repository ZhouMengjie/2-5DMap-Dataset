import os
import tqdm
import numpy as np
import pandas as pd

if __name__ == "__main__":
    area = 'trainstreetlearnU'
    data_path = os.path.join(os.getcwd(), 'datasets')
    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_set.csv')), sep=',', header=None)
    info = data.values

    for j in tqdm.tqdm(range(info.shape[0])):
        global_idx = int(info[j][1])-1 # for train only
        info[j][1] = global_idx

    np.savetxt(os.path.join(data_path, 'csv', (area+'_set.csv')), info, delimiter = ',', fmt='%s')