import os
import pandas as pd

if __name__ == "__main__":
    area = 'cmu5k'
    data_path = os.path.join(os.getcwd(), 'datasets')
    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_set.csv')), sep=',', header=None)
    d_data = data.drop([0,1,2,3,6], axis=1)
    d_data.to_csv(os.path.join(data_path, 'csv', (area+'_set.csv')), header=False, index=False, encoding="utf-8")