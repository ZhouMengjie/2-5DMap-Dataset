import os
import pandas as pd

if __name__ == "__main__":
    file1 = pd.read_csv(os.path.join('datasets', 'csv', 'trainstreetlearnU_set.csv'), sep=',', header=None)
    file2 = pd.read_csv(os.path.join('datasets', 'csv', 'cmu5kU_set.csv'), sep=',', header=None)

    df = pd.concat([file1, file2])
    print(df)
    df.to_csv(os.path.join('datasets', 'csv', ('trainstreetlearnU_cmu5kU_set'+'.csv')), header=False, index=False, encoding="utf-8")


