import pandas as pd
import numpy as np


def join_slope(file_path: str, slope_path: str):
    name = file_path.split('.')[0]
    df = pd.read_csv(file_path)
    df = df.dropna()
    slope = pd.read_csv(slope_path)
    slope = slope.dropna()
    Date = np.sort(list(set(df['Date'])))

    for i, day in enumerate(Date):
        dd = df[df['Date'] == day]
        result = pd.merge(dd, slope[['ID', 'Slope']], how='left', on='ID')
        result = result.dropna()
        print(day, len(result))
        if i == 0:
            result.to_csv(name + '_slope.csv', index=False)
        else:
            result.to_csv(name + '_slope.csv', mode='a', index=False, header=False)

    print('DONE')


join_slope('PTE_vert_fixed_hgtlvs.csv', 'GPS_ztd_slope.csv')
