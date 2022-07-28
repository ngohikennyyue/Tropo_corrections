import os
import math
import numpy as np
import datetime
import matplotlib.pyplot as plt
import glob
import pandas as pd
import xarray as xr
from datetime import datetime
from datetime import timedelta
# from extract_func.Extract_ee_function import *

hgtlvs = [-100, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400
    , 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000
    , 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000
    , 14000, 15000, 16000, 17000, 18000, 19000, 20000, 25000, 30000, 35000, 40000]


# add time margin for an input datetime
def datetime_offset(date_time, time_for='%Y-%m-%dT%H:%M:%S', margin=5):
    time = date_time
    given_time = datetime.strptime(time, time_for)
    start_time = given_time.strftime(time_for)
    end_time = (given_time + timedelta(days=margin)).strftime(time_for)
    return start_time, end_time


def extract_inter_param(df, workLoc='', wm_file_path: str = '', date_diff=12, from_scratch=False,
                        variables=('P_', 'T_', 'e_')):
    file_name = os.getcwd().split('/')[-1]
    Date = np.sort(list(set(df['Date'])))
    GOES = 'GOES_'
    res = GOES in variables
    print('Extracting params...')
    for i, day in enumerate(Date):
        start, end = datetime_offset(day, time_for='%Y-%m-%d', margin=date_diff)
        print(start + '_' + end)
        if end > Date[-1]:
            break
        else:
            df_start = df[df['Date'] == start]
            df_end = df[df['Date'] == end]
            if len(df_start) != len(df_end):
                df_end = df_end[df_end.ID.isin(df_start.ID)]
                df_start = df_start[df_start.ID.isin(df_end.ID)]
            else:
                pass
            inter_ZTD = df_end[['ZTD']].values - df_start[['ZTD']].values
            date = np.repeat((start + '_' + end), len(inter_ZTD))
            if not from_scratch:
                PTE_start = df_start[
                    df_start.columns[pd.Series(df_start.columns).str.startswith(variables)]].values
                PTE_end = df_end[df_end.columns[pd.Series(df_end.columns).str.startswith(variables)]].values
                inter_PTE = PTE_end - PTE_start
            else:
                if not res:
                    path1 = glob.glob(wm_file_path + 'ERA-5_{date}*[A-Z].nc'.format(date=start))
                    path2 = glob.glob(wm_file_path + 'ERA-5_{date}*[A-Z].nc'.format(date=end))
                    ds1 = xr.load_dataset(" ".join(path1))
                    ds2 = xr.load_dataset("  ".join(path2))

                    x = xr.DataArray(df_start['Lon'].ravel(), dims='x')
                    y = xr.DataArray(df_start['Lat'].ravel(), dims='y')
                    z = xr.DataArray(hgtlvs, dims='z')

                    P1 = pd.DataFrame(ds1.p.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose())
                    T1 = pd.DataFrame(ds1.t.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose())
                    e1 = pd.DataFrame(ds1.e.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose())
                    P2 = pd.DataFrame(ds2.p.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose())
                    T2 = pd.DataFrame(ds2.t.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose())
                    e2 = pd.DataFrame(ds2.e.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose())

                    data1 = pd.concat([P1, T1, e1], axis=1, ignore_index=True).values
                    data2 = pd.concat([P2, T2, e2], axis=1, ignore_index=True).values

                    inter_PTE = data2 - data1

                else:
                    print('At this moment we are not able to extract GOES data. It is still work in progress')
                    break
            dat = pd.DataFrame(
                np.hstack((df_start.ID.values.reshape(-1, 1), date.reshape(-1, 1), inter_ZTD.reshape(-1, 1),
                           df_start.Lat.values.reshape(-1, 1), df_start.Hgt_m.values.reshape(-1, 1),
                           inter_PTE)))
            if i == 0:
                if res:
                    name = ['ID', 'Date', 'inf_ZTD', 'Lat', 'Hgt_m'] + \
                           ['P_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                           ['T_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                           ['e_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                           ['GOES_' + str(i) for i in range(1, 5)]
                else:
                    name = ['ID', 'Date', 'inf_ZTD', 'Lat', 'Hgt_m'] + \
                           ['P_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                           ['T_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                           ['e_' + str(i) for i in range(1, len(hgtlvs) + 1)]
                dat.columns = name
                if res:
                    dat.to_csv(workLoc + file_name + "_Inter_PTE_vert_fixed_hgtlvs_goes.csv", index=False)
                else:
                    dat.to_csv(workLoc + file_name + "_Inter_PTE_vert_fixed_hgtlvs.csv", index=False)
            else:
                if res:
                    dat.to_csv(workLoc + file_name + "_Inter_PTE_vert_fixed_hgtlvs_goes.csv", mode='a', index=False,
                               header=False)
                else:
                    dat.to_csv(workLoc + file_name + "_Inter_PTE_vert_fixed_hgtlvs.csv", mode='a', index=False,
                           header=False)

    print('Finished')
