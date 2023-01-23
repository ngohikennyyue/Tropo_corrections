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
from extract_func.Extract_PTE_function import *


# add time margin for an input datetime
def datetime_offset(date_time, time_for='%Y-%m-%dT%H:%M:%S', margin=5):
    time = date_time
    given_time = datetime.strptime(time, time_for)
    start_time = given_time.strftime(time_for)
    end_time = (given_time + timedelta(days=margin)).strftime(time_for)
    return start_time, end_time


def extract_inter_param(df, slope_path: str, workLoc='', wm_file_path: str = '', date_diff=12, from_scratch=False,
                        int_f=False, variables=('P_', 'T_', 'e_')):
    file_name = os.getcwd().split('/')[-1]
    if int_f:
        file_name = file_name + '_param_inf'
    else:
        pass
    Date = np.sort(list(set(df['Date'])))
    GOES = 'GOES_'
    res = GOES in variables
    slope = pd.read_csv(slope_path)
    slope = slope.dropna()
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
            slope_ex = slope.loc[slope['ID'].isin(df_start.ID)]
            inter_ZTD = df_end[['ZTD']].values - df_start[['ZTD']].values
            start_date = np.repeat(start, len(inter_ZTD))
            end_date = np.repeat(end, len(inter_ZTD))
            if not from_scratch:
                PTE_start = df_start[df_start.columns[pd.Series(df_start.columns).str.startswith(variables)]].drop(
                    ['P_1', 'T_1', 'e_1'], axis=1).reset_index(drop=True)
                PTE_end = df_end[df_end.columns[pd.Series(df_end.columns).str.startswith(variables)]].drop(
                    ['P_1', 'T_1', 'e_1'], axis=1).reset_index(drop=True)
                print(PTE_start.head())
                print(PTE_end.head())
                inter_PTE = pd.concat([PTE_start, PTE_end], axis=1, ignore_index=True).values
            else:
                if not res:
                    path1 = glob.glob(wm_file_path + 'ERA-5_{date}*[A-Z].nc'.format(date=start.replace('-', '_')))
                    path2 = glob.glob(wm_file_path + 'ERA-5_{date}*[A-Z].nc'.format(date=end.replace('-', '_')))
                    ds1 = xr.load_dataset(" ".join(path1))
                    ds2 = xr.load_dataset(" ".join(path2))

                    x = xr.DataArray(df_start['Lon'].ravel(), dims='x')
                    y = xr.DataArray(df_start['Lat'].ravel(), dims='y')
                    z = xr.DataArray(hgtlvs, dims='z')

                    P1 = ds1.p.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose()
                    T1 = ds1.t.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose()
                    e1 = ds1.e.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose()
                    P2 = ds2.p.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose()
                    T2 = ds2.t.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose()
                    e2 = ds2.e.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose()
                    if not int_f:
                        inter_PTE = np.hstack((P1, T1, e1, P2, T2, e2))
                    else:
                        inter_PTE = np.hstack((P1 - P2, T1 - T2, e1 - e2))
                else:
                    print('At this moment we are not able to extract GOES data. It is still work in progress')
                    break
            dat = pd.DataFrame(np.hstack((df_start.ID.values.reshape(-1, 1),
                                          start_date.reshape(-1, 1),
                                          end_date.reshape(-1, 1),
                                          inter_ZTD.reshape(-1, 1),
                                          df_start.Lon.values.reshape(-1, 1),
                                          df_start.Lat.values.reshape(-1, 1),
                                          df_start.Hgt_m.values.reshape(-1, 1),
                                          inter_PTE)))
            if res:
                name = ['ID', 'start_date', 'end_date', 'int_ZTD', 'Lon', 'Lat', 'Hgt_m'] + \
                       ['date1_P_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                       ['date1_T_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                       ['date1_e_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                       ['date2_P_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                       ['date2_T_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                       ['date2_e_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                       ['GOES_' + str(i) for i in range(1, 5)]
            else:
                if not int_f:
                    name = ['ID', 'start_date', 'end_date', 'int_ZTD', 'Lon', 'Lat', 'Hgt_m'] + \
                           ['date1_P_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                           ['date1_T_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                           ['date1_e_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                           ['date2_P_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                           ['date2_T_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                           ['date2_e_' + str(i) for i in range(1, len(hgtlvs) + 1)]
                else:
                    name = ['ID', 'start_date', 'end_date', 'int_ZTD', 'Lon', 'Lat', 'Hgt_m'] + \
                           ['P_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                           ['T_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                           ['e_' + str(i) for i in range(1, len(hgtlvs) + 1)]

            dat.columns = name
            dat = pd.merge(dat, slope_ex[['ID', 'Slope']], how='left', on='ID')
            if i == 0:
                if res:
                    dat.dropna().to_csv(workLoc + file_name + "_Inter_PTE_vert_fixed_hgtlvs_goes.csv", index=False)
                else:
                    dat.dropna().to_csv(workLoc + file_name + "_Inter_PTE_vert_fixed_hgtlvs.csv", index=False)
            else:
                if res:
                    dat.dropna().to_csv(workLoc + file_name + "_Inter_PTE_vert_fixed_hgtlvs_goes.csv", mode='a',
                                        index=False, header=False)
                else:
                    dat.dropna().to_csv(workLoc + file_name + "_Inter_PTE_vert_fixed_hgtlvs.csv", mode='a',
                                        index=False, header=False)

    print('Finished')
