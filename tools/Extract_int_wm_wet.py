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
                        variables=('P_', 'T_', 'e_')):
    file_name = os.getcwd().split('/')[-1]
    Date = np.sort(list(set(df['Date'])))
    GOES = 'GOES_'
    res = GOES in variables
    # slope = pd.read_csv(slope_path)
    # slope = slope.dropna()
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
            # slope_ex = slope.loc[slope['ID'].isin(df_start.ID)]
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
                inter_hywt = pd.concat([PTE_start, PTE_end], axis=1, ignore_index=True).values
            else:
                if not res:
                    path1 = glob.glob(wm_file_path + 'ERA-5_{date}*[A-Z].nc'.format(date=start.replace('-', '_')))
                    path2 = glob.glob(wm_file_path + 'ERA-5_{date}*[A-Z].nc'.format(date=end.replace('-', '_')))
                    ds1 = xr.load_dataset(" ".join(path1))
                    ds2 = xr.load_dataset(" ".join(path2))

                    # Make interpretor
                    hydro1_interp = make_interpretor(ds1, para='hydro_total')
                    wet1_interp = make_interpretor(ds1, para='wet_total')
                    hydro2_interp = make_interpretor(ds2, para='hydro_total')
                    wet2_interp = make_interpretor(ds2, para='wet_total')

                    loc = df_start[['Lon', 'Lat', 'Hgt_m']].values

                    hydro1 = hydro1_interp(loc)
                    wet1 = wet1_interp(loc)
                    hydro2 = hydro2_interp(loc)
                    wet2 = wet2_interp(loc)

                    inter_hywt = pd.concat([hydro1, wet1, hydro2, wet2], axis=1, ignore_index=True).values
                else:
                    print('At this moment we are not able to extract GOES data. It is work in progress')
                    break
            dat = pd.DataFrame(np.hstack((df_start.ID.values.reshape(-1, 1),
                                          start_date.reshape(-1, 1),
                                          end_date.reshape(-1, 1),
                                          inter_ZTD.reshape(-1, 1),
                                          df_start.Lon.values.reshape(-1, 1),
                                          df_start.Lat.values.reshape(-1, 1),
                                          df_start.Hgt_m.values.reshape(-1, 1),
                                          inter_hywt)))
            if res:
                name = ['ID', 'start_date', 'end_date', 'int_ZTD', 'Lon', 'Lat', 'Hgt_m'] + \
                       ['hydro1', 'wet1', 'hydro2', 'wet2'] + \
                       ['GOES_' + str(i) for i in range(1, 5)]
            else:
                name = ['ID', 'start_date', 'end_date', 'int_ZTD', 'Lon', 'Lat', 'Hgt_m'] + \
                       ['hydro1', 'wet1', 'hydro2', 'wet2']
            dat.columns = name
            # dat = pd.merge(dat, slope_ex[['ID', 'Slope']], how='left', on='ID')
            if i == 0:
                if res:
                    dat.dropna().to_csv(workLoc + file_name + "_Inter_hywt_interp_goes.csv", index=False)
                else:
                    dat.dropna().to_csv(workLoc + file_name + "_Inter_hywt_interp.csv", index=False)
            else:
                if res:
                    dat.dropna().to_csv(workLoc + file_name + "_Inter_hywt_interp_goes.csv", mode='a',
                                        index=False, header=False)
                else:
                    dat.dropna().to_csv(workLoc + file_name + "_Inter_hywt_interp.csv", mode='a',
                                        index=False, header=False)

    print('Finished')
