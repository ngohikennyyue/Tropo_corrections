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

hgtlvs = [-100, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400,
          2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000,
          5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000,
          14000, 15000, 16000, 17000, 18000, 19000, 20000, 25000, 30000, 35000, 40000]


# lon_min, lat_min, lon_max, lat_max = -155.9, 18.9, -154.9, 19.9
# ifg_path: file path of the ifg file
# wm_file_path: file path of weather model (best with the hour before and after of needed time)
# dem_path: file path of DEM file
# los_path: file path of LOS file
# time: approximate time of the ifg acquisition
# lon_min, lat_min, lon_max, lat_max: bounding area
# samp: samp default by every 100 item
# years: years would need to extract in a list e.g. [2019,2020]
def extract_ifg_param(ifg_path: str, wm_file_path: str, dem_path: str, los_path: str, slope_path: str, coh_path: str,
                      time: str, lon_min, lat_min, lon_max, lat_max, name=None, years=None, samp=100, inter=False):
    file_name = os.getcwd().split('/')[-1]
    if years is None:
        Ifg = glob.glob(ifg_path + '*[0-9]')
    else:
        Ifg = np.concatenate([glob.glob(ifg_path + '{}*[0-9]'.format(year)) for year in years])
    Ifg.sort()
    print('Begin extracting')
    for i, ifg in enumerate(Ifg):
        date = ifg.split('/')[-1]
        print('Date pair: ', date)
        date1, date2 = get_datetime(date)
        ifg, grid = focus_bound(ifg, lon_min, lat_min, lon_max, lat_max)
        dem, dem_grid = Resamp_rasterio(dem_path, lon_min,
                                        lat_min, lon_max, lat_max, ifg)
        los, _ = Resamp_rasterio(los_path, lon_min, lat_min, lon_max, lat_max, ifg)
        slope, _ = Resamp_rasterio(slope_path, lon_min, lat_min, lon_max, lat_max, ifg)
        coh, _ = Resamp_rasterio(coh_path + date + '.vrt', lon_min, lat_min, lon_max, lat_max, ifg)

        # Coherence mask
        coh[coh >= 0.8] = 1.0
        coh[coh < 0.8] = np.nan

        # DEM mask
        mask = dem.copy()
        dem[dem <= 0] = np.nan
        mask[mask > 0] = 1
        mask[mask <= 0] = np.nan

        ifg = ifg * mask
        ifg = ifg * coh
        ifg = convert_rad(ifg, 5.6 / 100)
        ifg = (ifg - np.nanmean(ifg)) * np.cos(np.radians(los))  # convert from Slant to Zenith
        # Get Weather models for the two date and time by an hour difference
        WM1, wm1, minute = getWM(date1, time, wmLoc=wm_file_path)
        WM2, wm2, _ = getWM(date2, time, wmLoc=wm_file_path)
        P1, T1, E1 = interpByTime(WM1, wm1, minute, 'all')
        p1, t1, e1 = interpByTime(WM2, wm2, minute, 'all')

        # Get all the x,y coordinate
        x = list(set(dem_grid[:, 0]))
        x.sort()
        y = list(set(dem_grid[:, 1]))
        y.sort()

        # Day1 WM (PTe) parameters
        P1 = P1.interp(x=x, y=y, z=hgtlvs).values
        T1 = T1.interp(x=x, y=y, z=hgtlvs).values
        E1 = E1.interp(x=x, y=y, z=hgtlvs).values

        # Day2 WM (PTe) parameters
        p1 = p1.interp(x=x, y=y, z=hgtlvs).values
        t1 = t1.interp(x=x, y=y, z=hgtlvs).values
        e1 = e1.interp(x=x, y=y, z=hgtlvs).values
        if inter:
            # Get the interferometric P,T,e
            P = (P1 - p1).transpose().reshape((P1.shape[-1] * P1.shape[1], len(hgtlvs)))
            T = (T1 - t1).transpose().reshape((T1.shape[-1] * T1.shape[1], len(hgtlvs)))
            e = (E1 - e1).transpose().reshape((E1.shape[-1] * E1.shape[1], len(hgtlvs)))
            inf_dat = pd.DataFrame(
                np.hstack((np.repeat(date1, len(los.ravel())).reshape(-1, 1)[::samp],
                           np.repeat(date2, len(los.ravel())).reshape(-1, 1)[::samp],
                           los.reshape(-1, 1)[::samp],
                           ifg.reshape(-1, 1)[::samp],
                           dem_grid[::samp],
                           dem.ravel().reshape(-1, 1)[::samp],
                           P[::samp, :],
                           T[::samp, :],
                           e[::samp, :],
                           slope.reshape(-1, 1)[::samp]))).dropna()
        else:
            # Get all the P,T,e for both date
            inf_dat = pd.DataFrame(np.hstack((
                np.repeat(date1, len(los.ravel())).reshape(-1, 1)[::samp],
                np.repeat(date2, len(los.ravel())).reshape(-1, 1)[::samp],
                los.reshape(-1, 1)[::samp],
                ifg.reshape(-1, 1)[::samp],
                dem_grid[::samp],
                dem.ravel().reshape(-1, 1)[::samp],
                P1.transpose().reshape((P1.shape[-1] * P1.shape[1], len(hgtlvs)))[::samp, :],
                T1.transpose().reshape((T1.shape[-1] * T1.shape[1], len(hgtlvs)))[::samp, :],
                E1.transpose().reshape((E1.shape[-1] * E1.shape[1], len(hgtlvs)))[::samp, :],
                p1.transpose().reshape((p1.shape[-1] * p1.shape[1], len(hgtlvs)))[::samp, :],
                t1.transpose().reshape((t1.shape[-1] * t1.shape[1], len(hgtlvs)))[::samp, :],
                e1.transpose().reshape((e1.shape[-1] * e1.shape[1], len(hgtlvs)))[::samp, :],
                slope.reshape(-1, 1)[::samp]))).dropna()
        inf_dat = inf_dat.dropna()
        if i == 0:
            print('Type:', type(inf_dat))
            print('Sample:', inf_dat)
            print('Length of data: ', len(inf_dat))
            if inter:
                inf_dat.columns = ['date1', 'date2', 'los', 'ifg', 'Lon', 'Lat', 'Hgt_m'] + \
                                  ['P_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                                  ['T_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                                  ['e_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                                  ['slope']
                inf_dat.to_csv(file_name + '_' + name + '_ifg_PTE_fixed_hgtlvs_inter.csv', index=False)
            else:
                inf_dat.columns = ['date1', 'date2', 'los', 'ifg', 'Lon', 'Lat', 'Hgt_m'] + \
                                  ['date1_P_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                                  ['date1_T_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                                  ['date1_e_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                                  ['date2_P_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                                  ['date2_T_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                                  ['date2_e_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                                  ['slope']
                inf_dat.to_csv(file_name + '_' + name + '_ifg_PTE_fixed_hgtlvs.csv', index=False)
        else:
            print('Type:', type(inf_dat))
            print('Length of data: ', len(inf_dat))
            if inter:
                inf_dat.to_csv(file_name + '_' + name + '_ifg_PTE_fixed_hgtlvs_inter.csv', index=False, mode='a', header=False)
            else:
                inf_dat.to_csv(file_name + '_' + name + '_ifg_PTE_fixed_hgtlvs.csv', index=False, mode='a', header=False)
