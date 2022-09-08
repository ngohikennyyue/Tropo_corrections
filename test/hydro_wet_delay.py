import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

import rasterio.transform
import tensorflow as tf

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *
from datetime import datetime
import requests
lon_min, lat_min, lon_max, lat_max = -156, 18.9, -155, 19.9
date = '20190714_20190702'
dataset = rasterio.open('../InSAR/Large_scale/Hawaii/Extracted/DEM/SRTM_3arcsec_uncropped.tif')
dem = dataset.read(1)
dem[dem == dataset.get_nodatavals()[0]] = 0
ifg, grid = focus_bound('../InSAR/Large_scale/Hawaii/Extracted/unwrappedPhase/' + date, lon_min, lat_min, lon_max,
                            lat_max)
GPS = pd.read_csv('..InSAR/Hawaii/GPS_station/product/UNRcombinedGPS_ztd.csv')
date1 = GPS[GPS['Date'] == '2019-07-02']
date2 = GPS[GPS['Date'] == '2019-07-14']
wm1 = xr.load_dataset(
    '../InSAR/Large_scale/Hawaii/weather_files/ERA-5_2019_07_02_T04_00_00_15N_26N_162W_150W.nc').interp(
    z=hgtlvs)
wm1 = wm1.where((wm1.x < dataset.bounds.right) &
                (wm1.x > dataset.bounds.left) &
                (wm1.y > dataset.bounds.bottom) &
                (wm1.y < dataset.bounds.top), drop=True)
wm2 = xr.load_dataset(
    '../InSAR/Large_scale/Hawaii/weather_files/ERA-5_2019_07_14_T04_00_00_15N_26N_162W_150W.nc').interp(
    z=hgtlvs)
wm2 = wm2.where((wm2.x < dataset.bounds.right) &
                (wm2.x > dataset.bounds.left) &
                (wm2.y > dataset.bounds.bottom) &
                (wm2.y < dataset.bounds.top), drop=True)

wm1_wet = make_interpretor(wm1, 'wet_total')
wm1_hydro = make_interpretor(wm1, 'hydro_total')
wm2_wet = make_interpretor(wm2, 'wet_total')
wm2_hydro = make_interpretor(wm2, 'hydro_total')
X, Y = np.meshgrid(wm1.x.values, wm1.y.values)
row, col = rasterio.transform.rowcol(dataset.transform, X.ravel(), Y.ravel())
dem = dem[row, col]
loc = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1), dem.reshape(-1, 1)), axis=1)

# Interp to DEM
wm1_wet_int = wm1_wet(loc)
wm1_hydro_int = wm1_hydro(loc)
wm2_wet_int = wm2_wet(loc)
wm2_hydro_int = wm2_hydro(loc)

fig, ax = plt.subplots(2, 2)
ax[0, 0].hist(wm2.hydro_total.values[3, :, :].ravel() - wm1.hydro_total.values[3, :, :].ravel())
ax[0, 0].set_title('Hydro_delay')
ax[0, 1].hist(wm2.wet_total.values[3, :, :].ravel() - wm1.wet_total.values[3, :, :].ravel())
ax[0, 1].set_title('Wet_delay')
ax[1, 0].hist((wm2_hydro_int - wm1_hydro_int).ravel())
ax[1, 1].hist((wm2_wet_int - wm1_wet_int).ravel())
plt.savefig('Plots/Hawaii/hydro_wet_delay', dpi=200)
plt.clf()

plt.scatter(wm2.hydro_total.values[3, :, :].ravel() - wm1.hydro_total.values[3, :, :].ravel(),
            wm2.wet_total.values[3, :, :].ravel() - wm1.wet_total.values[3, :, :].ravel())
plt.xlabel('hydro (m)')
plt.ylabel('wet (m)')
plt.savefig('Plots/Hawaii/hydro_wet_corr', dpi=200)
plt.clf()

plt.scatter(dem, (wm2_hydro_int - wm1_hydro_int).ravel())
plt.xlabel('elevation (m)')
plt.ylabel('hydrostatic delay (m)')
plt.savefig('Plots/Hawaii/hydro_elev', dpi=200)
