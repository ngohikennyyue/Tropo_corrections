import os
import numpy as np
import glob
import rasterio
import matplotlib.pyplot as plt
import math
import pandas as pd

slope = '../../DEM/slope.tif'
GPS_station = pd.read_csv('GPS_stations/product2/gnssStationList_overbbox.csv')
xs = GPS_station['Lon'].values
ys = GPS_station['Lat'].values

with rasterio.open(slope) as src:
    rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
    slope_val = src.read(1)
    slope_val[slope_val<0.0] = np.nan
    dats = slope_val[rows, cols]
    GPS_station['Slope'] = dats.reshape(-1,1)

GPS_station.to_csv('GPS_ztd_slope.csv', index=False)

print('Finished')