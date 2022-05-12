import os
import math
import numpy as np
import datetime
import matplotlib.pyplot as plt
import glob
import pandas as pd
import rasterio
from rasterio.plot import show
import xarray as xr
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist


# Convert radian to cm or m
def convert_rad(data, lamda):
    result = (data * lamda) / (4 * np.pi)
    return result


def standardized(x):
    sc = StandardScaler().fit(x)
    X = sc.transform(x)
    return X, sc


import mpl_scatter_density  # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)


def using_mpl_scatter_density(fig, x, y):
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis)
    fig.colorbar(density, label='Number of points per pixel')


# Extract value from raster and their lat/lon grid of the bounding area
def focus_bound(Raster, left, bottom, right, top):
    from rasterio.windows import Window
    from rasterio.windows import from_bounds
    with rasterio.open(Raster) as src:
        window = from_bounds(left, bottom, right, top, src.transform)
        w = src.read(1, window=window)
        aff = src.window_transform(window=window)
        X = [i * aff[0] + aff[2] for i in range(w.shape[1])]
        Y = [i * aff[4] + aff[5] for i in range(w.shape[0])]
        X, Y = np.meshgrid(X, Y)
        grid = np.stack([X.ravel(), Y.ravel()], axis=-1)
    return w, grid


# Function that create an interpretor with an assigned parameter
def make_interpretor(ds, para: str):
    x = ds.x.values
    y = ds.y.values
    z = ds.z.values
    data = ds.variables[para].transpose('x', 'y', 'z')
    interpolator = rgi(points=(x, y, z), values=data.values, method='linear', bounds_error=False)
    return interpolator


# Function to resample raster to destinated resolution
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling


def Resamp_rasterio(fn: str, left, bottom, right, top, ref):
    with rasterio.open(fn) as src:
        r_width, r_height = ref.shape
        window = from_bounds(left, bottom, right, top, src.transform)
        data = src.read(out_shape=(
            src.count,
            int(r_width),
            int(r_height)),
            resampling=Resampling.cubic, window=window)
        return data[0]


# Function created to extract window data for wet_total and hydro_total
def extractWindowData_wt_ht(ds1, ds2, dem, los, Raster, left, bottom, right, top):
    w, grid = focus_bound(Raster, left, bottom, right, top)
    dem = Resamp_rasterio(dem, left, right, top, bottom, w).ravel()
    los = Resamp_rasterio(los, left, right, top, bottom, w).ravel()
    newGrid = np.hstack([grid, dem.reshape(len(dem), 1)])
    # Create the interpretor
    wet1_interp = make_interpretor(ds1, 'wet_total')
    hydro1_interp = make_interpretor(ds1, 'hydro_total')

    wet2_interp = make_interpretor(ds2, 'wet_total')
    hydro2_interp = make_interpretor(ds2, 'hydro_total')

    # Create parameter raster that interpolate along the topography
    wt1 = wet1_interp(newGrid).reshape(len(w.ravel()), 1)
    ht1 = hydro1_interp(newGrid).reshape(len(w.ravel()), 1)
    wt2 = wet2_interp(newGrid).reshape(len(w.ravel()), 1)
    ht2 = hydro2_interp(newGrid).reshape(len(w.ravel()), 1)

    return wt1, ht1, wt2, ht2, dem.reshape(len(dem), 1), los.reshape(len(los), 1), w.ravel().reshape(len(w.ravel()),
                                                                                                     1), w


def extract_data(ds1, dem, Raster, left, bottom, right, top):
    w, grid = focus_bound(Raster, left, bottom, right, top)
    dem = Resamp_rasterio(dem, left, right, top, bottom, w).ravel()
    newGrid = np.hstack([grid, dem.reshape(len(dem), 1)])
    P1_interp = make_interpretor(ds1, 'p')
    T1_interp = make_interpretor(ds1, 't')
    e1_interp = make_interpretor(ds1, 'e')
    p1 = P1_interp(newGrid).reshape(len(w.ravel()), 1)
    t1 = T1_interp(newGrid).reshape(len(w.ravel()), 1)
    e1 = e1_interp(newGrid).reshape(len(w.ravel()), 1)
    return p1, t1, e1, dem.reshape(len(dem), 1), w


# Function which is able to retrieve obtain the weather param for the GNSS and date
# file path: have to be the file where all the weather model are saved
# df : The list of combined data from downloading GNSS data using RAiDER Package
# Vertical: would like to interpolate everything from vertically from the lat lon
def extract_param_GNSS(df, file_path: str, Vertical=False):
    dataFrame = []
    Date = np.sort(list(set(df['Date'])))
    for i in Date:
        print(i)
        dd = df.loc[df['Date'] == i]
        date = i.replace('-', '_')  # Retrieve the date of the GNSS for NWM

        path_name = glob.glob(file_path + 'ERA-5_{date}*[A-Z].nc'.format(date=date))
        print(path_name)
        try:
            ds = xr.load_dataset(" ".join(path_name))  # xr to read the weather model
            # dd = dd[(dd['Lon'] >= ds.x.min())& (dd['Lon']<= ds.x.max())&(dd['Lat'] >= ds.y.min())& (dd['Lat']<= ds.y.max())]
            loc = dd[['Lon', 'Lat', 'Hgt_m']].values
        except:
            continue
        if Vertical == False:
            # Create interpretor for each param
            P_interp = make_interpretor(ds, 'p')
            T_interp = make_interpretor(ds, 't')
            e_interp = make_interpretor(ds, 'e')

            # Interp the param
            P = (P_interp(loc))
            T = (T_interp(loc))
            e = (e_interp(loc))

            dd['P'] = P
            dd['T'] = T
            dd['e'] = e

            dataFrame.append(dd)
            print('Done', i)

        else:
            # Get coordinate of the GPS station
            x = xr.DataArray(dd['Lon'].ravel(), dims='x')
            y = xr.DataArray(dd['Lat'].ravel(), dims='y')

            # Interp the data
            P = pd.DataFrame(ds.p.interp(x=x, y=y).values.transpose().diagonal().transpose())
            T = pd.DataFrame(ds.t.interp(x=x, y=y).values.transpose().diagonal().transpose())
            e = pd.DataFrame(ds.e.interp(x=x, y=y).values.transpose().diagonal().transpose())
            dataFrame.append(pd.concat([dd.reset_index(drop=True), P, T, e], axis=1).reset_index(drop=True))
            print('Done', i)
    Data = pd.concat(dataFrame).reset_index(drop=True)
    if Vertical == True:
        P_name = ['P_' + str(i) for i in range(1, len(ds.z) + 1)]
        T_name = ['T_' + str(i) for i in range(1, len(ds.z) + 1)]
        e_name = ['e_' + str(i) for i in range(1, len(ds.z) + 1)]
        Data.columns = np.concatenate((df.columns, P_name, T_name, e_name))
        return Data
    else:
        return Data
