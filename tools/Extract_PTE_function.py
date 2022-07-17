import os
import math
import numpy as np
import datetime
import matplotlib.pyplot as plt
import glob
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import xarray as xr
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import PReLU, LeakyReLU, ReLU
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Dense, concatenate, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import PReLU, LeakyReLU, ReLU

hgtlvs = [-100, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400
    , 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000
    , 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000
    , 14000, 15000, 16000, 17000, 18000, 19000, 20000, 25000, 30000, 35000, 40000]


def plot_result(true, predict):
    plt.hist(true - predict)
    plt.xlabel('Residual (m)')
    plt.ylabel('Count')
    plt.show()


# Convert radian to cm or m
def convert_rad(data, lamda):
    result = (data * lamda) / (4 * np.pi)
    return result


def standardized(x, scaler):
    if scaler == 'Standard':
        sc = StandardScaler().fit(x)
        X = sc.transform(x)
    elif scaler == 'MinMax':
        sc = MinMaxScaler().fit(x)
        X = sc.transform(x)
    elif scaler == 'Robust':
        sc = RobustScaler().fit(x)
        X = sc.transform(x)
    else:
        return print("Wrong scaler input")
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
        aff = src.window_transform(window=window)
        X = [i * aff[0] + aff[2] for i in range(r_width)]
        Y = [i * aff[4] + aff[5] for i in range(r_height)]
        X, Y = np.meshgrid(X, Y)
        grid = np.stack([X.ravel(), Y.ravel()], axis=-1)
        return data[0], grid


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


def PTE_interp(wm, loc, df):
    # Create interpretor for each param
    P_interp = make_interpretor(wm, 'p')
    T_interp = make_interpretor(wm, 't')
    e_interp = make_interpretor(wm, 'e')

    # Interp the param
    df['P'] = (P_interp(loc)).values
    df['T'] = (T_interp(loc)).values
    df['e'] = (e_interp(loc)).values

    return df


# Prepare data and scale them accordingly
# train: training data
# test: testing data
# variable: data that is of interest from training set and testing set
def process_PTE_data(train, test, variable):
    if not isinstance(variable, (str, list, tuple)):
        raise TypeError("Variable not of type str")
    else:
        cs = MinMaxScaler()
        trainX = cs.fit_transform(train[train.columns[pd.Series(train.columns).str.startswith(variable)]])
        testX = cs.transform(test[test.columns[pd.Series(test.columns).str.startswith(variable)]])
        return trainX, testX, cs


# Create the sequential model with specific nodes and dimensions.
# dim: input dimensions
# nodes: a list of node in each layer e.g. [256, 128,64, 32]
# regress: return the last node as the output in linear
def create_mlp(dim, nodes, regress=False):
    model = Sequential()
    for i, n in enumerate(nodes):
        if i == 0:
            model.add(Dense(n, input_dim=dim, activation='relu'))
        else:
            model.add(Dense(n, activation='relu'))
    if regress:
        model.add(Dense(1, activation='linear'))

    return model


# Get an array of distance from the each pixel to the weather model nodes
def get_distance(grid, ds):
    lons = ds.x.values
    lats = ds.y.values
    X, Y = np.meshgrid(lons, lats)
    dist = cdist(grid, np.stack([X.ravel(), Y.ravel()], axis=-1))
    return dist


# Get the actual index from the weather model parameters
def get_index(dist, shape, k):
    ind = np.argpartition(dist, k)
    index = ind[:, :k]
    clost_dist = np.take_along_axis(dist, index, axis=1)
    return np.stack(np.unravel_index(ind[:, :k], shape), axis=-1).reshape(len(index) * k, 2), clost_dist


# Function which is able to retrieve obtain the weather param for the GNSS and date
# file path: have to be the file where all the weather model are saved
# df : The list of combined data from downloading GNSS data using RAiDER Package
# Vertical: would like to interpolate everything from vertically from the lat lon

def extract_param_GNSS(df, wm_file_path: str, workLoc: str = '', vertical=False, fixed_hgt=False):
    Date = np.sort(list(set(df['Date'])))
    for num, i in enumerate(Date):
        print(i)
        dd = df.loc[df['Date'] == i]
        date = i.replace('-', '_')  # Retrieve the date of the GNSS for NWM

        path_name = glob.glob(wm_file_path + 'ERA-5_{date}*[A-Z].nc'.format(date=date))
        print(path_name)
        try:
            ds = xr.load_dataset(" ".join(path_name))  # xr to read the weather model
            # dd = dd[(dd['Lon'] >= ds.x.min())& (dd['Lon']<= ds.x.max())&(dd['Lat'] >= ds.y.min())& (dd['Lat']<= ds.y.max())]
            loc = dd[['Lon', 'Lat', 'Hgt_m']].values
        except:
            print('Can not read weather model')
            continue
        if not vertical:
            # Create interpreter for each param
            data = PTE_interp(ds, loc, dd)
            if num == 0:
                print('Write file')
                data.to_csv(workLoc + 'PTE_interp.csv', index=False)
            else:
                print('Append to file')
                data.to_csv(workLoc + 'PTE_interp.csv', mode='a', index=False, header=False)
            print('Done', i)

        else:
            if fixed_hgt:
                # Get coordinate of the GPS station
                x = xr.DataArray(dd['Lon'].ravel(), dims='x')
                y = xr.DataArray(dd['Lat'].ravel(), dims='y')
                z = xr.DataArray(hgtlvs, dims='z')

                # Interp and extract data
                P = pd.DataFrame(ds.p.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose())
                T = pd.DataFrame(ds.t.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose())
                e = pd.DataFrame(ds.e.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose())
                data = pd.concat([dd.reset_index(drop=True), P, T, e], axis=1, ignore_index=True)

                if num == 0:
                    name = ['P_' + str(i) for i in range(1, len(z) + 1)] + ['T_' + str(i) for i in
                                                                            range(1, len(z) + 1)] + ['e_' + str(i) for i
                                                                                                     in range(1,
                                                                                                              len(z) + 1)]
                    data.columns = np.concatenate((df.columns, name))
                    data.to_csv(workLoc + 'PTE_vert_fixed_hgtlvs.csv', index=False)
                else:
                    data.to_csv(workLoc + 'PTE_vert_fixed_hgtlvs.csv', mode='a', index=False, header=False)
                print('Done', i)

            else:
                # Get coordinate of the GPS station
                x = xr.DataArray(dd['Lon'].ravel(), dims='x')
                y = xr.DataArray(dd['Lat'].ravel(), dims='y')
                z = ds.z.values

                # Interp and extract data
                P = pd.DataFrame(ds.p.interp(x=x, y=y).values.transpose().diagonal().transpose())
                T = pd.DataFrame(ds.t.interp(x=x, y=y).values.transpose().diagonal().transpose())
                e = pd.DataFrame(ds.e.interp(x=x, y=y).values.transpose().diagonal().transpose())
                data = pd.concat([dd.reset_index(drop=True), P, T, e], axis=1, ignore_index=True)

                if num == 0:
                    name = ['P_' + str(i) for i in range(1, len(z) + 1)] + ['T_' + str(i) for i in
                                                                            range(1, len(z) + 1)] + ['e_' + str(i) for i
                                                                                                     in range(1,
                                                                                                              len(z) + 1)]
                    data.columns = np.concatenate((df.columns, name))
                    data.to_csv(workLoc + 'PTE_vert.csv', index=False)
                else:
                    data.to_csv(workLoc + 'PTE_vert.csv', mode='a', index=False, header=False)
                print('Done', i)
    print('Finished extraction.')


def extract_param_GNSS_4Closest(df, wm_file_path: str, workLoc: str = '', k=4, vertical=True, fixed_hgt=False):
    Date = np.sort(list(set(df['Date'])))
    for num, i in enumerate(Date):
        print(i)
        dd = df.loc[df['Date'] == i]
        date = i.replace('-', '_')  # Retrieve the date of the GNSS for NWM

        path_name = glob.glob(wm_file_path + 'ERA-5_{date}*[A-Z].nc'.format(date=date))
        print(path_name)
        try:
            ds = xr.load_dataset(" ".join(path_name))  # xr to read the weather model
            loc = dd[['Lon', 'Lat', 'Hgt_m']].values
            dist = get_distance(loc[:, 0:-1], ds)
            index, close4_dist = get_index(dist, (len(ds.x), len(ds.y)), k)
            row = index[:, 0].astype(int)
            col = index[:, 1].astype(int)
        except:
            print('Can not read weather model')
            continue
        if not vertical:  # Might need to cancel this
            # Create interpreter for each param
            data = PTE_interp(ds, loc, dd)
            if num == 0:
                print('Write file')
                data.to_csv(workLoc + 'PTE_interp.csv', index=False)
            else:
                print('Append to file')
                data.to_csv(workLoc + 'PTE_interp.csv', mode='a', index=False, header=False)
            print('Done', i)

        else:
            if fixed_hgt:
                z = xr.DataArray(hgtlvs, dims='z')

                P = pd.DataFrame(ds.p.interp(z=z).transpose().values[row, col, :].reshape(len(loc), len(hgtlvs) * k))
                T = pd.DataFrame(ds.t.interp(z=z).transpose().values[row, col, :].reshape(len(loc), len(hgtlvs) * k))
                e = pd.DataFrame(ds.e.interp(z=z).transpose().values[row, col, :].reshape(len(loc), len(hgtlvs) * k))
                data = pd.concat([dd.reset_index(drop=True), P, T, e, pd.DataFrame(close4_dist)], axis=1,
                                 ignore_index=True)
                if num == 0:
                    name = ['P_' + str(i) for i in range(1, len(hgtlvs) * k + 1)] + ['T_' + str(i) for i in
                                                                                     range(1, len(hgtlvs) * k + 1)] + [
                               'e_' + str(i) for i in range(1, len(hgtlvs) * k + 1)] + ['dist_' + str(i) for i in
                                                                                        range(1, k + 1)]
                    data.columns = np.concatenate((df.columns, name))
                    data.to_csv(workLoc + 'PTE_closest_4Nodes_vert_fixed_hgtlvs.csv', index=False)
                else:
                    data.to_csv(workLoc + 'PTE_closest_4Nodes_vert_fixed_hgtlvs.csv', mode='a', index=False,
                                header=False)
                print('Extracted date ', i)

            else:
                P = pd.DataFrame(ds.p.transpose().values[row, col, :].reshape(len(loc), len(ds.z) * k))
                T = pd.DataFrame(ds.t.transpose().values[row, col, :].reshape(len(loc), len(ds.z) * k))
                e = pd.DataFrame(ds.e.transpose().values[row, col, :].reshape(len(loc), len(ds.z) * k))
                data = pd.concat([dd.reset_index(drop=True), P, T, e, pd.DataFrame(close4_dist)], axis=1,
                                 ignore_index=True)
                if num == 0:
                    name = ['P_' + str(i) for i in range(1, len(ds.z) * k + 1)] + ['T_' + str(i) for i in
                                                                                   range(1, len(ds.z) * k + 1)] + [
                               'e_' + str(i) for i in range(1, len(ds.z) * k + 1)] + ['dist_' + str(i) for i in
                                                                                      range(1, k + 1)]
                    data.columns = np.concatenate((df.columns, name))
                    data.to_csv(workLoc + 'PTE_closest_4Nodes_vert.csv', index=False)
                else:
                    data.to_csv(workLoc + 'PTE_closest_4Nodes_vert.csv', mode='a', index=False, header=False)
                print('Extracted date ', i)
    print('Finished extraction.')


def extract_wethydro_GNSS(df, wm_file_path: str, fixed_hgt=False):
    Date = np.sort(list(set(df['Date'])))
    file_path = os.getcwd()
    file_name = os.getcwd().split('/')[-1]
    for num, i in enumerate(Date):
        print(i)
        dd = df.loc[df['Date'] == i]
        date = i.replace('-', '_')  # Retrieve the date of the GNSS for NWM

        path_name = glob.glob(wm_file_path + 'ERA-5_{date}*[A-Z].nc'.format(date=date))
        print(path_name)
        try:
            ds = xr.load_dataset(" ".join(path_name))  # xr to read the weather model
            # loc = dd[['Lon', 'Lat', 'Hgt_m']].values
        except:
            print('Can not read weather model')
            continue
        if not fixed_hgt:
            # Get coordinate of the GPS station
            x = xr.DataArray(dd['Lon'].ravel(), dims='x')
            y = xr.DataArray(dd['Lat'].ravel(), dims='y')
            z = ds.z.values

            # Interp and extract data
            wet = ds.wet.interp(x=x, y=y).values.transpose().diagonal().transpose()
            hydro = ds.hydro.interp(x=x, y=y).values.transpose().diagonal().transpose()
            total = pd.DataFrame(wet + hydro)
            data = pd.concat([dd.reset_index(drop=True), total], axis=1, ignore_index=True)

            if num == 0:
                name = ['total_' + str(i) for i in range(1, len(z) + 1)]
                data.columns = np.concatenate((df.columns, name))
                data.to_csv(file_path + '/' + file_name + '_nodes_delay_vert.csv', index=False)
            else:
                data.to_csv(file_path + '/' + file_name + '_nodes_delay_vert.csv', mode='a', index=False, header=False)
            print('Done', i)

        else:
            # Get coordinate of the GPS station
            x = xr.DataArray(dd['Lon'].ravel(), dims='x')
            y = xr.DataArray(dd['Lat'].ravel(), dims='y')
            z = xr.DataArray(hgtlvs, dims='z')

            # Interp and extract data
            wet = ds.wet.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose()
            hydro = ds.hydro.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose()
            total = pd.DataFrame(wet + hydro)
            data = pd.concat([dd.reset_index(drop=True), total], axis=1, ignore_index=True)

            if num == 0:
                name = ['total_' + str(i) for i in range(1, len(z) + 1)]
                data.columns = np.concatenate((df.columns, name))
                data.to_csv(file_path + '/' + file_name + '_node_delay_vert_fixed_hgtlvs.csv', index=False)
            else:
                data.to_csv(file_path + '/' + file_name + '_node_delay_vert_fixed_hgtlvs.csv', mode='a', index=False,
                            header=False)
            print('Done', i)
    print('Finished extraction.')
