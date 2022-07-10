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
    return np.stack(np.unravel_index(ind[:, :k], shape), axis=-1).reshape(len(index) * 4, 2), clost_dist


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


# Function which is able to retrieve obtain the weather param for the GNSS and date
# file path: have to be the file where all the weather model are saved
# df : The list of combined data from downloading GNSS data using RAiDER Package
# Vertical: would like to interpolate everything from vertically from the lat lon

def extract_param_GNSS_4Closest(df, wm_file_path: str, workLoc: str = '', k=4, vertical=False, fixed_hgt=False):
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
            dist = get_distance(loc[:, 0:-1],ds)
            index, close4_dist = get_index(dist, (len(ds.x), len(ds.y)), k)
            row = index[:, 0].astype(int)
            col = index[:, 1].astype(int)
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
                P = pd.DataFrame(ds.p.transpose().values[row, col, :].reshape(len(loc), len(hgtlvs) * k))
                T = pd.DataFrame(ds.t.transpose().values[row, col, :].reshape(len(loc), len(hgtlvs) * k))
                e = pd.DataFrame(ds.e.transpose().values[row, col, :].reshape(len(loc), len(hgtlvs) * k))
                data = pd.concat([dd.reset_index(drop=True), P, T, e], axis=1, ignore_index=True)
                if num == 0:
                    name = ['P_' + str(i) for i in range(1, len(hgtlvs) * k + 1)] + ['T_' + str(i) for i in
                                                                                     range(1, len(hgtlvs) * k + 1)] + [
                               'e_' + str(i) for i in range(1, len(hgtlvs) * k + 1)]
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
                data = pd.concat([dd.reset_index(drop=True), P, T, e], axis=1, ignore_index=True)
                if num == 0:
                    name = ['P_' + str(i) for i in range(1, len(ds.z) * k + 1)] + ['T_' + str(i) for i in
                                                                                   range(1, len(ds.z) * k + 1)] + [
                               'e_' + str(i) for i in range(1, len(ds.z) * k + 1)]
                    data.columns = np.concatenate((df.columns, name))
                    data.to_csv(workLoc + 'PTE_closest_4Nodes_vert.csv', index=False)
                else:
                    data.to_csv(workLoc + 'PTE_closest_4Nodes_vert.csv', mode='a', index=False, header=False)
                print('Extracted date ', i)
    print('Finished extraction.')
