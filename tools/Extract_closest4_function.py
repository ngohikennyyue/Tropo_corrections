import os
import math
import numpy as np
import datetime
import matplotlib.pyplot as plt
import glob
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist

hgtlvs = [-100, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400
    , 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000
    , 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000
    , 14000, 15000, 16000, 17000, 18000, 19000, 20000, 25000, 30000, 35000, 40000]


# Function that create an interpretor with an assigned parameter
def make_interpretor(ds, para: str):
    x = ds.x.values
    y = ds.y.values
    z = ds.z.values
    data = ds.variables[para].transpose('x', 'y', 'z')
    interpolator = rgi(points=(x, y, z), values=data.values, method='linear', bounds_error=False)
    return interpolator


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
    # Create interpreter for each param
    P_interp = make_interpretor(wm, 'p')
    T_interp = make_interpretor(wm, 't')
    e_interp = make_interpretor(wm, 'e')

    # Interp the param
    df['P'] = (P_interp(loc)).values
    df['T'] = (T_interp(loc)).values
    df['e'] = (e_interp(loc)).values

    return df


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

def extract_param_GNSS_4Closest(df, wm_file_path: str, k=4, vertical=True, fixed_hgt=False):
    Date = np.sort(list(set(df['Date'])))
    file_path = os.getcwd()
    file_name = os.getcwd().split('/')[-1]
    print(file_path, file_name)
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
                data.to_csv(file_path + '/' + file_name + '_PTE_interp.csv', index=False)
            else:
                print('Append to file')
                data.to_csv(file_path + '/' + file_name + '_PTE_interp.csv', mode='a', index=False, header=False)
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
                    data.to_csv(file_path + '/' + file_name + '_PTE_closest_4Nodes_vert_fixed_hgtlvs.csv', index=False)
                else:
                    data.to_csv(file_path + '/' + file_name + '_PTE_closest_4Nodes_vert_fixed_hgtlvs.csv', mode='a',
                                index=False, header=False)
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
                    data.to_csv(file_path + '/' + file_name + '_PTE_closest_4Nodes_vert.csv', index=False)
                else:
                    data.to_csv(file_path + '/' + file_name + '_PTE_closest_4Nodes_vert.csv', mode='a', index=False,
                                header=False)
                print('Extracted date ', i)
    print('Finished extraction.')
