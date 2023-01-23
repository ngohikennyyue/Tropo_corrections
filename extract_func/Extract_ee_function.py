import glob
import os
from datetime import datetime

import ee
import folium
import numpy as np
import pandas as pd
from scipy import ndimage

from extract_func.Extract_PTE_function import *

# Trigger the authentication flow.
# ee.Authenticate()
# ee.Initialize()

def applyScaleAndOffset(image, band):
    bandname = 'CMI_C' + str(100 + band)[-2:]
    offset = ee.Number(image.get(bandname + '_offset'))
    scale = ee.Number(image.get(bandname + '_scale'))
    bands = image.select(bandname).multiply(scale).add(offset)
    # dqfname =  'DQF_C' + str(100 + band)[-2:]
    # quality = image.select(dqfname)
    return ee.Image(bands)


Num_bands = 33


# Function use when selected bands for image
def applyScaleAndOffset_selected(image):
    band = image.bandNames().getInfo()
    bands = [None] * len(band)
    for i, b in enumerate(band):
        offset = ee.Number(image.get(b + '_offset'))
        scale = ee.Number(image.get(b + '_scale'))
        bands[i] = image.select(b).multiply(scale).add(offset)
        # num = b.split('_')[-1][-2:]
        # dqfname =  'DQF_C' + num
        # bands[i*2 + 1] = image.select(dqfname)
    return ee.Image(ee.Image(bands).copyProperties(image, image.propertyNames()))


def applyScaleAndOffset_all(image):
    bands = [None] * Num_bands
    for i in range(1, 17):
        bandname = 'CMI_C' + str(100 + i)[-2:]
        offset = ee.Number(image.get(bandname + '_offset'))
        scale = ee.Number(image.get(bandname + '_scale'))
        bands[(i - 1) * 2] = image.select(bandname).multiply(scale).add(offset)
        dqfname = 'DQF_C' + str(100 + i)[-2:]
        bands[(i - 1) * 2 + 1] = image.select(dqfname)
    BLUE_BAND_INDEX = (1 - 1) * 2
    RED_BAND_INDEX = (2 - 1) * 2
    VEGGIE_BAND_INDEX = (3 - 1) * 2
    GREEN_BAND_INDEX = Num_bands - 1
    # Bah, Gunshor, Schmit, Generation of GOES-16 True Color Imagery without a
    # Green Band, 2018. https://doi.org/10.1029/2018EA000379
    # Green = 0.45 * Red + 0.10 * NIR + 0.45 * Blue
    green1 = bands[RED_BAND_INDEX].multiply(0.45)
    green2 = bands[VEGGIE_BAND_INDEX].multiply(0.10)
    green3 = bands[BLUE_BAND_INDEX].multiply(0.45)
    green = green1.add(green2).add(green3)
    bands[GREEN_BAND_INDEX] = green.rename('GREEN')
    return ee.Image(ee.Image(bands).copyProperties(image, image.propertyNames()))


def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)


# to convert a google earth engine image to a python array
# Credit: Ryan's Github
def to_array(img, aoi):
    band_arrs = img.sampleRectangle(region=aoi)

    band_names = img.bandNames().getInfo()

    for kk in range(len(band_names)):
        if kk == 0:
            dat1 = np.array(band_arrs.get(band_names[kk]).getInfo())
            dat_full = np.zeros((dat1.shape[0], dat1.shape[1], len(band_names)))
            dat_full[:, :, kk] = dat1
        else:
            dat = np.array(band_arrs.get(band_names[kk]).getInfo())
            dat_full[:, :, kk] = dat
    return dat_full


# create an earth engine geometry polygon
def addGeometry(min_lon, max_lon, min_lat, max_lat):
    geom = ee.Geometry.Polygon(
        [[[min_lon, max_lat],
          [min_lon, min_lat],
          [max_lon, min_lat],
          [max_lon, max_lat]]])
    return geom


# add time margin for an input datetime
def datetime_margin(date_time, time_for='%Y-%m-%dT%H:%M:%S', margin=5):
    from datetime import datetime, timedelta
    time = date_time
    given_time = datetime.strptime(time, time_for)
    start_time = (given_time - timedelta(minutes=margin)).strftime(time_for)
    end_time = (given_time + timedelta(minutes=margin)).strftime(time_for)
    return start_time, end_time


# datetime: '2019-01-01T11:05:00'
# geometry: aoi can be created with addGeometry function
def get_GOES16_image(datetime, geometry=None):
    date1, date2 = datetime_margin(datetime)
    GOES = ee.ImageCollection('NOAA/GOES/16/MCMIPC')
    if geometry is None:
        GOES_img = GOES.filterDate(date1, date2).first()
    else:
        GOES_img = GOES.filterDate(date1, date2).filterBounds(geometry).first().clip(geometry)
    return GOES_img


def get_GOES17_image(datetime, geometry=None):
    date1, date2 = datetime_margin(datetime)
    GOES = ee.ImageCollection('NOAA/GOES/17/MCMIPC')
    if geometry is None:
        GOES_img = GOES.filterDate(date1, date2).first()
    else:
        GOES_img = GOES.filterDate(date1, date2).filterBounds(geometry).first().clip(geometry)
    return GOES_img


# convert list of coordinates of GPS stations into feature collection
# station: DataFrame of GPS station in the order of [Lon, Lat]
def list2features(df):
    features = []
    for i in df.index:
        x, y = df.Lon[i], df.Lat[i]
        latlong = [x, y]
        g = ee.Geometry.Point(latlong)
        feature = ee.Feature(g)
        features.append(feature)

    ee_object = ee.FeatureCollection(features)
    return ee_object


# Extract values from multispectral image by the point of interest and bands
# img: Image of interest
# points_list: list of [lon, lat]
# bands: list of band names that of interest.
def sampFeat2array(img, points_list, bands):
    multi_point = list2features(points_list)
    ft = img.reduceRegions(multi_point, ee.Reducer.first(), 10)
    try:
        data_full = np.hstack([
            np.array(ft.toList(len(points_list)).map(lambda feature: ee.Feature(feature).get(band)).getInfo()).reshape(
                -1, 1) for band in bands])
    except ee.ee_exception.EEException:
        data_full = np.zeros((len(points_list), len(bands)))
    return data_full


# add time margin for an input datetime
def datetime_offset(date_time, time_for='%Y-%m-%dT%H:%M:%S', margin=5):
    time = date_time
    given_time = datetime.strptime(time, time_for)
    start_time = given_time.strftime(time_for)
    end_time = (given_time + timedelta(days=margin)).strftime(time_for)
    return start_time, end_time


# function to extract bands value with existed database
# df: dataframe of the extracted P,T,e values with GPS station coordinate
# time: str object of time of interested eg. '11:00:00'
# geometry: AOI can be created with addGeometry function
# bands: bands that are of interested in a list []
def extract_param(GNSS_file_path: str, wm_path: str, time: str, file_name: str, bands=None):
    if bands is None:
        bands = ['CMI_C08', 'CMI_C09', 'CMI_C10']
    df = pd.read_csv(GNSS_file_path)
    df = df[(df['sigZTD'] < 0.01) & (df['Date'] > '2017-07-12')]
    Date = np.sort(list(set(df['Date'])))
    for i, day in enumerate(Date):
        cur_df = df[df['Date'] == day]
        loc = cur_df[['Lon', 'Lat', 'Hgt_m']].values
        # Get weather models
        if datetime.strptime(time, 'T%H_%M_%S').minute != 0:
            WM = getWM(day.replace('-', '_'), time, wm_path)
            # Interp values by time
            hydro_total, wet_total = interpByTime(WM[0], WM[1], WM[2], 'hydro_total'), interpByTime(WM[0], WM[1], WM[2],
                                                                                                    'wet_total')

            P, T, e = interpByTime(WM[0], WM[1], WM[2], 'all')

        else:
            WM = xr.load_dataset(
                " ".join(
                    glob.glob(wm_path + 'ERA-5_{date}_{time}*[A-Z].nc'.format(date=day.replace('-', '_'), time=time))))
            hydro_total, wet_total = WM.hydro_total, WM.wet_total
            P, T, e = WM.p, WM.t, WM.e

        # Make interpreter
        hydro_interp, wet_interp = make_interpretor(hydro_total), make_interpretor(wet_total)
        x = xr.DataArray(cur_df['Lon'].ravel(), dims='x')
        y = xr.DataArray(cur_df['Lat'].ravel(), dims='y')
        z = xr.DataArray(hgtlvs, dims='z')

        # Interp and extract data
        P = P.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose()
        T = T.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose()
        e = e.interp(x=x, y=y, z=z).values.transpose().diagonal().transpose()

        WM_ZTD = (hydro_interp(loc) + wet_interp(loc))

        # Get GOES data and set parameters to use either GOES 16/17
        if df['Lon'].values.min() < -126:
            GOES = get_GOES17_image(day.replace('_', '-') + time.replace('_', ':'))
        else:
            GOES = get_GOES16_image(day.replace('_', '-') + time.replace('_', ':'))

        # Apply scaling and extract to array
        GOES = applyScaleAndOffset_all(GOES)
        GOES = sampFeat2array(GOES, cur_df[['Lon', 'Lat']], bands)

        # Get the lat lon of the bbox area
        print('GOES:', GOES[:5, :])

        comb_dat = pd.DataFrame(
            np.hstack((cur_df[['ID', 'Date', 'sigZTD', 'Lat', 'Lon', 'Hgt_m']].values, WM_ZTD.reshape(-1, 1), P,
                       T, e, GOES, cur_df['ZTD'].values.reshape(-1, 1))))

        if i == 0:
            comb_dat.columns = ['ID', 'Date', 'sigZTD', 'Lat', 'Lon', 'Hgt_m', 'wm_ZTD'] + \
                               ['P_' + str(i) for i in range(len(hgtlvs))] + \
                               ['T_' + str(i) for i in range(len(hgtlvs))] + \
                               ['e_' + str(i) for i in range(len(hgtlvs))] + \
                               [i for i in bands] + ['ZTD']
            comb_dat.to_csv(file_name + 'fixed_hgtlvs_GNSS_GOES.csv', index=False)
        else:
            comb_dat.to_csv(file_name + 'fixed_hgtlvs_GNSS_GOES.csv', index=False, mode='a',
                            header=False)
    print('Finished extraction')
    print('File name:', file_name + '_GNSS_interp_interf.csv')


# ifg_path: file path of the interferogram
# wm_path: file path of weather files relatted to the interferogram
# dem_path: file path to the DEM file
# los_path: file path to the Line-of-sight file
# time: time of the intergerogram acquire e.g. 'T23_00_00'
# ref_point: set a reference point for the dereferencing [lon, lat] e.g. [-98, 30]
# file_name: set the file name
# left, bottom, right, top: are the bounding area that would like to focus for extracting data
# bands: Bands that would need from GOES data e.g. ('CMI_C08', 'CMI_C09', 'CMI_C10')
def get_interp_int_params(ifg_path: str, wm_path: str, dem_path: str, los_path: str,
                          time: str, ref_point, file_name: str, left, bottom, right, top, fact=None,
                          sep_date: str = False, bands: str = ('CMI_C08', 'CMI_C09', 'CMI_C10')):
    IFG = glob.glob(ifg_path + '*[0-9]')
    IFG.sort()
    bbox = addGeometry(left, right, bottom, top)

    def block_median(ar, fact):
        assert isinstance(fact, int), type(fact)
        sx, sy = ar.shape
        X, Y = np.ogrid[0:sx, 0:sy]
        regions = sy // fact * (X // fact) + Y // fact
        res = ndimage.median(ar, labels=regions, index=np.arange(regions.max() + 1))
        # res.shape = (sx//fact, sy//fact)
        return res

    for i, ifg in enumerate(IFG):
        print('Working on ifg ', ifg)
        date1, date2 = ifg.split('/')[-1].split('_')
        year = date1[:4]
        ifg_, grid = focus_bound(ifg, left, bottom, right, top)
        ifg_[ifg_ == 0] = np.nan
        ifg_ = convert_rad(ifg_, 0.056 / 100)

        DEM, _ = Resamp_rasterio(dem_path, left, bottom, right, top, ifg_)
        DEM[DEM == -32768] = np.nan
        loc = np.hstack((grid, DEM.ravel().reshape(-1, 1)))

        print('DEM shape', DEM.ravel().reshape(-1, 1).shape)
        print('ifg shape', ifg_.shape)
        print('grid shape', grid.shape)

        LOS, _ = Resamp_rasterio(los_path, left, bottom, right, top, ifg_)
        # Standardized IFG
        row, col = get_rowcol(ifg, left, bottom, right, top, ref_point[0], ref_point[1])
        reg_ifg = (ifg_ - ifg_[row, col]) * np.cos(np.radians(LOS))

        date_1 = datetime.strptime(date1, '%Y%m%d').strftime('%Y_%m_%d')
        date_2 = datetime.strptime(date2, '%Y%m%d').strftime('%Y_%m_%d')

        # Get weather models
        WM_1 = getWM(date_1, time, wm_path)
        WM_2 = getWM(date_2, time, wm_path)

        # Interp values by time
        hydro_total_1 = interpByTime(WM_1[0], WM_1[1], WM_1[2], 'hydro_total')
        wet_total_1 = interpByTime(WM_1[0], WM_1[1], WM_1[2], 'wet_total')
        hydro_total_2 = interpByTime(WM_2[0], WM_2[1], WM_2[2], 'hydro_total')
        wet_total_2 = interpByTime(WM_2[0], WM_2[1], WM_2[2], 'wet_total')

        P_1, T_1, e_1 = interpByTime(WM_1[0], WM_1[1], WM_1[2], 'all')
        P_2, T_2, e_2 = interpByTime(WM_2[0], WM_2[1], WM_2[2], 'all')

        # Make interpretor
        hydro_interp_1 = make_interpretor(hydro_total_1)
        wet_interp_1 = make_interpretor(wet_total_1)
        hydro_interp_2 = make_interpretor(hydro_total_2)
        wet_interp_2 = make_interpretor(wet_total_2)

        P_1_interp = make_interpretor(P_1)
        T_1_interp = make_interpretor(T_1)
        e_1_interp = make_interpretor(e_1)
        P_2_interp = make_interpretor(P_2)
        T_2_interp = make_interpretor(T_2)
        e_2_interp = make_interpretor(e_2)

        P_inter = (P_2_interp(loc) - P_1_interp(loc)).reshape(DEM.shape)
        T_inter = (T_2_interp(loc) - T_1_interp(loc)).reshape(DEM.shape)
        e_inter = (e_2_interp(loc) - e_1_interp(loc)).reshape(DEM.shape)

        P_inter = (P_inter - P_inter[row, col]).ravel().reshape(-1,1)
        T_inter = (T_inter - T_inter[row, col]).ravel().reshape(-1,1)
        e_inter = (e_inter - e_inter[row, col]).ravel().reshape(-1,1)

        WM_ZTD = ((hydro_interp_2(loc) + wet_interp_2(loc)) - (hydro_interp_1(loc) + wet_interp_1(loc))).reshape(
            DEM.shape)
        WM_ZTD = (WM_ZTD - WM_ZTD[row, col])

        # Get GOES data and set parameters to use either GOES 16/17
        if left < -126:
            GOES1 = get_GOES17_image(date_1.replace('_', '-') + time.replace('_', ':'))
            GOES2 = get_GOES17_image(date_2.replace('_', '-') + time.replace('_', ':'))
        else:
            GOES1 = get_GOES16_image(date_1.replace('_', '-') + time.replace('_', ':'))
            GOES2 = get_GOES16_image(date_2.replace('_', '-') + time.replace('_', ':'))

        # Apply scaling and extract to array
        GOES1 = applyScaleAndOffset_all(GOES1)
        GOES2 = applyScaleAndOffset_all(GOES2)
        GOES1 = to_array(GOES1.select(bands), bbox)
        GOES2 = to_array(GOES2.select(bands), bbox)

        # Get the lat lon of the bbox area
        lon = np.linspace(left, right, GOES1.shape[1])
        lat = np.linspace(bottom, top, GOES1.shape[0])
        data = [GOES2[:, :, i] - GOES1[:, :, i] for i in range(len(bands))]

        # Make interpretor
        interpretor = [rgi((lon, lat), data[i].transpose(), bounds_error=False,
                           fill_value=0) for i in range(len(bands))]

        # Interp data to match with DEM/IFG
        interp_data = [interpretor[i](grid).reshape(DEM.shape) for i in range(len(bands))]

        # Dereference data
        deref_GOES = [interp_data[i] - interp_data[i][row, col] for i in range(len(bands))]
        
        if fact:
            if sep_date:
                DEM = block_median(DEM, fact)
                P1 = P_1_interp(loc).reshape(reg_ifg.shape)
                P2 = P_2_interp(loc).reshape(reg_ifg.shape)
                T1 = T_1_interp(loc).reshape(reg_ifg.shape)
                T2 = T_2_interp(loc).reshape(reg_ifg.shape)
                e1 = e_1_interp(loc).reshape(reg_ifg.shape)
                e2 = e_2_interp(loc).reshape(reg_ifg.shape)
                WM_ZTD_1 = (hydro_interp_1(loc) + wet_interp_1(loc)).reshape(reg_ifg.shape)
                WM_ZTD_2 = (hydro_interp_1(loc) + wet_interp_1(loc)).reshape(reg_ifg.shape)
                ds_lon = block_median(grid[:, 0].reshape(reg_ifg.shape), fact).ravel().reshape(-1, 1)
                ds_lat = block_median(grid[:, 1].reshape(reg_ifg.shape), fact).ravel().reshape(-1, 1)
                grid = np.hstack((ds_lon, ds_lat))
                P_1 = block_median(P1 - P1[row, col], fact).ravel().reshape(-1, 1)
                P_2 = block_median(P2 - P2[row, col], fact).ravel().reshape(-1, 1)
                P_inter = np.hstack((P_1, P_2))
                T_1 = block_median(T1 - T1[row, col], fact).ravel().reshape(-1, 1)
                T_2 = block_median(T2 - T2[row, col], fact).ravel().reshape(-1, 1)
                T_inter = np.hstack((T_1, T_2))
                e_1 = block_median(e1 - e1[row, col], fact).ravel().reshape(-1, 1)
                e_2 = block_median(e2 - e2[row, col], fact).ravel().reshape(-1, 1)
                e_inter = np.hstack((e_1, e_2))
                WM_ZTD_1 = block_median(WM_ZTD_1 - WM_ZTD_1[row, col],
                                        fact).ravel().reshape(-1, 1)
                WM_ZTD_2 = block_median(WM_ZTD_2 - WM_ZTD_2[row, col],
                                        fact).ravel().reshape(-1, 1)
                WM_ZTD = np.hstack((WM_ZTD_1, WM_ZTD_2))
                deref_GOES = [block_median(GOES, fact) for GOES in deref_GOES]
                reg_ifg = block_median(reg_ifg, fact)
            else:
                DEM = block_median(DEM, fact)
                ds_lon = block_median(grid[:, 0].reshape(reg_ifg.shape), fact).ravel().reshape(-1, 1)
                ds_lat = block_median(grid[:, 1].reshape(reg_ifg.shape), fact).ravel().reshape(-1, 1)
                grid = np.hstack((ds_lon, ds_lat))
                P_inter = block_median(P_inter.reshape(reg_ifg.shape), fact).ravel().reshape(-1, 1)
                T_inter = block_median(T_inter.reshape(reg_ifg.shape), fact).ravel().reshape(-1, 1)
                e_inter = block_median(e_inter.reshape(reg_ifg.shape), fact).ravel().reshape(-1, 1)
                WM_ZTD = block_median(WM_ZTD, fact)
                deref_GOES = [block_median(GOES, fact) for GOES in deref_GOES]
                reg_ifg = block_median(reg_ifg, fact)

        else:
            pass
        DEM = DEM.ravel().reshape(-1, 1)
        deref_GOES = np.hstack([i.ravel().reshape(-1, 1) for i in deref_GOES])
        WM_ZTD = WM_ZTD.ravel().reshape(-1, 1)
        reg_ifg = reg_ifg.ravel().reshape(-1, 1)
        year = np.array([year] * int(len(DEM.ravel()))).ravel().reshape(-1, 1)
        date__1 = np.array([date_1.replace('_', '-')] * int(len(DEM.ravel()))).ravel().reshape(-1, 1)
        date__2 = np.array([date_2.replace('_', '-')] * int(len(DEM.ravel()))).ravel().reshape(-1, 1)
        df = pd.DataFrame(np.hstack((year, date__1, date__2, DEM, grid,
                                     P_inter, T_inter,
                                     e_inter, WM_ZTD,
                                     deref_GOES,
                                     reg_ifg)))

        if i == 0:
            if sep_date:
                df.columns = ['year', 'date_1', 'date_2', 'DEM', 'lon', 'lat', 'P_1', 'P_2', 'T_1', 'T_2', 'e_1', 'e_2',
                              'wm_ZTD_1', 'wm_ZTD_2'] + [i for i in bands] + ['ifg']
            else:
                df.columns = ['year', 'date_1', 'date_2', 'DEM', 'lon', 'lat', 'P', 'T', 'e', 'wm_ZTD'] + [i for i in
                                                                                                           bands] + [
                                 'ifg']
            df.to_csv(file_name + '_interp_interf.csv', index=False)
        else:
            df.to_csv(file_name + '_interp_interf.csv', index=False, mode='a',
                      header=False)
    print('Finished extraction')
    print('File name:', file_name + '_interp_interf.csv')


# ifg_path: file path of the interferogram
# wm_path: file path of weather files relatted to the interferogram
# dem_path: file path to the DEM file
# los_path: file path to the Line-of-sight file
# time: time of the intergerogram acquire e.g. 'T23_00_00'
# ref_point: set a reference point for the dereferencing [lon, lat] e.g. [-98, 30]
# file_name: set the file name
# left, bottom, right, top: are the bounding area that would like to focus for extracting data
# bands: Bands that would need from GOES data e.g. ('CMI_C08', 'CMI_C09', 'CMI_C10')
def get_PTE_interf_params(ifg_path: str, wm_path: str, dem_path: str, los_path: str,
                          time: str, ref_point, file_name: str, left, bottom, right, top, downsample=False, fact=5,
                          sep_date=False, bands: str = ('CMI_C08', 'CMI_C09', 'CMI_C10')):
    IFG = glob.glob(ifg_path + '*[0-9]')
    IFG.sort()
    bbox = addGeometry(left, right, bottom, top)

    def block_median(ar, fact):
        assert isinstance(fact, int), type(fact)
        sx, sy = ar.shape
        X, Y = np.ogrid[0:sx, 0:sy]
        regions = sy // fact * (X // fact) + Y // fact
        res = ndimage.median(ar, labels=regions, index=np.arange(regions.max() + 1))
        # res.shape = (sx//fact, sy//fact)
        return res

    for i, ifg in enumerate(IFG):
        print('Working on ifg ', ifg)
        date1, date2 = ifg.split('/')[-1].split('_')
        year = date1[:4]
        ifg_, grid = focus_bound(ifg, left, bottom, right, top)
        ifg_[ifg_ == 0] = np.nan
        ifg_ = convert_rad(ifg_, 5.6 / 100)
        DEM, _ = Resamp_rasterio(dem_path, left, bottom, right, top, ifg_)
        DEM[DEM == -32768] = np.nan
        loc = np.hstack((grid, DEM.ravel().reshape(-1, 1)))

        LOS, _ = Resamp_rasterio(los_path, left, bottom, right, top, ifg_)
        # Standardized IFG
        row, col = get_rowcol(ifg, left, bottom, right, top, ref_point[0], ref_point[1])
        reg_ifg = (ifg_ - ifg_[row, col]) * np.cos(np.radians(LOS))

        print('DEM shape', DEM.ravel().reshape(-1, 1).shape)
        print('ifg shape', ifg_.shape)
        print('grid shape', grid.shape)

        date_1 = datetime.strptime(date1, '%Y%m%d').strftime('%Y_%m_%d')
        date_2 = datetime.strptime(date2, '%Y%m%d').strftime('%Y_%m_%d')

        if datetime.strptime(time, 'T%H_%M_%S').minute != 0:
            WM_1 = getWM(date_1, time, wm_path)
            WM_2 = getWM(date_2, time, wm_path)
            # Interp values by time
            hydro_total_1 = interpByTime(WM_1[0], WM_1[1], WM_1[2], 'hydro_total')
            wet_total_1 = interpByTime(WM_1[0], WM_1[1], WM_1[2], 'wet_total')
            hydro_total_2 = interpByTime(WM_2[0], WM_2[1], WM_2[2], 'hydro_total')
            wet_total_2 = interpByTime(WM_2[0], WM_2[1], WM_2[2], 'wet_total')

            P_1, T_1, e_1 = interpByTime(WM_1[0], WM_1[1], WM_1[2], 'all')
            P_2, T_2, e_2 = interpByTime(WM_2[0], WM_2[1], WM_2[2], 'all')
        else:
            WM_1 = xr.load_dataset(
                " ".join(glob.glob(wm_path + 'ERA-5_{date}*[A-Z].nc'.format(date=date_1))))
            WM_2 = xr.load_dataset(
                " ".join(glob.glob(wm_path + 'ERA-5_{date}*[A-Z].nc'.format(date=date_2))))

            hydro_total_1 = WM_1.hydro_total
            wet_total_1 = WM_1.wet_total
            hydro_total_2 = WM_2.hydro_total
            wet_total_2 = WM_2.wet_total

            P_1 = WM_1.p
            T_1 = WM_1.t
            e_1 = WM_1.e
            P_2 = WM_2.p
            T_2 = WM_2.t
            e_2 = WM_2.e

        x = np.array(list(set(grid[:, 0])))
        x.sort()
        y = np.array(list(set(grid[:, 1])))
        y.sort()

        x = xr.DataArray(x, dims='x')
        y = xr.DataArray(y, dims='y')
        z = xr.DataArray(hgtlvs, dims='z')

        # Make interpreter
        hydro_interp_1 = make_interpretor(hydro_total_1)
        wet_interp_1 = make_interpretor(wet_total_1)
        hydro_interp_2 = make_interpretor(hydro_total_2)
        wet_interp_2 = make_interpretor(wet_total_2)

        # Interp and extract data
        P1 = P_1.interp(x=x, y=y, z=z).values
        T1 = T_1.interp(x=x, y=y, z=z).values
        e1 = e_1.interp(x=x, y=y, z=z).values
        P2 = P_2.interp(x=x, y=y, z=z).values
        T2 = T_2.interp(x=x, y=y, z=z).values
        e2 = e_2.interp(x=x, y=y, z=z).values

        # Dereference P, T, e
        P1, T1, e1 = P1 - P1[:, col, row].reshape(-1, 1, 1), \
                     T1 - T1[:, col, row].reshape(-1, 1, 1), \
                     e1 - e1[:, col, row].reshape(-1, 1, 1)
        P2, T2, e2 = P2 - P2[:, col, row].reshape(-1, 1, 1), \
                     T2 - T2[:, col, row].reshape(-1, 1, 1), \
                     e2 - e2[:, col, row].reshape(-1, 1, 1)
        WM_ZTD_1 = (hydro_interp_1(loc) + wet_interp_1(loc)).reshape(DEM.shape)
        WM_ZTD_2 = (hydro_interp_2(loc) + wet_interp_2(loc)).reshape(DEM.shape)
        WM_ZTD_1 = WM_ZTD_1 - WM_ZTD_1[row, col]
        WM_ZTD_2 = WM_ZTD_2 - WM_ZTD_2[row, col]

        if sep_date:
            P_inter, T_inter, e_inter = np.hstack((P1.reshape((-1, len(hgtlvs)), order='F'),
                                                   P2.reshape((-1, len(hgtlvs)), order='F'))), \
                                        np.hstack((T1.reshape((-1, len(hgtlvs)), order='F'),
                                                   T2.reshape((-1, len(hgtlvs)), order='F'))), \
                                        np.hstack((e1.reshape((-1, len(hgtlvs)), order='F'),
                                                   e2.reshape((-1, len(hgtlvs)), order='F')))
            WM_ZTD = np.hstack((WM_ZTD_1.ravel().reshape(-1, 1), WM_ZTD_2.ravel().reshape(-1, 1)))
        else:
            P_inter, T_inter, e_inter = (P2 - P1), (T2 - T1), (e2 - e1)
            P_inter, T_inter, e_inter = P_inter.reshape((-1, len(hgtlvs)), order='F'), \
                                        T_inter.reshape((-1, len(hgtlvs)), order='F'), \
                                        e_inter.reshape((-1, len(hgtlvs)), order='F')

            WM_ZTD = ((hydro_interp_2(loc) + wet_interp_2(loc)) -
                      (hydro_interp_1(loc) + wet_interp_1(loc))).reshape(DEM.shape)
            WM_ZTD = WM_ZTD - WM_ZTD[row, col]

        # Get GOES data and set parameters to use either GOES 16/17
        if left < -126:
            GOES1 = get_GOES17_image(date_1.replace('_', '-') + time.replace('_', ':'))
            GOES2 = get_GOES17_image(date_2.replace('_', '-') + time.replace('_', ':'))
        else:
            GOES1 = get_GOES16_image(date_1.replace('_', '-') + time.replace('_', ':'))
            GOES2 = get_GOES16_image(date_2.replace('_', '-') + time.replace('_', ':'))

        # Apply scaling and extract to array
        GOES1 = applyScaleAndOffset_all(GOES1)
        GOES2 = applyScaleAndOffset_all(GOES2)
        GOES1 = to_array(GOES1.select(bands), bbox)
        GOES2 = to_array(GOES2.select(bands), bbox)

        # Get the lat lon of the bbox area
        lon = np.linspace(left, right, GOES1.shape[1])
        lat = np.linspace(bottom, top, GOES1.shape[0])
        if sep_date:
            # Make interpreter
            interpretor1 = [rgi((lon, lat), GOES1[:, :, i].transpose(), bounds_error=False,
                                fill_value=0) for i in range(len(bands))]
            interpretor2 = [rgi((lon, lat), GOES2[:, :, i].transpose(), bounds_error=False,
                                fill_value=0) for i in range(len(bands))]
            # Interp data to match with DEM/IFG
            interp_data1 = [interpretor1[i](grid).reshape(DEM.shape) for i in range(len(bands))]
            interp_data2 = [interpretor2[i](grid).reshape(DEM.shape) for i in range(len(bands))]
            # Dereference data
            deref_GOES1 = [interp_data1[i] - interp_data1[i][row, col] for i in range(len(bands))]
            deref_GOES2 = [interp_data2[i] - interp_data2[i][row, col] for i in range(len(bands))]
            deref_GOES = np.hstack((deref_GOES1, deref_GOES2))
        else:
            data = [GOES2[:, :, i] - GOES1[:, :, i] for i in range(len(bands))]
            # Make interpretor
            interpretor = [rgi((lon, lat), data[i].transpose(), bounds_error=False,
                               fill_value=0) for i in range(len(bands))]
            # Interp data to match with DEM/IFG
            interp_data = [interpretor[i](grid).reshape(DEM.shape) for i in range(len(bands))]
            # Dereference data
            deref_GOES = [interp_data[i] - interp_data[i][row, col] for i in range(len(bands))]

        if downsample:
            if sep_date:
                DEM = block_median(DEM, fact).ravel().reshape(-1, 1)
                ds_lon = block_median(grid[:, 0].reshape(reg_ifg.shape), fact).ravel().reshape(-1, 1)
                ds_lat = block_median(grid[:, 1].reshape(reg_ifg.shape), fact).ravel().reshape(-1, 1)
                grid = np.hstack((ds_lon, ds_lat))
                P_1 = np.stack([block_median(P1[i], fact) for i in range(len(hgtlvs))]).T
                P_2 = np.stack([block_median(P2[i], fact) for i in range(len(hgtlvs))]).T
                P_inter = np.hstack((P_1, P_2))
                T_1 = np.stack([block_median(T1[i], fact) for i in range(len(hgtlvs))]).T
                T_2 = np.stack([block_median(T2[i], fact) for i in range(len(hgtlvs))]).T
                T_inter = np.hstack((T_1, T_2))
                e_1 = np.stack([block_median(e1[i], fact) for i in range(len(hgtlvs))]).T
                e_2 = np.stack([block_median(e1[i], fact) for i in range(len(hgtlvs))]).T
                e_inter = np.hstack((e_1, e_2))
                WM_ZTD_1 = block_median(WM_ZTD_1, fact).ravel().reshape(-1, 1)
                WM_ZTD_2 = block_median(WM_ZTD_2, fact).ravel().reshape(-1, 1)
                WM_ZTD = np.hstack((WM_ZTD_1, WM_ZTD_2))
                deref_GOES1 = [block_median(GOES, fact) for GOES in deref_GOES1]
                deref_GOES2 = [block_median(GOES, fact) for GOES in deref_GOES2]
                deref_GOES1 = np.hstack([i.ravel().reshape(-1, 1) for i in deref_GOES1])
                deref_GOES2 = np.hstack([i.ravel().reshape(-1, 1) for i in deref_GOES2])
                deref_GOES = np.hstack((deref_GOES1, deref_GOES2))
                reg_ifg = block_median(reg_ifg, fact).ravel().reshape(-1, 1)
            else:
                DEM = block_median(DEM, fact).ravel().reshape(-1, 1)
                ds_lon = block_median(grid[:, 0].reshape(reg_ifg.shape), fact).ravel().reshape(-1, 1)
                ds_lat = block_median(grid[:, 1].reshape(reg_ifg.shape), fact).ravel().reshape(-1, 1)
                grid = np.hstack((ds_lon, ds_lat))
                P_inter = block_median(P_inter.reshape(reg_ifg.shape), fact).ravel().reshape(-1, 1)
                T_inter = block_median(T_inter.reshape(reg_ifg.shape), fact).ravel().reshape(-1, 1)
                e_inter = block_median(e_inter.reshape(reg_ifg.shape), fact).ravel().reshape(-1, 1)
                WM_ZTD = block_median(WM_ZTD, fact).ravel().reshape(-1, 1)
                deref_GOES = [block_median(GOES, fact) for GOES in deref_GOES]
                deref_GOES = np.hstack([i.ravel().reshape(-1, 1) for i in deref_GOES])
                reg_ifg = block_median(reg_ifg, fact).ravel().reshape(-1, 1)
        else:
            pass
        year = np.array([year] * int(len(DEM.ravel()))).ravel().reshape(-1, 1)
        date__1 = np.array([date_1.replace('_', '-')] * int(len(DEM.ravel()))).ravel().reshape(-1, 1)
        date__2 = np.array([date_2.replace('_', '-')] * int(len(DEM.ravel()))).ravel().reshape(-1, 1)
        df = pd.DataFrame(np.hstack((year, date__1, date__2, DEM, grid,
                                     P_inter, T_inter, e_inter, WM_ZTD, deref_GOES,
                                     reg_ifg)))

        if i == 0:
            if sep_date:
                df.columns = ['year', 'date_1', 'date_2', 'DEM', 'lon', 'lat'] + \
                             ['P_1_' + str(i) for i in range(len(hgtlvs))] + \
                             ['T_1_' + str(i) for i in range(len(hgtlvs))] + \
                             ['e_1_' + str(i) for i in range(len(hgtlvs))] + \
                             ['P_2_' + str(i) for i in range(len(hgtlvs))] + \
                             ['T_2_' + str(i) for i in range(len(hgtlvs))] + \
                             ['e_2_' + str(i) for i in range(len(hgtlvs))] + \
                             ['wm_ZTD_1', 'wm_ZTD_2'] + \
                             [band + str('_1') for band in bands] + [band + str('_2') for band in bands] + ['ifg']
            else:
                df.columns = ['year', 'date_1', 'date_2', 'DEM', 'lon', 'lat', 'P', 'T', 'e', 'wm_ZTD'] + \
                             [i for i in bands] + ['ifg']
            df.to_csv(file_name + '_PTE_interp_interf.csv', index=False)
        else:
            df.to_csv(file_name + '_PTE_interp_interf.csv', index=False, mode='a',
                      header=False)
    print('Finished extraction')
    print('File name:', file_name + '_PTE_interp_interf.csv')


# GNSS_file_path: file path of the GNSS ZTD file
# wm_path: file path of weather files related to GNSS stations
# time: time of the GNSS station acquire e.g. 'T23_00_00'
# file_name: set the file name
# bands: Bands that would need from GOES data e.g. ['CMI_C08', 'CMI_C09', 'CMI_C10']
def get_GNSS_interp_int_params(GNSS_file_path: str, wm_path: str,
                               file_name: str, time: str, date_diff=12,
                               bands=None):
    if bands is None:
        bands = ['CMI_C08', 'CMI_C09', 'CMI_C10']
    df = pd.read_csv(GNSS_file_path)
    df = df[(df['sigZTD'] < 0.01) & (df['Date'] > '2017-07-12')]
    Date = np.sort(list(set(df['Date'])))
    last_date = Date[-1]
    for i, day in enumerate(Date):
        start, end = datetime_offset(day, time_for='%Y-%m-%d', margin=date_diff)
        print('Working on date ', start + '_' + end)
        if end > last_date:
            break
        else:
            df_start = df.loc[df['Date'] == start]
            df_end = df.loc[df['Date'] == end]
            if len(df_start) != len(df_end):
                df_end = df_end[df_end.ID.isin(df_start.ID)]
                df_start = df_start[df_start.ID.isin(df_end.ID)]
            else:
                pass
        loc = df_start[['Lon', 'Lat', 'Hgt_m']].values
        int_delay = (df_start['ZTD'].values - df_end['ZTD'].values)
        # Get weather models
        if datetime.strptime(time, 'T%H_%M_%S').minute != 0:
            WM_1 = getWM(start.replace('-', '_'), time, wm_path)
            WM_2 = getWM(end.replace('-', '_'), time, wm_path)
            # Interp values by time
            hydro_total_1 = interpByTime(WM_1[0], WM_1[1], WM_1[2], 'hydro_total')
            wet_total_1 = interpByTime(WM_1[0], WM_1[1], WM_1[2], 'wet_total')
            hydro_total_2 = interpByTime(WM_2[0], WM_2[1], WM_2[2], 'hydro_total')
            wet_total_2 = interpByTime(WM_2[0], WM_2[1], WM_2[2], 'wet_total')

            P_1, T_1, e_1 = interpByTime(WM_1[0], WM_1[1], WM_1[2], 'all')
            P_2, T_2, e_2 = interpByTime(WM_2[0], WM_2[1], WM_2[2], 'all')
        else:
            WM_1 = xr.load_dataset(
                " ".join(glob.glob(wm_path + 'ERA-5_{date}*[A-Z].nc'.format(date=start.replace('-', '_')))))
            WM_2 = xr.load_dataset(
                " ".join(glob.glob(wm_path + 'ERA-5_{date}*[A-Z].nc'.format(date=end.replace('-', '_')))))

            hydro_total_1 = WM_1.hydro_total
            wet_total_1 = WM_1.wet_total
            hydro_total_2 = WM_2.hydro_total
            wet_total_2 = WM_2.wet_total

            P_1 = WM_1.p
            T_1 = WM_1.t
            e_1 = WM_1.e
            P_2 = WM_2.p
            T_2 = WM_2.t
            e_2 = WM_2.e

        # Make interpreter
        hydro_interp_1 = make_interpretor(hydro_total_1)
        wet_interp_1 = make_interpretor(wet_total_1)
        hydro_interp_2 = make_interpretor(hydro_total_2)
        wet_interp_2 = make_interpretor(wet_total_2)

        P_1_interp = make_interpretor(P_1)
        T_1_interp = make_interpretor(T_1)
        e_1_interp = make_interpretor(e_1)
        P_2_interp = make_interpretor(P_2)
        T_2_interp = make_interpretor(T_2)
        e_2_interp = make_interpretor(e_2)

        P_inter = (P_1_interp(loc) - P_2_interp(loc))
        T_inter = (T_1_interp(loc) - T_2_interp(loc))
        e_inter = (e_1_interp(loc) - e_2_interp(loc))

        WM_ZTD = ((hydro_interp_1(loc) + wet_interp_1(loc)) - (hydro_interp_2(loc) + wet_interp_2(loc)))

        # Get GOES data and set parameters to use either GOES 16/17
        if df['Lon'].values.min() < -126:
            GOES1 = get_GOES17_image(start.replace('_', '-') + time.replace('_', ':'))
            GOES2 = get_GOES17_image(end.replace('_', '-') + time.replace('_', ':'))
        else:
            GOES1 = get_GOES16_image(start.replace('_', '-') + time.replace('_', ':'))
            GOES2 = get_GOES16_image(end.replace('_', '-') + time.replace('_', ':'))

        # Apply scaling and extract to array
        GOES1 = applyScaleAndOffset_all(GOES1)
        GOES2 = applyScaleAndOffset_all(GOES2)
        GOES1_ = sampFeat2array(GOES1, df_start[['Lon', 'Lat']], bands)
        GOES2_ = sampFeat2array(GOES2, df_start[['Lon', 'Lat']], bands)

        # Get the lat lon of the bbox area
        print('GOES1:', GOES1_[:5, :])
        print('GOES2:', GOES2_[:5, :])

        data = (pd.DataFrame(GOES1_) - pd.DataFrame(GOES2_)).values
        comb_dat = pd.DataFrame(np.hstack((df_start['ID'].values.reshape(-1, 1), loc, P_inter.reshape(-1, 1),
                                           T_inter.reshape(-1, 1), e_inter.reshape(-1, 1), WM_ZTD.reshape(-1, 1),
                                           data, int_delay.reshape(-1, 1))))

        if i == 0:
            comb_dat.columns = ['ID', 'Lon', 'Lat', 'Hgt_m', 'P', 'T', 'e', 'wm_ZTD'] + [i for i in bands] + ['inf_ZD']
            comb_dat.to_csv(file_name + '_GNSS_interp_interf.csv', index=False)
        else:
            comb_dat.to_csv(file_name + '_GNSS_interp_interf.csv', index=False, mode='a',
                            header=False)
    print('Finished extraction')
    print('File name:', file_name + '_GNSS_interp_interf.csv')
