import ee
import os
import pandas as pd
import numpy as np
import folium
import glob
import matplotlib.pyplot as plt
from extract_func.Extract_PTE_function import *

# Trigger the authentication flow.
# ee.Authenticate()
# ee.Initialize()

# Function to extract the band and apply scaling and offset
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
    import ee
    geom = ee.Geometry.Polygon(
        [[[min_lon, max_lat],
          [min_lon, min_lat],
          [max_lon, min_lat],
          [max_lon, max_lat]]])
    return geom


# add time margin for an input datetime
def datetime_margin(date_time, time_for='%Y-%m-%dT%H:%M:%S', margin=5):
    from datetime import datetime
    from datetime import timedelta
    time = date_time
    given_time = datetime.strptime(time, time_for)
    start_time = (given_time - timedelta(minutes=margin)).strftime(time_for)
    end_time = (given_time + timedelta(minutes=margin)).strftime(time_for)
    return start_time, end_time


# datetime: '2019-01-01T11:05:00'
# geometry: aoi can be created with addGeometry function
def get_GOES16_image(datetime, geometry=None):
    import ee
    date1, date2 = datetime_margin(datetime)
    GOES = ee.ImageCollection('NOAA/GOES/16/MCMIPC')
    if geometry is None:
        GOES_img = GOES.filterDate(date1, date2).first()
    else:
        GOES_img = GOES.filterDate(date1, date2).filterBounds(geometry).first().clip(geometry)
    return GOES_img


def get_GOES17_image(datetime, geometry=None):
    import ee
    date1, date2 = datetime_margin(datetime)
    GOES = ee.ImageCollection('NOAA/GOES/17/MCMIPC')
    if geometry is None:
        GOES_img = GOES.filterDate(date1, date2).first()
    else:
        GOES_img = GOES.filterDate(date1, date2).filterBounds(geometry).first().clip(geometry)
    return GOES_img


# convert list of coordinates of GPS stations into feature collection
# station: list of GPS station in the order of [Lon, Lat]
def list2features(station):
    valuesList = ee.List(station)

    def list2feature(el):
        el = ee.List(el)
        geom = ee.Geometry.Point([ee.Number(el.get(0)), ee.Number(el.get(1))])
        return ee.Feature(geom)

    multi_point = ee.FeatureCollection(ee.List(valuesList.map(list2feature)))
    return multi_point


# Extract values from multispectral image by the point of interest and bands
# img: Image of interest
# points_list: list of [lon, lat]
# bands: list of band names that of interest.
def sampFeat2array(img, points_list, bands):
    multi_point = list2features(points_list)
    ft = img.reduceRegions(multi_point, ee.Reducer.first(), 10)
    for kk, band in enumerate(bands):
        try:
            dat = ft.toList(len(points_list)).map(lambda feature: ee.Feature(feature).get(band)).getInfo()
            print(len(dat), len(points_list))
            if len(dat) == len(points_list):
                if kk == 0:
                    dat1 = ft.toList(len(points_list)).map(lambda feature: ee.Feature(feature).get(band)).getInfo()
                    data_full = np.zeros((len(dat1), len(bands)))
                    data_full[:, kk] = dat1
                else:
                    dat = ft.toList(len(points_list)).map(lambda feature: ee.Feature(feature).get(band)).getInfo()
                    data_full[:, kk] = dat
            else:
                data_full = np.zeros((len(points_list), len(bands)))
                return data_full
                continue
        except ee.ee_exception.EEException:
            data_full = np.zeros((len(points_list), len(bands)))
            return data_full
            continue
    # dat_full = pd.DataFrame(dat_full)
    # dat_full.columns = [bands]
    return data_full


# function to extract bands value with existed database
# df: dataframe of the extracted P,T,e values with GPS station coordinate
# time: str object of time of interested eg. '11:00:00'
# geometry: AOI can be created with addGeometry function
# bands: bands that are of interested in a list []
def extract_param(file_path: str, time: str, bands: list, GOES16=True):
    loc = os.path.dirname(os.path.abspath(file_path))
    name = file_path.split('/')[-1].split('.')[0]
    ext = file_path.split('/')[-1].split('.')[-1]
    print(loc, name, ext)
    if ext == 'csv':
        df = pd.read_csv(file_path)
        df = df[df['Date'] > '2017-07-10']  # GOES data only valid after 20170710
        # df = df[(df['Lat'] > 26) & (df['Lat'] < 48) & (df['Lon'] > -124) & (df['Lon'] < -44)]
    elif ext == 'ftr':
        df = pd.read_feather(file_path)
        df = df[df['Date'] > '2017-07-10']  # GOES data only valid after 20170710
        # df = df[(df['Lat'] > 26) & (df['Lat'] < 48) & (df['Lon'] > -124) & (df['Lon'] < -44)]
    else:
        print('Can not read this file type', ext)
        exit()
    # dataFrame = []
    Date = np.sort(list(set(df['Date'])))
    for num, i in enumerate(Date):
        print('Extracting...', i)
        dd = df.loc[df['Date'] == i].copy()
        print(len(dd))
        station = dd[['Lon', 'Lat']].values
        if GOES16:
            # get GOES 16 image
            img = get_GOES16_image(i + 'T' + time)
        else:
            # get GOES 17 image
            img = get_GOES17_image(i + 'T' + time)

        if isinstance(img.getInfo(), (float, int, str, list, dict, tuple)):  # skip any empty dataset
            # apply scale and offset image
            Img = applyScaleAndOffset_all(img)
            # convert feature to array
            ext = sampFeat2array(Img, station.tolist(), bands)
            if len(ext) != len(station):
                print(len(ext), len(station))
                print('Mismatching data... Skipped')
                print('')
                continue
            else:
                print('Adding bands to the Dataframe...')
                for b, band in enumerate(bands):
                    print(b, band)
                    dd.loc[:, band] = ext[:, b]
                if num == 0:
                    print('write first csv')
                    dd.to_csv(loc + '/' + name + '_cloud.csv', index=False)
                else:
                    print('append to existed csv')
                    dd.to_csv(loc + '/' + name + '_cloud.csv', index=False, header=False, mode='a')
                print('Done')
                print('')
        else:
            print('Empty data... Skipped')
            print('')
            continue
    print('Finished extraction')


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
                          time: str, ref_point, file_name: str, left, bottom, right, top,
                          bands: str = ('CMI_C08', 'CMI_C09', 'CMI_C10')):
    IFG = glob.glob(ifg_path + '*[0-9]')
    IFG.sort()
    bbox = addGeometry(left, right, bottom, top)
    for i, ifg in enumerate(IFG):
        print('Working on ifg ', ifg)
        date1, date2 = ifg.split('/')[-1].split('_')
        ifg_, grid = focus_bound(ifg, left, bottom, right, top)
        ifg_[ifg_ == 0] = np.nan
        ifg_ = convert_rad(ifg_, 5.6 / 100)

        DEM, _ = Resamp_rasterio(dem_path, left, bottom, right, top, ifg_)
        DEM[DEM == -32768] = np.nan
        loc = np.hstack((grid, DEM.ravel().reshape(-1, 1)))

        print('DEM shape', DEM.ravel().reshape(-1, 1).shape)
        print('ifg shape', ifg_.shape)
        print('grid shape', grid.shape)

        LOS, _ = Resamp_rasterio(los_path, left, bottom, right, top, ifg_)
        # Standardized IFG
        row, col = get_rowcol(ifg, left, bottom, right, top, ref_point[0], ref_point[1])
        reg_ifg = (ifg_ - ifg_[row, col])

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

        P_inter = (P_inter - P_inter[row, col])
        T_inter = (T_inter - T_inter[row, col])
        e_inter = (e_inter - e_inter[row, col])

        WM_ZTD = ((hydro_interp_2(loc) + wet_interp_2(loc)) - (hydro_interp_1(loc) + wet_interp_1(loc))).reshape(
            DEM.shape) / np.cos(np.radians(LOS))
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
        data = [GOES1[:, :, i] - GOES2[:, :, i] for i in range(len(bands))]

        # Make interpretor
        interpretor = [rgi((lon, lat), data[i].transpose(), bounds_error=False,
                           fill_value=0) for i in range(len(bands))]

        # Interp data to match with DEM/IFG
        interp_data = [interpretor[i](grid).reshape(DEM.shape) for i in range(len(bands))]

        # Dereference data
        deref_data = [interp_data[i] - interp_data[i][row, col] for i in range(len(bands))]

        df = pd.DataFrame(np.hstack((DEM.ravel().reshape(-1, 1), grid[:, 1].reshape(-1, 1),
                                     P_inter.reshape(-1, 1), T_inter.reshape(-1, 1),
                                     e_inter.reshape(-1, 1), WM_ZTD.ravel().reshape(-1, 1),
                                     np.hstack([i.ravel().reshape(-1, 1) for i in deref_data]),
                                     reg_ifg.ravel().reshape(-1, 1))))

        if i == 0:
            df.columns = ['DEM', 'lat', 'P', 'T', 'e', 'wm_ZTD'] + [i for i in bands] + ['ifg']
            df.to_csv(file_name + '_interp_interf.csv', index=False)
        else:
            df.to_csv(file_name + '_interp_interf.csv', index=False, mode='a',
                      header=False)
    print('Finished extraction')
    print('File name:', file_name + '_interp_interf.csv')