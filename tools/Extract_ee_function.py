import ee
import os
import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt


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


Num_bands = 32


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
    return (geom)


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
def extract_param(file_path: str, time: str, bands: list):
    loc = os.path.dirname(os.path.abspath(file_path))
    name = file_path.split('/')[-1].split('.')[0]
    ext = file_path.split('/')[-1].split('.')[-1]
    print(loc, name, ext)
    if ext == 'csv':
        df = pd.read_csv(file_path)
        df = df[df['Date'] > '2017-12-31']  # GOES data only valid after 20170710
        df = df[(df['Lat'] > 26) & (df['Lat'] < 48) & (df['Lon'] > -124) & (df['Lon'] < -44)]
    elif ext == 'ftr':
        df = pd.read_feather(file_path)
        df = df[df['Date'] > '2017-07-10']  # GOES data only valid after 20170710
        df = df[(df['Lat'] > 26) & (df['Lat'] < 48) & (df['Lon'] > -124) & (df['Lon'] < -44)]
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
        # get GOES 16 image
        img = get_GOES16_image(i + 'T' + time)
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
