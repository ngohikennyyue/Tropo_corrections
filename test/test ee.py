# import ee
# from tools.Extract_ee_function import *
# service_account = 'goes-extract@extract-goes-1655507865824.iam.gserviceaccount.com'
# KEY = 'tools/private_key.json'
# credentials = ee.ServiceAccountCredentials(service_account, KEY)
# ee.Initialize(credentials)
#
# # create AOI
# area = addGeometry(-89, -88, 36, 40)
#
# test_goes = get_GOES16_image('2017-10-01T11:00:00', area)
#
# # Scale and offset the image
# Img = applyScaleAndOffset_all(test_goes)
#
# print(test_goes.getInfo())
import ee
import pandas as pd

from tools.Extract_ee_function import *
import os

service_account = 'goes-extract@extract-goes-1655507865824.iam.gserviceaccount.com'
KEY = '../tools/private_key.json'
credentials = ee.ServiceAccountCredentials(service_account, KEY)

Num_bands = 32
print('Initialize Google Earth Engine...')
ee.Initialize(credentials)
file = '../test/UNRcombinedGPS_ztd.csv'
GPS = pd.read_csv(file)
GPS = GPS[GPS['Date'] == '2018-01-15']
GPS = GPS[GPS['sigZTD'] < 0.01]
station = GPS[['Lon', 'Lat']]
station.to_csv('Lat_lon.csv', index=False)
print('Size of GPS station: ', len(station))
# print(pd.read_csv(file))
# extract_param(file, '11:00:00', ['CMI_C08', 'CMI_C09', 'CMI_C10', 'CMI_C12'])
# station = [[-120,30],[-78,24],[-70,40]]
bands = ['CMI_C08', 'CMI_C09', 'CMI_C10', 'CMI_C12']
GOES = get_GOES16_image('2018-01-15T11:00:00').select('CMI_C08', 'CMI_C09', 'CMI_C10', 'CMI_C12')
projection = GOES.projection()
Img = applyScaleAndOffset_selected(GOES)
# point = list2features(station.values.tolist())
multi_point = ee.Geometry.MultiPoint(station.values.tolist(), projection)
IMG = Img.reduceRegion(multi_point, ee.Reducer.first())

print(IMG.getInfo())
# convert feature to array
# ext = sampFeat2array(Img, station, bands)
