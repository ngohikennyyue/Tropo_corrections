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
from tools.Extract_ee_function import *
import os
service_account = 'goes-extract@extract-goes-1655507865824.iam.gserviceaccount.com'
KEY = '../tools/private_key.json'
credentials = ee.ServiceAccountCredentials(service_account, KEY)

Num_bands = 32
print('Initialize Google Earth Engine...')
ee.Initialize(credentials)
file = '../test/GNSS_station_PTE.csv'
# print(pd.read_csv(file))
extract_param(file, '11:00:00', ['CMI_C08', 'CMI_C09', 'CMI_C10', 'CMI_C12'])