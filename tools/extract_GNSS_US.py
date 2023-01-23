import sys
import os

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
sys.path.append(parent)

from extract_func.Extract_PTE_function import *
from extract_func.Extract_ee_function import *

service_account = 'goes-extract@extract-goes-1655507865824.iam.gserviceaccount.com'
KEY = '../../private_key.json'
credentials = ee.ServiceAccountCredentials(service_account, KEY)

Num_bands = 33
print('Initialize Google Earth Engine...')
ee.Initialize(credentials)
GNSS_file_path = 'GPS_stations/product2/UNRcombinedGPS_ztd.csv'
wm_path = 'weather_model/weather_files/'
file_name = 'US_GNSS_'
time = 'T11_00_00'

get_GNSS_interp_int_params(GNSS_file_path, wm_path,
                           file_name, time, date_diff=12,
                           bands=None)

