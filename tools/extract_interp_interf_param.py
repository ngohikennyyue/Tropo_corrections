import sys
import os

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
parent = os.path.dirname(parent)
sys.path.append(parent)

from extract_func.Extract_PTE_function import *
from extract_func.Extract_ee_function import *

service_account = 'goes-extract@extract-goes-1655507865824.iam.gserviceaccount.com'
KEY = '../../../private_key.json'
credentials = ee.ServiceAccountCredentials(service_account, KEY)

Num_bands = 33
print('Initialize Google Earth Engine...')
ee.Initialize(credentials)
print('Reading in file...')
ifg_path = 'Extracted/unwrappedPhase/'
wm_path = 'weather_files/'
dem_path = 'Extracted/DEM/SRTM_3arcsec_uncropped.tif'
los_path = 'Angle/los.rdr'
time = 'T23_13_00'
ref_point = [-78, 40]
file_name = 'East'
left, bottom, right, top = -79, 39, -77, 41
bands = ('CMI_C08', 'CMI_C09', 'CMI_C10')
get_interp_int_params(ifg_path, wm_path, dem_path, los_path,
                      time, ref_point, file_name, left, bottom, right, top,
                      bands)
