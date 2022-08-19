import sys
import os

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
sys.path.append(parent)

from extract_func.Extract_ifg_parameters_function import *

extract_ifg_param(ifg_path='products/Extracted/unwrappedPhase/', wm_file_path='weather_model/weather_files/',
                  dem_path='products/Extracted/DEM/SRTM_3arcsec_uncropped.tif', los_path='Angle/los.rdr',
                  slope_path='slope.tif', coh_path='products/Extracted/coherence/', time='T04_30_00', lon_min=-155.9, lat_min=18.9, lon_max=-154.9, lat_max=19.9,
                  inter=False)

print('Finished extraction')
