import sys
import os

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
parent = os.path.dirname(parent)
sys.path.append(parent)

from extract_func.Extract_ifg_parameters_function import *

print('Extract East')
print('Bounding area: lon_min=-78, lat_min=34.2, lon_max=-77, lat_max=45')

extract_ifg_param(ifg_path='Extracted/unwrappedPhase/', wm_file_path='weather_files/',
                  dem_path='Extracted/DEM/SRTM_3arcsec_uncropped.tif', los_path='Angle/los.rdr',
                  slope_path='slope.tif', coh_path='Extracted/coherence/', time='T23_05_00', years=[2021],
                  lon_min=-78, lat_min=34.2, lon_max=-77, lat_max=45.2, inter=False, samp=250, name='test')

print('Finished extraction')
