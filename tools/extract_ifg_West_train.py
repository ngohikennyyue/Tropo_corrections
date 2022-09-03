import sys
import os

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
parent = os.path.dirname(parent)
sys.path.append(parent)

from extract_func.Extract_ifg_parameters_function import *

print('Extract West (CA)')
extract_ifg_param(ifg_path='Extracted/unwrappedPhase/', wm_file_path='weather_files/',
                  dem_path='Extracted/DEM/SRTM_3arcsec_uncropped.tif', los_path='Angle/los.rdr',
                  slope_path='slope.tif', coh_path='Extracted/coherence/', time='T01_50_00', years=[2019, 2020],
                  lon_min=-118.5, lat_min=32.2, lon_max=-117.5, lat_max=41.2, inter=False, samp=500, name='train',
                  ref_point=[])

print('Finished extraction')
