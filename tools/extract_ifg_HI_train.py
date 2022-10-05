import sys
import os

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
parent = os.path.dirname(parent)
sys.path.append(parent)

from extract_func.Extract_ifg_parameters_function import *

print('Extract Hawaii')
print('Bounding area: lon_min=-155.9, lat_min=18.9, lon_max=-154.9, lat_max=19.9')

extract_ifg_param(ifg_path='Extracted/unwrappedPhase/', wm_file_path='weather_files/',
                  dem_path='Extracted/DEM/SRTM_3arcsec_uncropped.tif', los_path='Angle/los.rdr',
                  slope_path='slope.tif', coh_path='Extracted/coherence/', time='T04_30_00', years=[2019, 2020],
                  lon_min=-155.9, lat_min=18.9, lon_max=-154.9, lat_max=19.9, samp=100, name='train_ref',
                  ref_point=[-155.4, 19.6])

print('Finished extraction')

# Define the area
area = ee.Geometry.Polygon(
    [[[105.532, 19.059], [105.606, 19.058], [105.605, 19.108], [105.530, 19.110], [105.532, 19.059]]])

# define the image
img = ee.Image('COPERNICUS/S2/20160209T034234_20160209T090731_T48QWG')

# do any ee operation here
ndvi = ee.Image(img.normalizedDifference(['B8', 'B4']))
timedate = img.get('GENERATION_TIME').getInfo()

# get the lat lon and add the ndvi
latlon = ee.Image.pixelLonLat().addBands(ndvi)

# apply reducer to list
latlon = latlon.reduceRegion(
    reducer=ee.Reducer.toList(),
    geometry=area,
    maxPixels=1e8,
    scale=20);

# get data into three different arrays
data = np.array((ee.Array(latlon.get('nd')).getInfo()))
lats = np.array((ee.Array(latlon.get('latitude')).getInfo()))
lons = np.array((ee.Array(latlon.get('longitude')).getInfo()))