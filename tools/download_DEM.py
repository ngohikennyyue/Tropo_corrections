from dem_stitcher.stitcher import stitch_dem
import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
import richdem as rd
import numpy as np

# xmin, ymin, xmax, ymax
bounds = [-123, 25, -66, 50]

dst_area_or_point = 'Point'
dst_ellipsoidal_height = True
dem_name = 'srtm_v3'

X, p = stitch_dem(bounds,
                  dem_name=dem_name,
                  dst_ellipsoidal_height=dst_ellipsoidal_height,
                  dst_area_or_point=dst_area_or_point)

fig, ax = plt.subplots(figsize=(10, 10))
ax = plot.show(X, transform=p['transform'], ax=ax)
ax.set_xlabel('Longitude', size=15)
ax.set_ylabel('Latitude', size=15)
fig.savefig('US_dem_{}.png'.format(dem_name), dpi=200)

height_type = 'ellipsoidal' if dst_ellipsoidal_height else 'geoid'
with rasterio.open(f'{dem_name}_{height_type}_{dst_area_or_point}.tif', 'w', **p) as ds:
    ds.write(X, 1)
    ds.update_tags(AREA_OR_POINT=dst_area_or_point)

print('Finished DEM Download')

# Read in dem.tif for calculating slope
dataset = rasterio.open(f'{dem_name}_{height_type}_{dst_area_or_point}.tif')
data = dataset.read()
data = np.squeeze(data)
_dem = rd.rdarray(data, no_data=dataset.nodata)
slope = rd.TerrainAttribute(_dem, attrib='slope_riserun')

print(slope[:10])
profile = dataset.profile
profile['nodata'] = np.nan
profile['dtype'] = slope.dtype

# Save derived slope
with rasterio.open('slope.tif', 'w', **profile) as dst:
    dst.write(slope, 1)

print('Finished slope')
