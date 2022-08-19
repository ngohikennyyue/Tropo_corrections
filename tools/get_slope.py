import rasterio
import richdem as rd
import numpy as np
import matplotlib.pyplot as plt

def get_slope(dem_path: str):
    # Read in dem.tif for calculating slope
    dataset = rasterio.open(dem_path)
    data = dataset.read(1)
    data[data==dataset.nodata] = np.nan
    _dem = rd.rdarray(data, no_data=dataset.nodata)
    slope = rd.TerrainAttribute(_dem, attrib='slope_riserun')

    print(slope[:10])
    profile = dataset.profile
    profile['nodata'] = np.nan
    profile['dtype'] = slope.dtype

    # Save derived slope
    with rasterio.open('slope.tif', 'w', **profile) as dst:
        dst.write(slope, 1)

    # Plot slope
    plt.imshow(slope, cmap='Reds')
    plt.colorbar()
    plt.savefig('slope.png')

    print('Finished slope')

get_slope('Extracted/DEM/SRTM_3arcsec_uncropped.tif')