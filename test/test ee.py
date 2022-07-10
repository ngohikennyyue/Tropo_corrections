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
import matplotlib.pyplot as plt
import pandas as pd
import requests
from tools.Extract_ee_function import *
from tools.Extract_PTE_function import *
import os

service_account = 'goes-extract@extract-goes-1655507865824.iam.gserviceaccount.com'
KEY = '../tools/private_key.json'
credentials = ee.ServiceAccountCredentials(service_account, KEY)

Num_bands = 32
print('Initialize Google Earth Engine...')
ee.Initialize(credentials)
date_pairs = ['20190714_20190702', '20190714_20190708', '20190720_20190714', ' 20190807_20190801']
geom = addGeometry(-155.9, -154.9, 18.9, 19.9)
ifg, grid = focus_bound('../20190714_20190702', -155.9, 18.9, -154.9, 19.9)

dem, dem_grid = Resamp_rasterio('../SRTM_3arcsec_uncropped.tif', -155.9, 18.9, -154.9, 19.9, ifg)
dem[dem <= 0.0] = np.nan
mask = dem.copy()
mask[mask > 0] = 1
ifg = ifg * mask
bands = ['CMI_C08', 'CMI_C09', 'CMI_C10', 'CMI_C12']
dimension = (ifg.shape[-1], ifg.shape[0])
GOES_17 = get_GOES17_image('2019-07-14T16:30:00', geom)
img = applyScaleAndOffset_all(GOES_17)
for i in bands:
    data_url = img.getDownloadURL({'name': 'test_img',
                                   'crs': 'EPSG:4326',
                                   'bands': i,
                                   'dimensions': dimension,
                                   'format': 'GEO_TIFF',
                                   'region': geom})
    # print('Downloading...', imgName)
    data_to_download = 'GOES_17'
    day = 14
    month = '07'
    year = 2019
    r = requests.get(data_url, allow_redirects=True)
    local_fileName = os.path.join(os.getcwd(),
                                  data_to_download + '_' + i + '_' + str(year) + str(month) + str(day) + '.tif')
    open(local_fileName, 'wb').write(r.content)
# CMI_C08, _ = Resamp_rasterio('', -155.8, 18.9, -154.9, 20.3, ifg)
# img = to_array(img, geom)
# img = sampFeat2array(img, dem_grid.tolist(), ['CMI_C08', 'CMI_C09', 'CMI_C10', 'CMI_C12'])
print(ifg.shape, dem.shape)
# plt.imshow(convert_rad((ifg - np.nanmean(ifg)), 5.6 / 100),
#            extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()], cmap='RdYlBu')
# plt.colorbar(label='m')
# plt.show()
#
# plt.imshow(CMI_C08, cmap='RdYlBu')
# plt.colorbar(label='K')
# plt.show()
#
# plt.imshow(dem, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()])
# plt.colorbar(label='m')
# plt.show()
