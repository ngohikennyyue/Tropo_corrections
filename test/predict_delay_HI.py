import pandas as pd
import sys
import os
import tensorflow as tf

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *
from joblib import load
from datetime import datetime
import requests

# service_account = 'goes-extract@extract-goes-1655507865824.iam.gserviceaccount.com'
# KEY = 'private_key.json'
# credentials = ee.ServiceAccountCredentials(service_account, KEY)
# print('Initialize Google Earth Engine...')
# ee.Initialize(credentials)
# print('Done')

date_pairs = ['20190714_20190702', '20190714_20190708', '20190720_20190714', '20190807_20190801',
              '20190819_20190813', '20190831_20190825', '20190906_20190831', '20190918_20190906']

lon_min, lat_min, lon_max, lat_max = -155.9, 18.9, -154.9, 19.9

hgtlvs = [-100, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400,
          2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000,
          5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000,
          14000, 15000, 16000, 17000, 18000, 19000, 20000, 25000, 30000, 35000, 40000]
# Load Model
# GOES_model = tf.keras.models.load_model('../../ML/Model/Full_US_WE_PTE_fixed_hgtlvs_cloud_model')
Norm_model = tf.keras.models.load_model('../ML/Model/Full_US_PTE_fixed_hgtlvs_model')
inter_model = tf.keras.models.load_model('../ML/Inter_model/Model/inter_PTE_fixed_hgtlvs_model')
# Load scaler
# scaler_x_g = load('../../ML/Scaler/US_WE_MinMax_scaler_x.bin')
# scaler_y_g = load('../../ML/Scaler/US_WE_MinMax_scaler_y.bin')
scaler_x = load('../ML/Scaler/US_WE_noGOES_MinMax_scaler_x.bin')
scaler_y = load('../ML/Scaler/US_WE_noGOES_MinMax_scaler_y.bin')
inter_scaler_x = load('../ML/Inter_model/Scaler/interferometric_MinMax_scaler_x.bin')
inter_scaler_y = load('../ML/Inter_model/Scaler/interferometric_MinMax_scaler_y.bin')

for i, date in enumerate(date_pairs):

    ifg, grid = focus_bound('../InSAR/Hawaii/products/Extracted/unwrappedPhase/' + date, lon_min, lat_min, lon_max, lat_max)
    dem, dem_grid = Resamp_rasterio('../InSAR/Hawaii/products/Extracted/DEM/SRTM_3arcsec_uncropped.tif', lon_min, lat_min, lon_max,
                                    lat_max, ifg)
    los, _ = Resamp_rasterio('../InSAR/Hawaii/Angle/los.rdr', lon_min, lat_min, lon_max, lat_max, ifg)
    # print('DEM: ', dem_grid[0:5])
    # Create a mask for IFG
    mask = dem.copy()
    mask[mask > 0] = 1
    mask[mask <= 0] = np.nan
    ifg = ifg * mask
    # Get date and wm
    date1, date2 = get_datetime(date)
    wm1 = xr.load_dataset(" ".join(glob.glob('../InSAR/Hawaii/weather_model/weather_files/' + 'ERA-5_{date}_T04_00_00*[A-Z].nc'.format(date=date1))))
    wm2 = xr.load_dataset(" ".join(glob.glob('../InSAR/Hawaii/weather_model/weather_files/' + 'ERA-5_{date}_T04_00_00*[A-Z].nc'.format(date=date2))))
    # Plot ifg
    plt.style.use('seaborn')
    plt.imshow(convert_rad(ifg - np.nanmean(ifg), 5.6 / 100),
               extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
               cmap='RdYlBu')
    plt.colorbar(label='meters')
    plt.title('Raw Interferogram')
    plt.savefig('Plots/Hawaii/Norm/Ifg_' + date + '.png', dpi=300)
    plt.clf()

    if i == 0:
        # Plot DEM (only the first one)
        dem = dem.astype(float)
        dem[dem <= 0.0] = np.nan
        plt.imshow(dem, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()])
        plt.colorbar()
        plt.savefig('Plots/Hawaii/Norm/DEM.png', dpi=300)
    else:
        pass
    dem = dem.astype(float)
    dem[dem <= 0.0] = np.nan
    x = list(set(dem_grid[:, 0]))
    x.sort()
    y = list(set(dem_grid[:, 1]))
    y.sort()

    # Day1 WM (PTe) parameters
    P1 = wm1.p.interp(x=x, y=y, z=hgtlvs).values
    T1 = wm1.t.interp(x=x, y=y, z=hgtlvs).values
    e1 = wm1.e.interp(x=x, y=y, z=hgtlvs).values

    # Day2 WM (PTe) parameters
    P2 = wm2.p.interp(x=x, y=y, z=hgtlvs).values
    T2 = wm2.t.interp(x=x, y=y, z=hgtlvs).values
    e2 = wm2.e.interp(x=x, y=y, z=hgtlvs).values

    Day_1 = np.hstack((dem_grid[:, 1].reshape(-1, 1), dem.ravel().reshape(-1, 1),
                       np.vstack(P1.transpose().reshape((P1.shape[-1] * P1.shape[1], 51))),
                       np.vstack(T1.transpose().reshape((T1.shape[-1] * T1.shape[1], 51))),
                       np.vstack(e1.transpose().reshape((e1.shape[-1] * e1.shape[1], 51)))))
    Day_2 = np.hstack((dem_grid[:, 1].reshape(-1, 1), dem.ravel().reshape(-1, 1),
                       np.vstack(P2.transpose().reshape((P2.shape[-1] * P2.shape[1], 51))),
                       np.vstack(T2.transpose().reshape((T2.shape[-1] * T2.shape[1], 51))),
                       np.vstack(e2.transpose().reshape((e2.shape[-1] * e2.shape[1], 51)))))
    inf_wm = np.hstack((dem_grid[:, 1].reshape(-1, 1), dem.ravel().reshape(-1, 1),
                        np.vstack((P1-P2).transpose().reshape((P2.shape[-1] * P2.shape[1], 51))),
                        np.vstack((T1-T2).transpose().reshape((T2.shape[-1] * T2.shape[1], 51))),
                        np.vstack((e1-e2).transpose().reshape((e2.shape[-1] * e2.shape[1], 51)))))

    # Norm_model_prediction
    pred_1 = scaler_y.inverse_transform(Norm_model.predict(scaler_x.transform(Day_1)))
    pred_2 = scaler_y.inverse_transform(Norm_model.predict(scaler_x.transform(Day_2)))
    pred_inf = inter_scaler_y.inverse_transform(inter_model.predict(inter_scaler_x.transform(inf_wm)))

    # Plot TD prediction of each date
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    im1 = ax1.imshow(pred_1.reshape(dem.shape), cmap='RdYlBu')
    plt.colorbar(im1, ax=ax1, fraction=0.05, pad=0.2, label='Total Delay (m)')
    ax1.set_title('Total Delay of Date {}'.format(date1))
    im2 = ax2.imshow(pred_2.reshape(dem.shape), cmap='RdYlBu')
    plt.colorbar(im2, ax=ax2, fraction=0.05, pad=0.2, label='Total Delay (m)')
    ax2.set_title('Total Delay of Date {}'.format(date2))
    fig.savefig('Plots/Hawaii/Norm/Norm_model_TD_predict_{}.png'.format(date), dpi=300)
    plt.clf()

    TD = (pred_2 - pred_1).reshape(dem.shape)  # The difference of two dates will become the total delay
    plt.style.use('seaborn')
    plt.imshow(TD, cmap='RdYlBu')
    plt.colorbar(label='Prediction Total Delay(m)')
    plt.title('Total Delay ({})'.format(date))
    plt.savefig('Plots/Hawaii/Norm/Total_Delay_{}.png'.format(date), dpi=300)
    plt.clf()

    inf_td = pred_inf.reshape(dem.shape)
    plt.style.use('seaborn')
    plt.imshow(inf_td, cmap='RdYlBu')
    plt.colorbar(label='Prediction Total Delay(m)')
    plt.title('Interferic model Total Delay ({})'.format(date))
    plt.savefig('Plots/Hawaii/Norm/Int_Total_Delay_{}.png'.format(date), dpi=300)
    plt.clf()

    # Plot Ifg along side with TD (predicted)
    ifg_img = convert_rad(ifg - np.nanmean(ifg), 5.6 / 100)
    plt.style.use('seaborn')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, figsize=(15, 15))
    im1 = ax1.imshow(TD - np.nanmean(TD), cmap='RdYlBu',)
    plt.colorbar(im1, ax=ax1, fraction=0.05, pad=0.1, label='(m)')
    ax1.set_title('Total Delay of Date {}'.format(date))
    im2 = ax2.imshow(ifg_img,
                     extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()], cmap='RdYlBu')
    plt.colorbar(im2, ax=ax2, fraction=0.05, pad=0.1, label='(m)')
    ax2.set_title('Raw Interferogram {}'.format(date))
    im3 = ax3.imshow(ifg_img - (TD - np.nanmean(TD)), extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                     cmap='RdYlBu')
    plt.colorbar(im3, ax=ax3, fraction=0.05, pad=0.1, label='(m)')
    ax3.set_title('Tropo Corrected Ifg {}'.format(date))
    im4 = ax4.imshow(inf_td,extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                     cmap='RdYlBu' )
    plt.colorbar(im4, ax=ax4, fraction=0.05, pad=0.1, label='(m)')
    ax4.set_title('Inter model total delay - {}'.format(date))
    fig.savefig('Plots/Hawaii/Norm/Norm_model_Ifg_PredTD_{}.png'.format(date), dpi=300)
    plt.clf()

    # GOES_data
    # bands = ['CMI_C08', 'CMI_C09', 'CMI_C10', 'CMI_C12']
    # dimension = (ifg.shape[-1], ifg.shape[0])
    # geom = addGeometry(lon_min, lon_max, lat_min, lat_max)
    # goes_1 = get_GOES17_image(date1.replace('_', '-') + 'T16:30:00')
    # goes_2 = get_GOES17_image(date2.replace('_', '-') + 'T16:30:00')
    # img_1 = applyScaleAndOffset_all(goes_1).select('CMI_C08', 'CMI_C09', 'CMI_C10', 'CMI_C12')
    # img_2 = applyScaleAndOffset_all(goes_2).select('CMI_C08', 'CMI_C09', 'CMI_C10', 'CMI_C12')
    # # goes_1_array = sampFeat2array(img_1, dem_grid.tolist(), bands)
    # # goes_2_array = sampFeat2array(img_2, dem_grid.tolist(), bands)
    # imgs = [img_1, img_2]
    # dates = [date1, date2]
    # data_to_download = 'GOES17'
    #
    # for band in bands:
    #     for n, img in enumerate(imgs):
    #         data_url = img.getDownloadURL({'name': 'img',
    #                                        'crs': 'EPSG:4326',
    #                                        'bands': band,
    #                                        'dimensions': dimension,
    #                                        'format': 'GEO_TIFF',
    #                                        'region': geom})
    #         r = requests.get(data_url, allow_redirects=True)
    #         local_fileName = os.path.join(os.getcwd() + '/GOES_data/',
    #                                       data_to_download + '_' + str(band) + '_' + str(dates[n]) + '.tif')
    #         open(local_fileName, 'wb').write(r.content)
    # GOES_date_1 = glob.glob('GOES_data/GOES17_*_{}'.format(date1))
    # GOES_date_2 = glob.glob('GOES_data/')
    # print('Length of array: ', len(goes_1_array), len(goes_2_array))

    # Day_1 = np.hstack((Day_1, goes_1_array))
    # Day_2 = np.hstack((Day_2, goes_2_array))
    # print('Added GOES data shape: ', Day_1.shape, Day_2.shape)
    #
    # # GOES_model_prediction
    # pred_1 = scaler_y_g.inverse_transform(Norm_model.predict(scaler_x_g.transform(Day_1)))
    # pred_2 = scaler_y_g.inverse_transform(Norm_model.predict(scaler_x_g.transform(Day_2)))
    #
    # # Plot TD prediction of each date
    # plt.style.use('seaborn')
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    # im1 = ax1.imshow(pred_1.reshape(dem.shape), cmap='RdYlBu')
    # plt.colorbar(im1, ax=ax1, fraction=0.05, pad=0.2, label='Total Delay (m)')
    # ax1.set_title('Total Delay of Date {}'.format(date1))
    # im2 = ax2.imshow(pred_2.reshape(dem.shape), cmap='RdYlBu')
    # plt.colorbar(im2, ax=ax2, fraction=0.05, pad=0.2, label='Total Delay (m)')
    # ax2.set_title('Total Delay of Date {}'.format(date2))
    # fig.savefig('Plots/GOES_model_TD_predict_{}.png'.format(date), dpi=300)
    # plt.clf()
    #
    # TD = (pred_2 - pred_1).reshape(dem.shape)  # The difference of two dates will become the total delay
    # plt.style.use('seaborn')
    # plt.imshow(TD, cmap='RdYlBu')
    # plt.colorbar(label='Prediction  and GPS delay (m)')
    # plt.title('Total Delay ({})'.format(date))
    # plt.savefig('Plots/GOES_Total_Delay_{}.png'.format(date), dpi=300)
    # plt.clf()
