import numpy as np
import pandas as pd
import sys
import os
import tensorflow as tf

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *
from joblib import load

# service_account = 'goes-extract@extract-goes-1655507865824.iam.gserviceaccount.com'
# KEY = 'private_key.json'
# credentials = ee.ServiceAccountCredentials(service_account, KEY)
# print('Initialize Google Earth Engine...')
# ee.Initialize(credentials)
# print('Done')

# date_pairs = ['20200623_20200611', '20200729_20200717', '20200810_20200729', '20200212_20200131',
#               '20200219_20200207', '20200903_20200822', '20200226_20200214', '20200229_20200217']
date_pairs = glob.glob('../InSAR/Pennsylvania/products/Extracted/unwrappedPhase/*[0-9]')
date_pairs.sort()
lon_min, lat_min, lon_max, lat_max = -78, 40, -77, 41
print('Bounding area: ', lon_min, lat_min, lon_max, lat_max)

hgtlvs = [-100, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400,
          2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000,
          5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000,
          14000, 15000, 16000, 17000, 18000, 19000, 20000, 25000, 30000, 35000, 40000]
# Load Model
# GOES_model = tf.keras.models.load_model('../../ML/Model/Full_US_WE_PTE_fixed_hgtlvs_cloud_model')
Norm_model = tf.keras.models.load_model('../ML/Model/Full_US_PTE_fixed_hgtlvs_model')
IFG_model = tf.keras.models.load_model('../ML/IFG_model/Model/ifg_PTE_slope_model')
IFG_model_new = tf.keras.models.load_model('../ML/IFG_model/Model/ifg_PTE_slope_model_new')
IFG_model_15000 = tf.keras.models.load_model('../ML/IFG_model/Model/ifg_PTE_model_15000')
IFG_model_BN = tf.keras.models.load_model('../ML/IFG_model/Model/ifg_PTE_slope_BN_model')

# Load scaler
# scaler_x_g = load('../../ML/Scaler/US_WE_MinMax_scaler_x.bin')
# scaler_y_g = load('../../ML/Scaler/US_WE_MinMax_scaler_y.bin')
scaler_x = load('../ML/Scaler/US_WE_noGOES_MinMax_scaler_x.bin')
scaler_y = load('../ML/Scaler/US_WE_noGOES_MinMax_scaler_y.bin')
scaler_ifg_x = load('../ML/IFG_model/Scaler/ifg_PTE_slope_MinMax_scaler_x.bin')
scaler_ifg_y = load('../ML/IFG_model/Scaler/ifg_PTE_slope_MinMax_scaler_y.bin')
scaler_ifg_x_new = load('../ML/IFG_model/Scaler/ifg_PTE_slope_MinMax_scaler_x_new.bin')
scaler_ifg_x_15000 = load('../ML/IFG_model/Scaler/ifg_PTE_MinMax_scaler_x_15000.bin')
scaler_ifg_y_15000 = load('../ML/IFG_model/Scaler/ifg_PTE_slope_MinMax_scaler_y_15000.bin')
scaler_ifg_y_BN = load('../ML/IFG_model/Scaler/ifg_PTE_slope_BN_MinMax_scaler_y.bin')

for i, intfg in enumerate(date_pairs[::4]):
    ifg, grid = focus_bound(intfg, lon_min, lat_min, lon_max, lat_max)
    dem, dem_grid = Resamp_rasterio('../InSAR/Pennsylvania/products/Extracted/DEM/SRTM_3arcsec_uncropped.tif', lon_min,
                                    lat_min, lon_max, lat_max, ifg)
    slope, _ = Resamp_rasterio('../InSAR/Pennsylvania/slope.tif', lon_min, lat_min, lon_max, lat_max, ifg)
    los, _ = Resamp_rasterio('../InSAR/Pennsylvania/Angle/los.rdr', lon_min, lat_min, lon_max, lat_max, ifg)
    date = intfg.split('/')[-1]
    print('LOS: ', los[:10])
    print('DEM lat_lon: ', dem_grid[0:5])
    # Create a mask for IFG
    mask = dem.copy()
    mask[mask != -32768] = 1
    mask[mask == -32768] = np.nan
    ifg = ifg * mask

    # Plot ifg
    plt.style.use('seaborn')
    plt.imshow(convert_rad(ifg - np.nanmean(ifg), 5.6 / 100),
               extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
               cmap='RdYlBu')
    plt.colorbar(label='meters')
    plt.title('Raw Interferogram')
    plt.savefig('Plots/Pennsylvania/Ifg_' + date + '.png', dpi=300)
    plt.clf()

    dem = dem.astype(float)
    dem[dem == -32768] = np.nan
    if i == 0:
        # Plot DEM (only the first one)
        plt.imshow(dem, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()])
        plt.colorbar()
        plt.savefig('Plots/Pennsylvania/DEM.png', dpi=300)
        plt.clf()

        plt.imshow(los, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()])
        plt.colorbar()
        plt.savefig('Plots/Pennsylvania/LOS.png', dpi=300)
        plt.clf()
    else:
        pass
    # Get Dates and WM
    date1, date2 = get_datetime(date)
    wm1 = xr.load_dataset(
        " ".join(glob.glob('../InSAR/Pennsylvania/weather_files/' + 'ERA-5_{date}*[A-Z].nc'.format(date=date1))))
    wm2 = xr.load_dataset(
        " ".join(glob.glob('../InSAR/Pennsylvania/weather_files/' + 'ERA-5_{date}*[A-Z].nc'.format(date=date2))))

    # Get all the x,y coordinate
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
    data = np.hstack((dem_grid[:, 1].reshape(-1, 1), dem.ravel().reshape(-1, 1),
                      np.vstack(P1.transpose().reshape((P1.shape[-1] * P1.shape[1], 51))),
                      np.vstack(T1.transpose().reshape((T1.shape[-1] * T1.shape[1], 51))),
                      np.vstack(e1.transpose().reshape((e1.shape[-1] * e1.shape[1], 51))),
                      np.vstack(P2.transpose().reshape((P2.shape[-1] * P2.shape[1], 51))),
                      np.vstack(T2.transpose().reshape((T2.shape[-1] * T2.shape[1], 51))),
                      np.vstack(e2.transpose().reshape((e2.shape[-1] * e2.shape[1], 51))),
                      slope.ravel().reshape(-1, 1)
                      ))
    # Norm_model_prediction
    pred_1 = scaler_y.inverse_transform(Norm_model.predict(scaler_x.transform(Day_1)))
    pred_2 = scaler_y.inverse_transform(Norm_model.predict(scaler_x.transform(Day_2)))
    pred = scaler_ifg_y.inverse_transform(IFG_model.predict(scaler_ifg_x.transform(data)))
    pred_new = IFG_model_new.predict(scaler_ifg_x_new.transform(data))
    pred_15000 = scaler_ifg_y_15000.inverse_transform(
        IFG_model_15000.predict(scaler_ifg_x_15000.transform(data[:, :-1])))
    pred_BN = scaler_ifg_y_BN.inverse_transform(IFG_model_BN.predict(data))

    # Plot TD prediction of each date
    # plt.style.use('seaborn')
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    # im1 = ax1.imshow(pred_1.reshape(dem.shape), cmap='RdYlBu')
    # plt.colorbar(im1, ax=ax1, fraction=0.05, pad=0.2, label='Total Delay (m)')
    # ax1.set_title('Total Delay of Date {}'.format(date1))
    # im2 = ax2.imshow(pred_2.reshape(dem.shape), cmap='RdYlBu')
    # plt.colorbar(im2, ax=ax2, fraction=0.05, pad=0.2, label='Total Delay (m)')
    # ax2.set_title('Total Delay of Date {}'.format(date2))
    # fig.savefig('Plots/Pennsylvania/Norm_model_TD_predict_{}.png'.format(date), dpi=300)
    # plt.clf()

    TD = (pred_2 - pred_1).reshape(dem.shape)  # The difference of two dates will become the total delay
    # plt.style.use('seaborn')
    # plt.imshow(TD, cmap='RdYlBu')
    # plt.colorbar(label='Prediction Total Delay(m)')
    # plt.title('Total Delay ({})'.format(date))
    # plt.savefig('Plots/Pennsylvania/Total_Delay_{}.png'.format(date), dpi=300)
    # plt.clf()

    IFG_pred = (pred - np.nanmean(pred)).reshape(dem.shape)
    # plt.imshow(IFG_pred, cmap='RdYlBu')
    # plt.colorbar(label='Prediction Total Delay(m)')
    # plt.title('IFG model prediction TD ({})'.format(date))
    # plt.savefig('Plots/Pennsylvania/IFG_Total_Delay_{}.png'.format(date), dpi=300)
    # plt.clf()

    IFG_pred_new = (pred_new - np.nanmean(pred_new)).reshape(dem.shape)
    # plt.imshow(IFG_pred, cmap='RdYlBu')
    # plt.colorbar(label='Prediction Total Delay(m)')
    # plt.title('New IFG model prediction TD ({})'.format(date))
    # plt.savefig('Plots/Pennsylvania/New_IFG_Total_Delay_{}.png'.format(date), dpi=300)
    # plt.clf()

    IFG_pred_15000 = (pred_15000 - np.nanmean(pred_15000)).reshape(dem.shape)
    # plt.imshow(IFG_pred, cmap='RdYlBu')
    # plt.colorbar(label='Prediction Total Delay(m)')
    # plt.title('IFG model (15000) prediction TD ({})'.format(date))
    # plt.savefig('Plots/Pennsylvania/IFG_15000_Total_Delay_{}.png'.format(date), dpi=300)
    # plt.clf()

    IFG_pred_BN = (pred_BN - np.nanmean(pred_BN)).reshape(dem.shape)
    plt.imshow(IFG_pred, cmap='RdYlBu')
    plt.colorbar(label='Prediction Total Delay(m)')
    plt.title('IFG model BN prediction TD ({})'.format(date))
    plt.savefig('Plots/Pennsylvania/IFG_BN_Total_Delay_{}.png'.format(date), dpi=300)
    plt.clf()

    # Plot Ifg along side with TD (predicted)
    ifg_img = convert_rad(ifg - np.nanmean(ifg), 5.6 / 100)
    plt.style.use('seaborn')
    fig, ax = plt.subplots(3, 2, figsize=(15, 15))
    im1 = ax[0, 0].imshow(TD / np.cos(np.radians(los)),
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu', vmin=np.nanmin(ifg_img), vmax=np.nanmax(ifg_img))
    ax[0, 0].set_title('Normal Model TD of Date {}'.format(date))
    im2 = ax[0, 1].imshow(ifg_img,
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu')
    ax[0, 1].set_title('Raw Interferogram {}'.format(date))
    im3 = ax[1, 0].imshow(IFG_pred / np.cos(np.radians(los)),
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu', vmin=np.nanmin(ifg_img), vmax=np.nanmax(ifg_img))

    ax[1, 0].set_title('IFG model TD of Date {}'.format(date))
    im4 = ax[1, 1].imshow(IFG_pred_new / np.cos(np.radians(los)),
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu', vmin=np.nanmin(ifg_img), vmax=np.nanmax(ifg_img))
    ax[1, 1].set_title('New IFG model TD of Date {}'.format(date))
    im5 = ax[2, 0].imshow(IFG_pred_15000 / np.cos(np.radians(los)),
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu', vmin=np.nanmin(ifg_img), vmax=np.nanmax(ifg_img))
    ax[2, 0].set_title('IFG model (15000) TD of Date {}'.format(date))
    im6 = ax[2, 1].imshow(IFG_pred_BN / np.cos(np.radians(los)),
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu', vmin=np.nanmin(ifg_img), vmax=np.nanmax(ifg_img))
    ax[2, 1].set_title('IFG model BN TD of Date {}'.format(date))
    fig.colorbar(im2)
    fig.suptitle('Normal Model, IFG Model, Raw IFG in slanted direction')
    fig.savefig('Plots/Pennsylvania/Norm_model_Ifg_PredTD_{}.png'.format(date), dpi=300)
    plt.clf()
