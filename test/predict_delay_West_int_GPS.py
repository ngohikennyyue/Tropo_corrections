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

# date_pairs = ['20200623_20200611', '20200729_20200717', '20200810_20200729', '20200212_20200131',
#               '20200219_20200207', '20200903_20200822', '20200226_20200214', '20200229_20200217']

hgtlvs = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400,
          2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000,
          5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000,
          14000, 15000, 16000, 17000, 18000, 19000, 20000, 25000, 30000, 35000, 40000]
date_pairs = ['20190411_20190330', '20190716_20190710']
# date_pairs = glob.glob('../InSAR/Pennsylvania/products/Extracted/unwrappedPhase/*[0-9]')
# date_pairs.sort()
lon_min, lat_min, lon_max, lat_max = -117.5, 35, -116.5, 36
print('Bounding area: ', lon_min, lat_min, lon_max, lat_max)

# Load Model
NI_GPS_int_model = tf.keras.models.load_model('../ML/GNSS_Int_model/Model/GPS_int_PTE_slope_newinput_model')
GPS_int_model = tf.keras.models.load_model('../ML/GNSS_Int_model/Model/GPS_int_PTE_slope_model')
GPS_nolatlon_model = tf.keras.models.load_model('../ML/GNSS_Int_model/Model/GPS_int_PTE_slope_noLatLon_model')
# Load scaler
NI_scaler_x = load('../ML/GNSS_Int_model/Scaler/GPS_int_PTE_slope_model_newinput_MinMax_scaler_x.bin')
NI_scaler_y = load('../ML/GNSS_Int_model/Scaler/GPS_int_PTE_slope_model_newinput_MinMax_scaler_y.bin')
scaler_x = load('../ML/GNSS_Int_model/Scaler/GPS_int_PTE_slope_model_MinMax_scaler_x.bin')
scaler_y = load('../ML/GNSS_Int_model/Scaler/GPS_int_PTE_slope_model_MinMax_scaler_y.bin')
noLatLon_scaler_x = load('../ML/GNSS_Int_model/Scaler/GPS_int_PTE_slope_model_noLatLon_MinMax_scaler_x.bin')
noLatLon_scaler_y = load('../ML/GNSS_Int_model/Scaler/GPS_int_PTE_slope_model_noLatLon_MinMax_scaler_y.bin')

for i, intfg in enumerate(date_pairs):
    ifg, grid = focus_bound('../InSAR/Large_scale/West/Extracted/unwrappedPhase/{}'.format(intfg), lon_min, lat_min,
                            lon_max, lat_max)
    dem, dem_grid = Resamp_rasterio('../InSAR/Large_scale/West/Extracted/DEM/SRTM_3arcsec_uncropped.tif', lon_min,
                                    lat_min, lon_max, lat_max, ifg)
    slope, _ = Resamp_rasterio('../InSAR/Large_scale/West/slope.tif', lon_min, lat_min, lon_max, lat_max, ifg)
    los, _ = Resamp_rasterio('../InSAR/Large_scale/West/Angle/los.rdr', lon_min, lat_min, lon_max, lat_max, ifg)
    # date = intfg.split('/')[-1]
    date = intfg
    # print('LOS: ', los[:10])
    # print('DEM lat_lon: ', dem_grid[0:5])
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
    plt.savefig('Plots/West/Ifg_' + date + '.png', dpi=300)
    plt.clf()

    dem = dem.astype(float)
    dem[dem == -32768] = np.nan
    if i == 0:
        # Plot DEM (only the first one)
        plt.imshow(dem, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()])
        plt.colorbar()
        plt.savefig('Plots/West/DEM.png', dpi=300)
        plt.clf()

        plt.imshow(los, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()])
        plt.colorbar()
        plt.savefig('Plots/West/LOS.png', dpi=300)
        plt.clf()
    else:
        pass
    # Get Dates and WM
    date1, date2 = get_datetime(date)
    # WM1, wm1, minute = getWM(date1, 'T01_50_00', '../InSAR/Large_scale/West/weather_files/' )
    # WM2, wm2, _ = getWM(date1, 'T01_50_00', '../InSAR/Large_scale/West/weather_files/' )
    wm1 = xr.load_dataset(
        " ".join(glob.glob(
            '../InSAR/Large_scale/West/weather_files/' + 'ERA-5_{date}_T02_00_00*[A-Z].nc'.format(date=date1))))
    wm2 = xr.load_dataset(
        " ".join(glob.glob(
            '../InSAR/Large_scale/West/weather_files/' + 'ERA-5_{date}_T02_00_00*[A-Z].nc'.format(date=date2))))

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

    data1 = np.hstack((dem_grid, dem.ravel().reshape(-1, 1),
                       np.vstack((P1 / T1).transpose().reshape((P1.shape[-1] * P1.shape[1], len(hgtlvs)))),
                       np.vstack(e1.transpose().reshape((e1.shape[-1] * e1.shape[1], len(hgtlvs)))),
                       np.vstack((P2 / T2).transpose().reshape((P2.shape[-1] * P2.shape[1], len(hgtlvs)))),
                       np.vstack(e2.transpose().reshape((e2.shape[-1] * e2.shape[1], len(hgtlvs)))),
                       slope.ravel().reshape(-1, 1)
                       ))
    data2 = np.hstack((dem_grid, dem.ravel().reshape(-1, 1),
                       np.vstack(P1.transpose().reshape((P1.shape[-1] * P1.shape[1], len(hgtlvs)))),
                       np.vstack(T1.transpose().reshape((T1.shape[-1] * T1.shape[1], len(hgtlvs)))),
                       np.vstack(e1.transpose().reshape((e1.shape[-1] * e1.shape[1], len(hgtlvs)))),
                       np.vstack(P2.transpose().reshape((P2.shape[-1] * P2.shape[1], len(hgtlvs)))),
                       np.vstack(T2.transpose().reshape((T2.shape[-1] * T2.shape[1], len(hgtlvs)))),
                       np.vstack(e2.transpose().reshape((e2.shape[-1] * e2.shape[1], len(hgtlvs)))),
                       slope.ravel().reshape(-1, 1)
                       ))
    data3 = np.hstack((dem.ravel().reshape(-1, 1),
                       np.vstack(P1.transpose().reshape((P1.shape[-1] * P1.shape[1], len(hgtlvs)))),
                       np.vstack(T1.transpose().reshape((T1.shape[-1] * T1.shape[1], len(hgtlvs)))),
                       np.vstack(e1.transpose().reshape((e1.shape[-1] * e1.shape[1], len(hgtlvs)))),
                       np.vstack(P2.transpose().reshape((P2.shape[-1] * P2.shape[1], len(hgtlvs)))),
                       np.vstack(T2.transpose().reshape((T2.shape[-1] * T2.shape[1], len(hgtlvs)))),
                       np.vstack(e2.transpose().reshape((e2.shape[-1] * e2.shape[1], len(hgtlvs)))),
                       slope.ravel().reshape(-1, 1)
                       ))
    print(data1.shape, data2.shape, data3.shape)
    # Norm_model_prediction
    pred_NI = NI_scaler_y.inverse_transform(NI_GPS_int_model.predict(NI_scaler_x.transform(data1)))
    pred_norm = scaler_y.inverse_transform(GPS_int_model.predict(scaler_x.transform(data2)))
    pred_noLatLon = noLatLon_scaler_y.inverse_transform(GPS_nolatlon_model.predict(noLatLon_scaler_x.transform(data3)))

    ifg_img = convert_rad(ifg - np.nanmean(ifg), 5.6 / 100)
    pred_NI_ifg = (pred_NI - np.nanmean(pred_NI)).reshape(dem.shape) / np.cos(np.radians(los))
    pred_norm_ifg = (pred_norm - np.nanmean(pred_norm)).reshape(dem.shape) / np.cos(np.radians(los))
    pred_noLatLon_ifg = (pred_noLatLon - np.nanmean(pred_noLatLon)).reshape(dem.shape) / np.cos(np.radians(los))

    print(date)
    print('Ifg std: ', np.nanstd(ifg_img))
    print('Pred_NI std:  ', np.nanstd(pred_NI))
    print('Pred_NI mean: ', np.nanmean(pred_NI))
    print('Pred_NI_ifg std: ', np.nanstd(pred_NI_ifg))
    print('Pred_NI_ifg mean: ', np.nanmean(pred_NI_ifg))
    print('NI_model corrected std: ', np.nanstd(ifg_img - pred_NI_ifg))
    print()
    print('Pred_norm std:  ', np.nanstd(pred_norm))
    print('Pred_norm mean: ', np.nanmean(pred_norm))
    print('Pred_norm_ifg std: ', np.nanstd(pred_norm_ifg))
    print('Pred_norm_ifg mean: ', np.nanmean(pred_norm_ifg))
    print('Norm_model corrected std: ', np.nanstd(ifg_img - pred_norm_ifg))
    print()
    print('Pred_noLatLon std:  ', np.nanstd(pred_noLatLon))
    print('Pred_noLatLon mean: ', np.nanmean(pred_noLatLon))
    print('Pred_noLatLon_ifg std: ', np.nanstd(pred_noLatLon_ifg))
    print('Pred_noLatLon_ifg mean: ', np.nanmean(pred_noLatLon_ifg))
    print('Norm_model corrected std: ', np.nanstd(ifg_img - pred_noLatLon_ifg))

    plt.style.use('seaborn')
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    im1 = ax[0, 0].imshow(ifg_img,
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu')
    ax[0, 0].set_title('Ifg {}'.format(date))
    im2 = ax[0, 1].imshow(ifg_img - pred_NI_ifg,
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu', vmin=np.nanmin(ifg_img), vmax=np.nanmax(ifg_img))
    ax[0, 1].set_title('NI GPS int model corrected')
    im3 = ax[1, 0].imshow(ifg_img - pred_norm_ifg,
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu', vmin=np.nanmin(ifg_img), vmax=np.nanmax(ifg_img))
    ax[1, 0].set_title('Norm GPS int model corrected')
    im4 = ax[1, 1].imshow(ifg_img - pred_noLatLon_ifg,
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu', vmin=np.nanmin(ifg_img), vmax=np.nanmax(ifg_img))
    ax[1, 1].set_title('NoLatLon model corrected')
    fig.colorbar(im1, orientation='horizontal')
    fig.suptitle('NI GPS int Model corrected and Ifg {}'.format(date))
    fig.savefig('Plots/West/GPS_Model_corrected_v_Ifg{}.png'.format(date), dpi=300)
    plt.clf()

    plt.imshow(pred_NI_ifg, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
               cmap='RdYlBu')
    plt.title('NewInput model demean')
    plt.colorbar()
    plt.savefig('Plots/West/NI_GPS_pred_date_{}'.format(date), dpi=300)
    plt.clf()

    plt.imshow(pred_NI, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
               cmap='RdYlBu')
    plt.title('NewInput model')
    plt.colorbar()
    plt.savefig('Plots/West/NI_GPS_int_model_date_{}'.format(date), dpi=300)
    plt.clf()

    plt.imshow(pred_norm_ifg, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
               cmap='RdYlBu')
    plt.title('Norm model demean')
    plt.colorbar()
    plt.savefig('Plots/West/Norm_GPS_pred_date_{}'.format(date), dpi=300)
    plt.clf()

    plt.imshow(pred_norm, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
               cmap='RdYlBu')
    plt.title('Norm model')
    plt.colorbar()
    plt.savefig('Plots/West/Norm_GPS_int_model_date_{}'.format(date), dpi=300)
    plt.clf()

    plt.imshow(pred_noLatLon_ifg, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
               cmap='RdYlBu')
    plt.title('noLatLon model demean')
    plt.colorbar()
    plt.savefig('Plots/West/noLatLon_GPS_pred_date_{}'.format(date), dpi=300)
    plt.clf()

    plt.imshow(pred_noLatLon, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
               cmap='RdYlBu')
    plt.title('noLatLon model')
    plt.colorbar()
    plt.savefig('Plots/West/noLatLon_GPS_int_model_date_{}'.format(date), dpi=300)
    plt.clf()
