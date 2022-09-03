import matplotlib.pyplot as plt
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
import numpy as np

date_pairs = ['20190714_20190702', '20190714_20190708', '20190720_20190714', '20190807_20190801',
              '20190819_20190813', '20190831_20190825', '20190906_20190831', '20190918_20190906']

lon_min, lat_min, lon_max, lat_max = -156, 18.9, -155, 19.9

# Load Model
model = tf.keras.models.load_model('../ML/IFG_model/Hawaii_model/Model/ifg_Hawaii_ref_model_batchsize_512')
GPS_int_model = tf.keras.models.load_model('../ML/GNSS_Int_model/Model/GPS_int_PTE_slope_model')
# Load scaler
scaler_x = load('../ML/IFG_model/Hawaii_model/Scaler/ifg_Hawaii_ref_model_MinMax_scaler_x.bin')
scaler_y = load('../ML/IFG_model/Hawaii_model/Scaler/ifg_Hawaii_ref_model_MinMax_scaler_y.bin')
GPS_scaler_x = load('../ML/GNSS_Int_model/Scaler/GPS_int_PTE_slope_model_MinMax_scaler_x.bin')
GPS_scaler_y = load('../ML/GNSS_Int_model/Scaler/GPS_int_PTE_slope_model_MinMax_scaler_y.bin')
# GPS = pd.read_csv('../InSAR/Hawaii/GPS_station/product/UNRcombinedGPS_ztd.csv')
# GPS = GPS[GPS['sigZTD'] < 0.01]
wm_file_path = '../InSAR/Large_scale/Hawaii/weather_files/'
time = 'T04_30_00'
ref_point = [-155.4, 19.6]
for i, date in enumerate(date_pairs):
    print(i + 1)
    ifg, grid = focus_bound('../InSAR/Large_scale/Hawaii/Extracted/unwrappedPhase/' + date, lon_min, lat_min, lon_max,
                            lat_max)
    dem, dem_grid = Resamp_rasterio('../InSAR/Large_scale/Hawaii/Extracted/DEM/SRTM_3arcsec_uncropped.tif', lon_min,
                                    lat_min, lon_max,
                                    lat_max, ifg)
    print('DEM_size:', dem_grid.shape)
    print('ifg_size: ', grid.shape)
    los, _ = Resamp_rasterio('../InSAR/Large_scale/Hawaii/Angle/los.rdr', lon_min, lat_min, lon_max, lat_max, ifg)
    slope, _ = Resamp_rasterio('../InSAR/Large_scale/Hawaii/slope.tif', lon_min, lat_min, lon_max, lat_max, ifg)
    # print('DEM: ', dem_grid[0:5])
    # Create a mask for IFG
    mask = dem.copy()
    mask[mask > 0] = 1
    mask[mask <= 0] = np.nan
    ifg = ifg * mask
    convert_ifg = convert_rad(ifg - np.nanmean(ifg), 5.6 / 100)
    # Plot ifg
    plt.style.use('seaborn')
    plt.imshow(convert_ifg,
               extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
               cmap='RdYlBu')
    plt.colorbar(label='meters')
    plt.title('Raw Interferogram')
    plt.savefig('Plots/Hawaii/Ifg_512/Ifg_' + date + '.png', dpi=300)
    plt.clf()

    if i == 0:
        # Plot DEM (only the first one)
        dem = dem.astype(float)
        dem[dem <= 0.0] = np.nan
        plt.imshow(dem, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()])
        plt.colorbar()
        plt.savefig('Plots/Hawaii/Ifg_512/DEM.png', dpi=300)
        plt.clf()

        plt.imshow(los, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()])
        plt.colorbar()
        plt.savefig('Plots/Hawaii/Ifg_512/LOS.png', dpi=300)
        plt.clf()

    else:
        pass
    dem = dem.astype(float)
    dem[dem <= 0.0] = np.nan

    # Get date and WM
    date1, date2 = get_datetime(date)
    WM1, wm1, minute = getWM(date1, time, wmLoc=wm_file_path)
    WM2, wm2, _ = getWM(date2, time, wmLoc=wm_file_path)
    P1, T1, E1 = interpByTime(WM1, wm1, minute, 'all')
    p1, t1, e1 = interpByTime(WM2, wm2, minute, 'all')

    # Get all the x,y coordinate
    x = list(set(dem_grid[:, 0]))
    x.sort()
    y = list(set(dem_grid[:, 1]))
    y.sort()

    # Day1 WM (PTe) parameters dereference
    P1 = (P1 - P1.sel(x=ref_point[0], y=ref_point[1], method='nearest')).interp(x=x, y=y).values
    T1 = (T1 - T1.sel(x=ref_point[0], y=ref_point[1], method='nearest')).interp(x=x, y=y).values
    E1 = (E1 - E1.sel(x=ref_point[0], y=ref_point[1], method='nearest')).interp(x=x, y=y).values

    # Day2 WM (PTe) parameters dereference
    p1 = (p1 - p1.sel(x=ref_point[0], y=ref_point[1], method='nearest')).interp(x=x, y=y).values
    t1 = (t1 - t1.sel(x=ref_point[0], y=ref_point[1], method='nearest')).interp(x=x, y=y).values
    e1 = (e1 - e1.sel(x=ref_point[0], y=ref_point[1], method='nearest')).interp(x=x, y=y).values

    inf_dat = pd.DataFrame(np.hstack((dem_grid,
                                      dem.ravel().reshape(-1, 1),
                                      P1.transpose().reshape((P1.shape[-1] * P1.shape[1], len(hgtlvs))),
                                      T1.transpose().reshape((T1.shape[-1] * T1.shape[1], len(hgtlvs))),
                                      E1.transpose().reshape((E1.shape[-1] * E1.shape[1], len(hgtlvs))),
                                      p1.transpose().reshape((p1.shape[-1] * p1.shape[1], len(hgtlvs))),
                                      t1.transpose().reshape((t1.shape[-1] * t1.shape[1], len(hgtlvs))),
                                      e1.transpose().reshape((e1.shape[-1] * e1.shape[1], len(hgtlvs))),
                                      slope.reshape(-1, 1))))
    inf_dat.columns = ['Lon', 'Lat', 'Hgt_m'] + \
                      ['date1_P_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                      ['date1_T_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                      ['date1_e_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                      ['date2_P_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                      ['date2_T_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                      ['date2_e_' + str(i) for i in range(1, len(hgtlvs) + 1)] + \
                      ['slope']
    hgtlv = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400
        , 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000
        , 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000
        , 14000, 15000, 16000, 17000, 18000, 19000, 20000, 25000, 30000, 35000, 40000]

    P1_, T1_, E1_ = interpByTime(WM1, wm1, minute, 'all')
    p1_, t1_, e1_ = interpByTime(WM2, wm2, minute, 'all')

    # Day1 WM (PTe) parameters dereference
    P1_ = P1_.interp(x=x, y=y, z=hgtlv).values
    T1_ = T1_.interp(x=x, y=y, z=hgtlv).values
    E1_ = E1_.interp(x=x, y=y, z=hgtlv).values

    # Day2 WM (PTe) parameters dereference
    p1_ = p1_.interp(x=x, y=y, z=hgtlv).values
    t1_ = t1_.interp(x=x, y=y, z=hgtlv).values
    e1_ = e1_.interp(x=x, y=y, z=hgtlv).values

    dat = pd.DataFrame(np.hstack((dem_grid,
                                  dem.ravel().reshape(-1, 1),
                                  P1_.transpose().reshape((P1_.shape[-1] * P1_.shape[1], len(hgtlv))),
                                  T1_.transpose().reshape((T1_.shape[-1] * T1_.shape[1], len(hgtlv))),
                                  E1_.transpose().reshape((E1_.shape[-1] * E1_.shape[1], len(hgtlv))),
                                  p1_.transpose().reshape((p1_.shape[-1] * p1_.shape[1], len(hgtlv))),
                                  t1_.transpose().reshape((t1_.shape[-1] * t1_.shape[1], len(hgtlv))),
                                  e1_.transpose().reshape((e1_.shape[-1] * e1_.shape[1], len(hgtlv))),
                                  slope.reshape(-1, 1))))
    dat.columns = ['Lon', 'Lat', 'Hgt_m'] + \
                  ['date1_P_' + str(i) for i in range(1, len(hgtlv) + 1)] + \
                  ['date1_T_' + str(i) for i in range(1, len(hgtlv) + 1)] + \
                  ['date1_e_' + str(i) for i in range(1, len(hgtlv) + 1)] + \
                  ['date2_P_' + str(i) for i in range(1, len(hgtlv) + 1)] + \
                  ['date2_T_' + str(i) for i in range(1, len(hgtlv) + 1)] + \
                  ['date2_e_' + str(i) for i in range(1, len(hgtlv) + 1)] + \
                  ['slope']
    X = inf_dat[
        inf_dat.columns[pd.Series(inf_dat.columns).str.startswith(('Lat', 'Hgt_m', 'date1_', 'date2_', 'slope'))]]
    x = dat[dat.columns[pd.Series(dat.columns).str.startswith(('Lat', 'Hgt_m', 'date1_', 'date2_', 'slope'))]]
    # Norm_model_prediction
    pred = scaler_y.inverse_transform(model.predict(scaler_x.transform(X)))
    pred_GPS = GPS_scaler_y.inverse_transform(GPS_int_model.predict(GPS_scaler_x.transform(x)))
    # print(convert_ifg.ravel().reshape(-1,1).shape)
    # print(pred.reshape(dem.shape) / np.cos(np.radians(los)))
    # print(los.ravel().shape)
    # print_metric(convert_ifg, pred.reshape(dem.shape) / np.cos(np.radians(los)),
    #              name='Ref_512batch_model_{}'.format(date))
    # plot_graphs(convert_ifg, pred.reshape(dem.shape) / np.cos(np.radians(los)),
    #             model='Ref_512batch_model_{}'.format(date), save_loc='Plots/Hawaii/Ifg_512')
    # print_metric(convert_ifg, pred_GPS.reshape(dem.shape) / np.cos(np.radians(los)),
    #              name='GPS_Int_model_{}'.format(date))
    # plot_graphs(convert_ifg, pred_GPS.reshape(dem.shape) / np.cos(np.radians(los)),
    #             model='GPS_Int_model_{}'.format(date), save_loc='Plots/Hawaii/Ifg_512')
    print('GPS_int_model')
    # Plot TD prediction of each date
    ref_pred = pred.reshape(dem.shape) / np.cos(np.radians(los))
    ref_pred = ref_pred - np.nanmean(ref_pred)
    gps_int_pred = pred_GPS.reshape(dem.shape) / np.cos(np.radians(los))
    gps_int_pred =  gps_int_pred - np.nanmean(gps_int_pred)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))
    fig.tight_layout(pad=5)
    im1 = ax1.imshow(ref_pred, cmap='RdYlBu', vmin=np.nanmin(convert_ifg), vmax=np.nanmax(convert_ifg))
    ax1.set_title('Ref 512 STD of Date {}'.format(date))
    im2 = ax2.imshow(convert_ifg, cmap='RdYlBu')
    plt.colorbar(im2, ax=ax2, fraction=0.05, pad=0.02, label='IFG (m)')
    ax2.set_title('IFG {}'.format(date))
    im3 = ax3.imshow(gps_int_pred, cmap='RdYlBu', vmin=np.nanmin(convert_ifg), vmax=np.nanmax(convert_ifg))
    ax3.set_title('GPS STD of Date {}'.format(date))
    fig.suptitle('Predict STD and IFG')
    fig.savefig('Plots/Hawaii/Ifg_512/STD_predict_IFG_{}.png'.format(date), dpi=300)
    fig.clf()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))
    fig.tight_layout(pad=5)
    im1 = ax1.imshow(convert_ifg - gps_int_pred, cmap='RdYlBu', vmin=np.nanmin(convert_ifg),
                     vmax=np.nanmax(convert_ifg))
    ax1.set_title('GPS int model corrected {}'.format(date))
    im2 = ax2.imshow(convert_ifg, cmap='RdYlBu')
    plt.colorbar(im2, ax=ax2, fraction=0.05, pad=0.02, label='(m)')
    ax2.set_title('IFG {}'.format(date))
    im3 = ax3.imshow(convert_ifg - ref_pred, cmap='RdYlBu', vmin=np.nanmin(convert_ifg),
                     vmax=np.nanmax(convert_ifg))
    ax3.set_title('Ref_model corrected {}'.format(date))
    fig.suptitle('GPS int model corrected and IFG')
    fig.savefig('Plots/Hawaii/Ifg_512/GPS_int_model_corrected_predict_IFG_{}.png'.format(date), dpi=300)
    fig.clf()
    print('IFG std {}:'.format(date), np.nanstd(convert_ifg))
    print('GPS Corrected std {}:'.format(date), np.nanstd(convert_ifg - gps_int_pred))
    print('Ref_model Corrected std {}:'.format(date), np.nanstd(convert_ifg - ref_pred))
    print()
