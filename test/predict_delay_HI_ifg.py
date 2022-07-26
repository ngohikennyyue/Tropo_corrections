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
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging

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

# Load scaler
# scaler_x_g = load('../../ML/Scaler/US_WE_MinMax_scaler_x.bin')
# scaler_y_g = load('../../ML/Scaler/US_WE_MinMax_scaler_y.bin')
scaler_x = load('../ML/Scaler/US_WE_noGOES_MinMax_scaler_x.bin')
scaler_y = load('../ML/Scaler/US_WE_noGOES_MinMax_scaler_y.bin')
GPS = pd.read_csv('../InSAR/Hawaii/GPS_station/product/UNRcombinedGPS_ztd.csv')
GPS = GPS[GPS['sigZTD'] < 0.01]

for i, date in enumerate(date_pairs):
    print(i + 1)
    ifg, grid = focus_bound('../InSAR/Hawaii/products/Extracted/unwrappedPhase/' + date, lon_min, lat_min, lon_max,
                            lat_max)
    dem, dem_grid = Resamp_rasterio('../InSAR/Hawaii/products/Extracted/DEM/SRTM_3arcsec_uncropped.tif', lon_min,
                                    lat_min, lon_max,
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
    wm1 = xr.load_dataset(" ".join(glob.glob(
        '../InSAR/Hawaii/weather_model/weather_files/' + 'ERA-5_{date}_T04_00_00*[A-Z].nc'.format(date=date1))))
    wm2 = xr.load_dataset(" ".join(glob.glob(
        '../InSAR/Hawaii/weather_model/weather_files/' + 'ERA-5_{date}_T04_00_00*[A-Z].nc'.format(date=date2))))
    df1 = GPS[(GPS['Date'] == date1.replace('_', '-')) & (GPS['Lat'] < lat_max) & (GPS['Lat'] > lat_min) & (
            GPS['Lon'] < lon_max) & (GPS['Lon'] > lon_min)]
    df2 = GPS[(GPS['Date'] == date2.replace('_', '-')) & (GPS['Lat'] < lat_max) & (GPS['Lat'] > lat_min) & (
            GPS['Lon'] < lon_max) & (GPS['Lon'] > lon_min)]

    if len(df1) != len(df2):
        df1 = df1[df1.ID.isin(df2.ID)]
        df2 = df2[df2.ID.isin(df1.ID)]
        xs = df1['Lon'].values
        ys = df1['Lat'].values
    else:
        xs = df1['Lon'].values
        ys = df1['Lat'].values

    rows, cols = get_rowcol('../InSAR/Hawaii/products/Extracted/unwrappedPhase/' + date, lon_min, lat_min, lon_max,
                            lat_max, xs, ys)

    # Plot ifg
    plt.style.use('seaborn')
    plt.imshow(convert_rad(ifg - np.nanmean(ifg), 5.6 / 100),
               extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
               cmap='RdYlBu')
    plt.colorbar(label='meters')
    plt.title('Raw Interferogram')
    plt.savefig('Plots/Hawaii/Ifg_GPS/Ifg_' + date + '.png', dpi=300)
    plt.clf()

    if i == 0:
        # Plot DEM (only the first one)
        dem = dem.astype(float)
        dem[dem <= 0.0] = np.nan
        plt.imshow(dem, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()])
        plt.colorbar()
        plt.savefig('Plots/Hawaii/Ifg_GPS/DEM.png', dpi=300)
        plt.clf()

        plt.imshow(los, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()])
        plt.colorbar()
        plt.savefig('Plots/Hawaii/Ifg_GPS/LOS.png', dpi=300)
        plt.clf()

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

    # Norm_model_prediction
    pred_1 = scaler_y.inverse_transform(Norm_model.predict(scaler_x.transform(Day_1)))
    pred_2 = scaler_y.inverse_transform(Norm_model.predict(scaler_x.transform(Day_2)))

    # Plot TD prediction of each date
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    fig.tight_layout(pad=5)
    im1 = ax1.imshow(pred_1.reshape(dem.shape), cmap='RdYlBu')
    sct1 = ax1.scatter(cols, rows, c=df1['ZTD'].values, cmap='RdYlBu', edgecolors='k', vmin=np.nanmin(pred_1),
                       vmax=np.nanmax(pred_1))
    plt.colorbar(sct1, ax=ax1, fraction=0.05, pad=0.02, label='Total Delay (m)')
    ax1.set_title('Total Delay of Date {}'.format(date1))
    im2 = ax2.imshow(pred_2.reshape(dem.shape), cmap='RdYlBu')
    sct2 = ax2.scatter(cols, rows, c=df2['ZTD'].values, cmap='RdYlBu', edgecolors='k', vmin=np.nanmin(pred_2),
                       vmax=np.nanmax(pred_2))
    plt.colorbar(sct2, ax=ax2, fraction=0.05, pad=0.02, label='Total Delay (m)')
    ax2.set_title('Total Delay of Date {}'.format(date2))
    fig.suptitle('TD prediction and GPS stations')
    fig.savefig('Plots/Hawaii/Ifg_GPS/TD_predict_cGPS_{}.png'.format(date), dpi=300)
    fig.clf()

    print('')
    print('Predict TD in two dates compare with GPS - ', date)
    print(date1)
    comb_date1 = pd.DataFrame(
        np.hstack((df1.ID.values.reshape(-1, 1), pred_1.reshape(dem.shape)[rows, cols].reshape(-1, 1),
                   df1['ZTD'].values.reshape(-1, 1),
                   (df1['ZTD'].values - pred_1.reshape(dem.shape)[rows, cols]).reshape(-1, 1))))
    comb_date1.columns = ['GPS_ID', 'Pred_ZTD', 'GPS_ZTD', 'Diff GPS - Pred']
    print(comb_date1)
    print('')
    print(date2)
    comb_date2 = pd.DataFrame(
        np.hstack((df1.ID.values.reshape(-1, 1), pred_2.reshape(dem.shape)[rows, cols].reshape(-1, 1),
                   df2['ZTD'].values.reshape(-1, 1),
                   (df2['ZTD'].values - pred_2.reshape(dem.shape)[rows, cols]).reshape(-1, 1))))
    comb_date2.columns = ['GPS_ID', 'Pred_ZTD', 'GPS_ZTD', 'Diff GPS - Pred']
    print(comb_date2)
    print('')

    TD = (pred_2 - pred_1).reshape(dem.shape)  # The difference of two dates will become the total delay
    GPS_TD = df2['ZTD'].values - df1['ZTD'].values
    plt.style.use('seaborn')
    plt.imshow(TD, cmap='RdYlBu')
    plt.scatter(cols, rows, c=GPS_TD, cmap='RdYlBu', edgecolors='k', vmin=np.nanmin(TD), vmax=np.nanmax(TD))
    plt.colorbar(label='Prediction Total Delay(m)')
    plt.title('Predicted ZTD compare with GPS ZTD ({})'.format(date))
    for n, value in enumerate(df1.ID):
        plt.annotate(value, (cols[n], rows[n]))
    plt.savefig('Plots/Hawaii/Ifg_GPS/Total_Delay_nGPS_{}.png'.format(date), dpi=300)
    plt.clf()

    print('')
    print('Predicted ZTD compare with GPS ZTD - ', date)
    comb_ztd = pd.DataFrame(
        np.hstack((df1.ID.values.reshape(-1, 1), TD[rows, cols].reshape(-1, 1), GPS_TD.reshape(-1, 1),
                   (GPS_TD - TD[rows, cols]).reshape(-1, 1))))
    comb_ztd.columns = ['GPS_ID', 'Pred_TD', 'GPS_TD', 'Diff GPS - TD']
    print(comb_ztd)
    print('')
    # Plot Ifg along side with TD (predicted)
    ifg_img = convert_rad(ifg - np.nanmean(ifg), 5.6 / 100)
    norm_TD = TD - np.nanmean(TD)
    S_norm_TD = norm_TD / np.cos(np.radians(los))
    S_GPS_TD = GPS_TD / np.cos(np.radians(los[rows, cols]))

    OK = OrdinaryKriging(xs.reshape(-1, 1),
                         ys.reshape(-1, 1),
                         S_GPS_TD.reshape(-1, 1),
                         variogram_model='linear',
                         verbose=False,
                         enable_plotting=False,
                         coordinates_type='geographic')
    XX = np.linspace(xs.min(), xs.max(), 100)
    YY = np.linspace(ys.min(), ys.max(), 100)
    z, ss = OK.execute('grid', XX, YY)

    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(10, 10))
    fig.tight_layout(pad=5)
    im1 = ax1[0].imshow(S_norm_TD, cmap='RdYlBu')
    sct1 = ax1[0].scatter(cols, rows, c=S_GPS_TD, cmap='RdYlBu', edgecolors='k', vmin=np.nanmin(S_norm_TD),
                          vmax=np.nanmax(S_norm_TD))
    plt.colorbar(sct1, ax=ax1[0], fraction=0.05, pad=0.02, label='(m)')
    ax1[0].set_title('Total Delay of Date {}'.format(date))
    im2 = ax1[1].imshow(ifg_img, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                        cmap='RdYlBu')
    sct2 = ax1[1].scatter(xs, ys, c=S_GPS_TD, cmap='RdYlBu', edgecolors='k', vmin=np.nanmin(ifg_img),
                          vmax=np.nanmax(ifg_img))
    plt.colorbar(sct2, ax=ax1[1], fraction=0.05, pad=0.02, label='(m)')
    ax1[1].set_title('Raw Interferogram {}'.format(date))
    im3 = ax2[0].imshow(z, cmap='RdYlBu', extent=[xs.min(), xs.max(), ys.min(), ys.max()])
    sct3 = ax2[0].scatter(xs, ys, c=S_GPS_TD, cmap='RdYlBu', edgecolors='k', vmin=np.nanmin(z), vmax=np.nanmax(z))
    plt.colorbar(sct3, ax=ax2[0], fraction=0.05, pad=0.02, label='(m)')
    ax2[0].set_title('Kriging delay from GPS station')
    fig.suptitle('Predicted TD, GPS TD, and Ifg in STD {}'.format(date))
    fig.savefig('Plots/Hawaii/Ifg_GPS/Ifg_Pred_GPS_TD_{}.png'.format(date), dpi=300)
    fig.clf()
    print('')
    print('Predicted TD, GPS TD , and Ifg in STD - ', date)
    comp_std = pd.DataFrame(np.hstack((df1.ID.values.reshape(-1, 1), (norm_TD / np.cos(los))[rows, cols].reshape(-1, 1),
                                       ifg_img[rows, cols].reshape(-1, 1), S_GPS_TD.reshape(-1, 1),
                                       (ifg_img[rows, cols] - S_GPS_TD).reshape(-1, 1))))
    comp_std.columns = ['GPS_ID', 'ML_pred', 'IFG', 'GPS_station', 'Diff GPS-IFG']

    print(comp_std)
    print('')
    # print('')
    # print('Compare STD ML prediction, GPS and Ifg ', date)
    # print('Prediction: ', (norm_TD / np.cos(los))[rows, cols], ' std: ', np.std((norm_TD / np.cos(los))[rows, cols]))
    # print('IFG value: ', ifg_img[rows, cols], ' std: ', np.std(ifg_img[rows, cols]))
    # print('GPS station: ', S_GPS_TD, ' std: ', np.std(GPS_TD - np.mean(GPS_TD)))
    # print('Diff GPS - IFG: ', ifg_img[rows, cols] - S_GPS_TD)
    # print('')
