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
date_pairs = glob.glob('../InSAR/Pennsylvania/Extracted/unwrappedPhase/*[0-9]')
date_pairs.sort()
lon_min, lat_min, lon_max, lat_max = -78, 40, -77, 41
print('Bounding area: ', lon_min, lat_min, lon_max, lat_max)

hgtlvs = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400,
          2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000,
          5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000,
          14000, 15000, 16000, 17000, 18000, 19000, 20000, 25000, 30000, 35000, 40000, 45000]
# Load Model
hywt_model = tf.keras.models.load_model('../ML/GNSS_Int_model/Model/GPS_int_hywt_interp_model')
GNSS_int_model = tf.keras.models.load_model('../ML/Inter_model/Model/New_inter_PTE_fixed_hgtlvs_model')

# Load scaler
hywt_scaler_x = load('../ML/GNSS_Int_model/Scaler/GPS_int_hywt_interp_model_MinMax_scaler_x.bin')
hywt_scaler_y = load('../ML/GNSS_Int_model/Scaler/GPS_int_hywt_interp_model_MinMax_scaler_y.bin')
GNSS_int_scaler_x = load('../ML/Inter_model/Scaler/New_interferometric_MinMax_scaler_x.bin')
GNSS_int_scaler_y = load('../ML/Inter_model/Scaler/New_interferometric_MinMax_scaler_y.bin')

for i, intfg in enumerate(date_pairs[::5]):
    ifg, grid = focus_bound(intfg, lon_min, lat_min, lon_max, lat_max)
    dem, dem_grid = Resamp_rasterio('../InSAR/Pennsylvania/Extracted/DEM/SRTM_3arcsec_uncropped.tif', lon_min,
                                    lat_min, lon_max, lat_max, ifg)
    # slope, _ = Resamp_rasterio('../InSAR/Pennsylvania/slope.tif', lon_min, lat_min, lon_max, lat_max, ifg)
    los, _ = Resamp_rasterio('../InSAR/Pennsylvania/Angle/los.rdr', lon_min, lat_min, lon_max, lat_max, ifg)
    date = intfg.split('/')[-1]
    print('LOS: ', los[:10])
    print('DEM lat_lon: ', dem_grid[0:5])
    # Create a mask for IFG
    mask = dem.copy()
    mask[mask != -32768] = 1
    mask[mask == -32768] = np.nan
    ifg = ifg * mask
    row, col = get_rowcol(intfg, lon_min, lat_min, lon_max, lat_max, -77.6, 40.6)
    ref = ifg[row, col]
    # Plot ifg
    plt.style.use('seaborn')
    plt.imshow(convert_rad(ifg - ref, 5.6 / 100),
               extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
               cmap='RdYlBu')
    plt.colorbar(label='meters')
    plt.title('Raw Interferogram')
    plt.savefig('Plots/Pennsylvania/New/Ifg_' + date + '.png', dpi=300)
    plt.clf()

    dem = dem.astype(float)
    dem[dem == -32768] = np.nan
    if i == 0:
        # Plot DEM (only the first one)
        plt.imshow(dem, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()])
        plt.colorbar()
        plt.savefig('Plots/Pennsylvania/New/DEM.png', dpi=300)
        plt.clf()

        plt.imshow(los, extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()])
        plt.colorbar()
        plt.savefig('Plots/Pennsylvania/LOS.png', dpi=300)
        plt.clf()
    else:
        pass
    # Get Dates and WM
    date1, date2 = get_datetime(date)
    print("date1: ", date1)
    print('date2: ', date2)

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

    hydro1_interp = make_interpretor(wm1, 'hydro_total')
    wet1_interp = make_interpretor(wm1, 'wet_total')
    hydro2_interp = make_interpretor(wm2, 'hydro_total')
    wet2_interp = make_interpretor(wm2, 'wet_total')
    loc = np.hstack((dem_grid, dem.ravel().reshape(-1, 1)))
    print(loc[5, :])

    hydro1 = hydro1_interp(loc)
    wet1 = wet1_interp(loc)
    hydro2 = hydro2_interp(loc)
    wet2 = wet2_interp(loc)

    data1 = np.hstack((dem_grid[:, 1].reshape(-1, 1), dem.ravel().reshape(-1, 1),
                       hydro2.ravel().reshape(-1, 1), wet2.ravel().reshape(-1, 1),
                       hydro1.ravel().reshape(-1, 1), wet1.ravel().reshape(-1, 1)))

    data2 = np.hstack((dem_grid[:, 1].reshape(-1, 1), dem.ravel().reshape(-1, 1),
                       np.vstack((P2 - P1).transpose().reshape((P2.shape[-1] * P2.shape[1], 51))),
                       np.vstack((T2 - T1).transpose().reshape((T2.shape[-1] * T2.shape[1], 51))),
                       np.vstack((e2 - e1).transpose().reshape((e2.shape[-1] * e2.shape[1], 51)))))

    # Norm_model_prediction
    pred_1 = hywt_scaler_y.inverse_transform(hywt_model.predict(hywt_scaler_x.transform(data1)))
    pred_2 = GNSS_int_scaler_y.inverse_transform(GNSS_int_model.predict(GNSS_int_scaler_x.transform(data2)))
    IFG_pred_1 = pred_1.reshape(dem.shape) - pred_1.reshape(dem.shape)[row, col]

    plt.imshow(IFG_pred_1, cmap='RdYlBu')
    plt.colorbar(label='Prediction Total Delay(m)')
    plt.title('hywt model prediction ({})'.format(date))
    plt.savefig('Plots/Pennsylvania/New/hywt_model_inf_Delay_{}.png'.format(date), dpi=300)
    plt.clf()

    IFG_pred_2 = pred_2.reshape(dem.shape) - pred_2.reshape(dem.shape)[row, col]
    plt.imshow(IFG_pred_2, cmap='RdYlBu')
    plt.colorbar(label='Prediction Total Delay(m)')
    plt.title('GNSS int model prediction ({})'.format(date))
    plt.savefig('Plots/Pennsylvania/New/GNSS_int_model_inf_Delay_{}.png'.format(date), dpi=300)
    plt.clf()

    raider_delay = ((hydro2 + wet2) - (hydro1 + wet1)).reshape(dem.shape)
    raider_delay = raider_delay - raider_delay[row, col]
    plt.imshow(raider_delay, cmap='RdYlBu')
    plt.colorbar(label='Predicted Delay(m)')
    plt.title('Raider prediction {}'.format(date))
    plt.savefig('Plots/Pennsylvania/New/Raider_inf_Delay_{}.png'.format(date), dpi=300)
    plt.clf()

    # Plot Ifg along side with TD (predicted)
    ifg_img = convert_rad(ifg - ref, 5.6 / 100)
    s_IFG_pred_1 = IFG_pred_1 / np.cos(np.radians(los))
    s_IFG_pred_2 = IFG_pred_2 / np.cos(np.radians(los))
    s_raider_delay = raider_delay / np.cos(np.radians(los))

    plt.style.use('seaborn')
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    im1 = ax[0, 0].imshow(s_IFG_pred_1,
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu', vmin=np.nanmin(ifg_img), vmax=np.nanmax(ifg_img))
    ax[0, 0].set_title('hywt Model inf delay {}'.format(date))
    im2 = ax[0, 1].imshow(ifg_img,
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu')
    ax[0, 1].set_title('Raw Interferogram {}'.format(date))
    im3 = ax[0, 2].imshow(ifg_img - s_IFG_pred_1,
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu', vmin=np.nanmin(ifg_img), vmax=np.nanmax(ifg_img))
    ax[0, 2].set_title('hywt model corrected {}'.format(date))
    im4 = ax[1, 0].imshow(s_IFG_pred_2,
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu', vmin=np.nanmin(ifg_img), vmax=np.nanmax(ifg_img))
    ax[1, 0].set_title('GNSS int model inf Delay {}'.format(date))
    im5 = ax[1, 1].imshow(ifg_img,
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu')

    im6 = ax[1, 2].imshow(ifg_img - s_IFG_pred_2,
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu', vmin=np.nanmin(ifg_img), vmax=np.nanmax(ifg_img))
    ax[1, 2].set_title('GNSS int model corrected {}'.format(date))
    im7 = ax[2, 0].imshow(s_raider_delay,
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu', vmin=np.nanmin(ifg_img), vmax=np.nanmax(ifg_img))
    ax[2, 0].set_title('Raider delay {}'.format(date))
    im8 = ax[2, 1].imshow(ifg_img,
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu')
    im9 = ax[2, 2].imshow(ifg_img - s_raider_delay,
                          extent=[grid[:, 0].min(), grid[:, 0].max(), grid[:, -1].min(), grid[:, -1].max()],
                          cmap='RdYlBu', vmin=np.nanmin(ifg_img), vmax=np.nanmax(ifg_img))
    ax[2, 2].set_title('RAiDER corrected {}'.format(date))
    fig.colorbar(im2)
    fig.suptitle('GNSS int Model, hywt Model, Raw IFG in slanted direction')
    fig.savefig('Plots/Pennsylvania/New/Models_prediction_{}.png'.format(date), dpi=300)
    plt.clf()
