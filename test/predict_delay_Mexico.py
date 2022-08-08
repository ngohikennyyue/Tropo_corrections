import pandas as pd
import sys
import os
import tensorflow as tf

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *
from joblib import load

df = pd.read_csv('../GNSS_US/Mexico/PTE_vert_fixed_hgtlvs.csv')
df = df.dropna()
df = df[df['sigZTD'] < 0.1]
dat = pd.read_csv('../GNSS_US/Mexico/Mexico_node_delay_vert_fixed_hgtlvs.csv')
dat = dat.dropna()
dat = dat[dat['sigZTD'] < 0.1]
int_dat = pd.read_csv('../GNSS_US/Mexico/Mexico_inter_PTE_vert_fixed_hgtlvs.csv')
int_dat = int_dat.dropna()
slope = pd.read_csv('../GNSS_US/Mexico/PTE_vert_fixed_hgtlvs_slope.csv')
slope = slope.dropna()
cloud = pd.read_csv('../GNSS_US/Mexico/PTE_vert_fixed_hgtlvs_cloud.csv')
cloud = cloud.dropna()

# lon_min, lat_min, lon_max, lat_max = -155.9, 18.9, -154.9, 19.9

# hgtlvs = [-100, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400,
#           2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000,
#           5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000,
#           14000, 15000, 16000, 17000, 18000, 19000, 20000, 25000, 30000, 35000, 40000]
# Load Model
Norm_model = tf.keras.models.load_model('../ML/Model/Full_US_PTE_fixed_hgtlvs_model')
Multi_model = tf.keras.models.load_model('../ML/Multiple_Input_Model/Model'
                                         '/Test_New_model3_US_PTE_fixed_hgtlvs_cloud_model')
wet_hydro_model = tf.keras.models.load_model('../ML/Wet_hydro_model/Model/Full_wet_hydro_US_PTE_fixed_hgtlvs_model')
combined_mod_model = tf.keras.models.load_model(
    '../ML/Combined_model/Model/combined_model_mod_US_PTE_fixed_hgtlvs_model')
new_Norm_model = tf.keras.models.load_model('../ML/No_GOES_model/Model/US_PTE_fixed_hgtlvs_model')
inter_model = tf.keras.models.load_model('../ML/Inter_model/Model/inter_PTE_fixed_hgtlvs_model')
slope_model = tf.keras.models.load_model('../ML/Slope_model/Model/PTE_fixed_hgtlvs_slope_model')
GOES_model = tf.keras.models.load_model('../ML/GOES_model/Model/PTE_fixed_hgtlvs_GOES_model')
GOES_DOY_model = tf.keras.models.load_model('../ML/GOES_model/Model/PTE_fixed_hgtlvs_GOES_DOY_model')

# Load scaler
scaler_x = load('../ML/Scaler/US_WE_noGOES_MinMax_scaler_x.bin')
scaler_y = load('../ML/Scaler/US_WE_noGOES_MinMax_scaler_y.bin')
scalerP = load('../ML/Multiple_Input_Model/Scaler/Test_New_model3_pScaler_x.bin')
scalerT = load('../ML/Multiple_Input_Model/Scaler/Test_New_model3_tScaler_x.bin')
scalerE = load('../ML/Multiple_Input_Model/Scaler/Test_New_model3_eScaler_x.bin')
scaler_y1 = load('../ML/Multiple_Input_Model/Scaler/Test_New_model3_scaler_y.bin')
wet_scaler_x = load('../ML/Wet_hydro_model/Scaler/Full_wet_hydro_model_scaler_x.bin')
wet_scaler_y = load('../ML/Wet_hydro_model/Scaler/Full_wet_hydro_model_scaler_y.bin')
Nscaler_x = load('../ML/No_GOES_model/Scaler/US_noGOES_MinMax_scaler_x.bin')
Nscaler_y = load('../ML/No_GOES_model/Scaler/US_noGOES_MinMax_scaler_y.bin')
inter_scaler_x = load('../ML/Inter_model/Scaler/interferometric_MinMax_scaler_x.bin')
inter_scaler_y = load('../ML/Inter_model/Scaler/interferometric_MinMax_scaler_y.bin')
slope_scaler_x = load('../ML/Slope_model/Scaler/Slope_MinMax_scaler_x.bin')
slope_scaler_y = load('../ML/Slope_model/Scaler/Slope_MinMax_scaler_y.bin')
GOES_scaler_x = load('../ML/GOES_model/Scaler/GOES_MinMax_scaler_x.bin')
GOES_scaler_y = load('../ML/GOES_model/Scaler/GOES_MinMax_scaler_y.bin')
GOES_DOY_scaler_x = load('../ML/GOES_model/Scaler/GOES_DOY_MinMax_scaler_x.bin')
GOES_DOY_scaler_y = load('../ML/GOES_model/Scaler/GOES_DOY_MinMax_scaler_y.bin')

# Obtain the input variables
X = df[df.columns[pd.Series(df.columns).str.startswith(('Lat', 'Hgt_m', 'P_', 'T_', 'e_'))]]
DOY_X = cloud[cloud.columns[pd.Series(cloud.columns).str.startswith(('DOY', 'Lat', 'Hgt_m', 'P_', 'T_', 'e_', 'CMI_C'))]]
P = df[df.columns[pd.Series(df.columns).str.startswith(('Lat', 'Hgt_m', 'P_'))]]
T = df[df.columns[pd.Series(df.columns).str.startswith(('Lat', 'Hgt_m', 'T_'))]]
E = df[df.columns[pd.Series(df.columns).str.startswith(('Lat', 'Hgt_m', 'e_'))]]
wet = dat[dat.columns[pd.Series(dat.columns).str.startswith(('Lat', 'Hgt_m', 'total_'))]]
int_X = int_dat[int_dat.columns[pd.Series(int_dat.columns).str.startswith(('Lat', 'Hgt_m', 'P_', 'T_', 'e_'))]]
slopeX = slope[slope.columns[pd.Series(slope.columns).str.startswith(('Lat', 'Hgt_m', 'P_', 'T_', 'e_', 'Slope'))]]
cloudX = cloud[cloud.columns[pd.Series(cloud.columns).str.startswith(('Lat', 'Hgt_m', 'P_', 'T_', 'e_', 'CMI_C'))]]

print('length of wet data:', len(wet))
print('length of others: ',  len(X))
# Predict
predict1 = scaler_y.inverse_transform(Norm_model.predict(scaler_x.transform(X)))
predict2 = scaler_y1.inverse_transform(
    Multi_model.predict([scalerP.transform(P), scalerT.transform(T), scalerE.transform(E)]))
predict3 = wet_scaler_y.inverse_transform(wet_hydro_model.predict(wet_scaler_x.transform(wet)))
predict4 = combined_mod_model.predict(np.hstack((predict1.reshape(-1, 1), predict2.reshape(-1, 1))))
predict5 = Nscaler_y.inverse_transform(new_Norm_model.predict(Nscaler_x.transform(X)))
predict6 = inter_scaler_y.inverse_transform(inter_model.predict(inter_scaler_x.transform(int_X)))
predict7 = slope_scaler_y.inverse_transform(slope_model.predict(slope_scaler_x.transform(slopeX)))
predict8 = GOES_scaler_y.inverse_transform(GOES_model.predict(GOES_scaler_x.transform(cloudX)))
predict9 = GOES_DOY_scaler_y.inverse_transform(GOES_DOY_model.predict(GOES_DOY_scaler_x.transform(DOY_X)))

true1 = df[['ZTD']].values
true2 = dat[['ZTD']].values
true3 = int_dat[['inf_ZTD']].values
true4 = slope[['ZTD']].values
true5 = cloud[['ZTD']].values

print('')
print('Normal_model:')
print('Predict: ', predict1[:5].ravel())
print('True: ', true1[:5].ravel())
print('Diff: ', true1[:5].ravel() - predict1[:5].ravel())
print('')
print('Multi_input model:')
print('Predict: ', predict2[:5].ravel())
print('True: ', true1[:5].ravel())
print('Diff: ', true1[:5].ravel() - predict2[:5].ravel())
print('')
print('Wet hydro model:')
print('Predict: ', predict3[:5].ravel())
print('True: ', true2[:5].ravel())
print('Diff: ', true2[:5].ravel() - predict3[:5].ravel())
print('')
print('Combined model:')
print('Predict: ', predict4[:5].ravel())
print('True: ', true1[:5].ravel())
print('Diff: ', true1[:5].ravel() - predict4[:5].ravel())
print('')
print('New Normal model:')
print('Predict: ', predict5[:5].ravel())
print('True: ', true1[:5].ravel())
print('Diff: ', true1[:5].ravel() - predict5[:5].ravel())
print('')
print('Interferometric model:')
print('Predict: ', predict6[:5].ravel())
print('True: ', true3[:5].ravel())
print('Diff: ', true3[:5].ravel() - predict6[:5].ravel())
print('')
print('Slope model:')
print('Predict: ', predict7[:5].ravel())
print('True: ', true4[:5].ravel())
print('Diff: ', true4[:5].ravel() - predict7[:5].ravel())
print('')
print('GOES model:')
print('Predict: ', predict8[:5].ravel())
print('True: ', true5[:5].ravel())
print('Diff: ', true5[:5].ravel() - predict8[:5].ravel())
print('')
print('GOES DOY model:')
print('Predict: ', predict9[:5].ravel())
print('True: ', true5[:5].ravel())
print('Diff: ', true5[:5].ravel() - predict9[:5].ravel())
print('')

from sklearn.metrics import mean_squared_error, r2_score

print_metric(true1, predict1, 'Normal model')
print_metric(true1, predict2, 'Multi-input model')
print_metric(true2, predict3, 'Full Wet Hydro model')
print_metric(true1, predict4, 'Combined model')
print_metric(true1, predict5, 'New Normal model')
print_metric(true3, predict6, 'Interferometric model')
print_metric(true4, predict7, 'Slope model')
print_metric(true5, predict8, 'GOES model')
print_metric(true5, predict9, 'GOES DOY model')

print('Make plots')
plot_graphs(true1, predict1, 'Normal model', 'Plots/Mexico')
plot_graphs(true1, predict2, 'Multi-input model', 'Plots/Mexico')
plot_graphs(true2, predict3, 'Full Wet Hydro model', 'Plots/Mexico')
plot_graphs(true1, predict4, 'Combined model', 'Plots/Mexico')
plot_graphs(true1, predict5, 'New Normal model', 'Plots/Mexico')
plot_graphs(true3, predict6, 'Interferometric model', 'Plots/Mexico')
plot_graphs(true4, predict7, 'Slope model', 'Plots/Mexico')
plot_graphs(true5, predict8, 'GOES model', 'Plots/Mexico')
plot_graphs(true5, predict9, 'GOES DOY model', 'Plots/Mexico')

# # G-matrix comparison
# G = np.stack((predict1.ravel(), predict2.ravel(), np.ones_like(predict1.ravel())), axis=1)
# print(G[:5, :])
# mhat, res, rank, s = np.linalg.lstsq(G, true)
# print(mhat, res, rank, s)
# y_pred = np.dot(G, mhat)
#
# print('')
# print("G-matrix")
# # The mean squared error
# print('Mean squared error: %.10f' % mean_squared_error(true, y_pred))
# # The R2 score
# print('R2: %.5f' % r2_score(true, y_pred))
# # The RMSE
# rmse = np.sqrt(mean_squared_error(true, y_pred))
# print('RMSE: %.5f' % rmse)
# errors = y_pred - true
# print('Average error: %.5f' % np.mean(abs(errors)))
# print('')
#
# # Plot of residual of the prediction
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
# density = ax.scatter_density(true, true - y_pred, cmap=white_viridis)
# cbar = fig.colorbar(density)
# cbar.set_label(label='Number of points per pixel', size=10)
# ax.tick_params(axis='both', which='major', labelsize=10)
# plt.xlabel('True', fontsize=10)
# plt.ylabel('Residual', fontsize=10)
# cbar.ax.tick_params(labelsize=10)
# fig.suptitle('G-matrix Residual')
# fig.savefig('Plots/Mexico/G-matrix_Resid_true.png', dpi=300)
#
# # Plot of Observation vs Prediction
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
# density = ax.scatter_density(true, y_pred, cmap=white_viridis)
# cbar = fig.colorbar(density)
# cbar.set_label(label='Number of points per pixel', size=10)
# ax.tick_params(axis='both', which='major', labelsize=10)
# plt.xlabel('Observed', fontsize=10)
# plt.ylabel('Predicted', fontsize=10)
# cbar.ax.tick_params(labelsize=10)
# fig.suptitle('G-matrix obs vs pred')
# fig.savefig('Plots/Mexico/G-matrix_Ob_v_Pred.png', dpi=300)
