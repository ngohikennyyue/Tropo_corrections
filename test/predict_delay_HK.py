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

df = pd.read_csv('../GNSS/Subtrop/HK/GNSS_subtrop_hgtlvs_fixed.csv')
df = df.dropna()
df = df[df['sigZTD'] < 0.1]
dat = pd.read_csv('../GNSS/Subtrop/HK/HK_node_delay_vert_fixed_hgtlvs.csv')
dat = dat.dropna()
dat = dat[dat['sigZTD'] < 0.1]
# lon_min, lat_min, lon_max, lat_max = -155.9, 18.9, -154.9, 19.9

# hgtlvs = [-100, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400,
#           2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000,
#           5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000,
#           14000, 15000, 16000, 17000, 18000, 19000, 20000, 25000, 30000, 35000, 40000]

# Load Model
Norm_model = tf.keras.models.load_model('../ML/Model/Full_US_PTE_fixed_hgtlvs_model')
Multi_model = tf.keras.models.load_model('../ML/Multiple_Input_Model/Model'
                                         '/Test_New_model3_US_PTE_fixed_hgtlvs_cloud_model')
wet_hydro_model = tf.keras.models.load_model('../ML/Wet_hydro_model/Model/wet_hydro_US_PTE_fixed_hgtlvs_model')

# Load scaler
scaler_x = load('../ML/Scaler/US_WE_noGOES_MinMax_scaler_x.bin')
scaler_y = load('../ML/Scaler/US_WE_noGOES_MinMax_scaler_y.bin')
scalerP = load('../ML/Multiple_Input_Model/Scaler/Test_New_model3_pScaler_x.bin')
scalerT = load('../ML/Multiple_Input_Model/Scaler/Test_New_model3_tScaler_x.bin')
scalerE = load('../ML/Multiple_Input_Model/Scaler/Test_New_model3_eScaler_x.bin')
scaler_y1 = load('../ML/Multiple_Input_Model/Scaler/Test_New_model3_scaler_y.bin')
wet_scaler_x = load('../ML/Wet_hydro_model/Scaler/wet_hydro_model_scaler_x.bin')
wet_scaler_y = load('../ML/Wet_hydro_model/Scaler/wet_hydro_model_scaler_y.bin')

# Obtain the input variables
X = df[df.columns[pd.Series(df.columns).str.startswith(('Lat', 'Hgt_m', 'P_', 'T_', 'e_'))]]
P = df[df.columns[pd.Series(df.columns).str.startswith(('Lat', 'Hgt_m', 'P_'))]]
T = df[df.columns[pd.Series(df.columns).str.startswith(('Lat', 'Hgt_m', 'T_'))]]
E = df[df.columns[pd.Series(df.columns).str.startswith(('Lat', 'Hgt_m', 'e_'))]]
wet = dat[dat.columns[pd.Series(dat.columns).str.startswith(('Lat', 'Hgt_m', 'total_'))]]

# Predict
predict1 = scaler_y.inverse_transform(Norm_model.predict(scaler_x.transform(X)))
predict2 = scaler_y1.inverse_transform(
    Multi_model.predict([scalerP.transform(P), scalerT.transform(T), scalerE.transform(E)]))
predict3 = wet_scaler_y.inverse_transform(wet_hydro_model.predict(wet_scaler_x.transform(wet)))
true1 = df[['ZTD']].values
true2 = dat[['ZTD']].values

print('Normal_model:')
print('Predict: ', predict1[:5].ravel())
print('True: ', true1[:5].ravel())
print('')
print('Multi_input model:')
print('Predict: ', predict2[:5].ravel())
print('True: ', true1[:5].ravel())
print('')
print('Wet hydro model:')
print('Predict: ', predict3[:5].ravel())
print('True: ', true2[:5].ravel())
print('')

from sklearn.metrics import mean_squared_error, r2_score

print('')
print("Normal model")
# The mean squared error
print('Mean squared error: %.10f' % mean_squared_error(true1, predict1))
# The R2 score
print('R2: %.5f' % r2_score(true1, predict1))
# The RMSE
rmse = np.sqrt(mean_squared_error(true1, predict1))
print('RMSE: %.5f' % rmse)
errors = predict1 - true1
print('Average error: %.5f' % np.mean(abs(errors)))
print('')

print("Multi-input model")
# The mean squared error
print('Mean squared error: %.10f' % mean_squared_error(true1, predict2))
# The R2 score
print('R2: %.5f' % r2_score(true1, predict2))
# The RMSE
rmse = np.sqrt(mean_squared_error(true1, predict2))
print('RMSE: %.5f' % rmse)
errors = predict2 - true1
print('Average error: %.5f' % np.mean(abs(errors)))
print('')

print("Wet Hydro model")
# The mean squared error
print('Mean squared error: %.10f' % mean_squared_error(true2, predict3))
# The R2 score
print('R2: %.5f' % r2_score(true2, predict3))
# The RMSE
rmse = np.sqrt(mean_squared_error(true2, predict3))
print('RMSE: %.5f' % rmse)
errors = predict3 - true2
print('Average error: %.5f' % np.mean(abs(errors)))
print('')

# Plot of Observation vs Prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true1, predict1, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Observed', fontsize=10)
plt.ylabel('Predicted', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.suptitle('Normal model obs vs pred')
fig.savefig('Plots/HK/US_WE_noGOES_model_Ob_v_Pred.png', dpi=300)

# Plot of Observation vs Prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true1, predict2, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Observed', fontsize=10)
plt.ylabel('Predicted', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.suptitle('Multi-input model obs vs pred')
fig.savefig('Plots/HK/US_WE_noGOES_model_MIP_Ob_v_Pred.png', dpi=300)

# Plot of Observation vs Prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true2, predict3, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Observed', fontsize=10)
plt.ylabel('Predicted', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.suptitle('Wet Hydro model obs vs pred')
fig.savefig('Plots/HK/wet_hydro_model_Ob_v_Pred.png', dpi=300)

# Plot of residual of the prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true1, true1 - predict1, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('True', fontsize=10)
plt.ylabel('Residual', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.suptitle('Normal model Residual')
fig.savefig('Plots/HK/US_WE_noGOES_model_Resid_true.png', dpi=300)

# Plot of residual of the prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true1, true1 - predict2, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('True', fontsize=10)
plt.ylabel('Residual', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.suptitle('Multi-input model Residual')
fig.savefig('Plots/HK/US_WE_noGOES_model_MIP_Resid_true.png', dpi=300)

# Plot of residual of the prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true2, true2 - predict3, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('True', fontsize=10)
plt.ylabel('Residual', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.suptitle('Wet Hydro model Residual')
fig.savefig('Plots/HK/wet_hydro_model_Resid_true.png', dpi=300)

print('')
# G-matrix comparison
G = np.stack((predict1.ravel(), predict2.ravel(), np.ones_like(predict1.ravel())), axis=1)
print(G[:5, :])
mhat, res, rank, s = np.linalg.lstsq(G, true1)
print(mhat, res, rank, s)
y_pred = np.dot(G, mhat)

print("G-matrix")
# The mean squared error
print('Mean squared error: %.10f' % mean_squared_error(true1, y_pred))
# The R2 score
print('R2: %.5f' % r2_score(true1, y_pred))
# The RMSE
rmse = np.sqrt(mean_squared_error(true1, y_pred))
print('RMSE: %.5f' % rmse)
errors = y_pred - true1
print('Average error: %.5f' % np.mean(abs(errors)))

# Plot of residual of the prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true1, true1 - y_pred, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('True', fontsize=10)
plt.ylabel('Residual', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.suptitle('G-matrix Residual')
fig.savefig('Plots/HK/G-matrix_Resid_true.png', dpi=300)

# Plot of Observation vs Prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true1, y_pred, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Observed', fontsize=10)
plt.ylabel('Predicted', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.suptitle('G-matrix obs vs pred')
fig.savefig('Plots/HK/G-matrix_Ob_v_Pred.png', dpi=300)
