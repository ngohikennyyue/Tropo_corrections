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

# lon_min, lat_min, lon_max, lat_max = -155.9, 18.9, -154.9, 19.9

# hgtlvs = [-100, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400,
#           2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000,
#           5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000,
#           14000, 15000, 16000, 17000, 18000, 19000, 20000, 25000, 30000, 35000, 40000]
# Load Model
Norm_model = tf.keras.models.load_model('../ML/Model/Full_US_WE_PTE_fixed_hgtlvs_model')
Multi_model = tf.keras.models.load_model('../ML/Multiple_Input_Model/Model/Test_New_model3_US_PTE_fixed_hgtlvs_cloud_model')
# Load scaler
scaler_x = load('../ML/Scaler/US_WE_noGOES_MinMax_scaler_x.bin')
scaler_y = load('../ML/Scaler/US_WE_noGOES_MinMax_scaler_y.bin')
scalerP = load('../ML/Multiple_Input_Model/Scaler/Test_New_model3_pScaler_x.bin')
scalerT = load('../ML/Multiple_Input_Model/Scaler/Test_New_model3_tScaler_x.bin')
scalerE = load('../ML/Multiple_Input_Model/Scaler/Test_New_model3_eScaler_x.bin')
scaler_y1 = load('../ML/Multiple_Input_Model/Scaler/Test_New_model3_scaler_y.bin')

# Obtain the input variables
X = df[df.columns[pd.Series(df.columns).str.startswith(('Lat', 'Hgt_m', 'P_', 'T_', 'e_'))]]
P = df[df.columns[pd.Series(df.columns).str.startswith(('Lat', 'Hgt_m', 'P_'))]]
T = df[df.columns[pd.Series(df.columns).str.startswith(('Lat', 'Hgt_m', 'T_'))]]
E = df[df.columns[pd.Series(df.columns).str.startswith(('Lat', 'Hgt_m', 'e_'))]]

# Predict
predict1 = scaler_y.inverse_transform(Norm_model.predict(scaler_x.transform(X)))
true = df[['ZTD']].values
predict2 = scaler_y1.inverse_transform(Multi_model.predict([scalerP.transform(P), scalerT.transform(T), scalerE.transform(E)]))
print('')
print('Normal_model:')
print('Predict: ', predict1[:5].ravel())
print('True: ', true[:5].ravel())
print('')
print('Multi_input model:')
print('Predict: ', predict2[:5].ravel())
print('True: ', true[:5].ravel())
from sklearn.metrics import mean_squared_error, r2_score
print('')
print("Normal model")
# The mean squared error
print('Mean squared error: %.10f' % mean_squared_error(true, predict1))
# The R2 score
print('R2: %.5f' % r2_score(true, predict1))
# The RMSE
rmse = np.sqrt(mean_squared_error(true, predict1))
print('RMSE: %.5f' % rmse)
errors = predict1 - true
print('Average error: %.5f' % np.mean(abs(errors)))
print('')

print("Multi-input model")
# The mean squared error
print('Mean squared error: %.10f' % mean_squared_error(true, predict2))
# The R2 score
print('R2: %.5f' % r2_score(true, predict2))
# The RMSE
rmse = np.sqrt(mean_squared_error(true, predict2))
print('RMSE: %.5f' % rmse)
errors = predict2 - true
print('Average error: %.5f' % np.mean(abs(errors)))
print('')
# Plot of Observation vs Prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true, predict1, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Observed', fontsize=10)
plt.ylabel('Predicted', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.suptitle('Normal model obs vs pred')
fig.savefig('Plots/Mexico/US_WE_noGOES_model_Ob_v_Pred.png', dpi=300)

# Plot of Observation vs Prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true, predict2, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Observed', fontsize=10)
plt.ylabel('Predicted', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.suptitle('Multi-input model obs vs pred')
fig.savefig('Plots/Mexico/US_WE_noGOES_model_MIP_Ob_v_Pred.png', dpi=300)

# Plot of residual of the prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true, true - predict1, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('True', fontsize=10)
plt.ylabel('Residual', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.suptitle('Normal model Residual')
fig.savefig('Plots/Mexico/US_WE_noGOES_model_Resid_true.png', dpi=300)

# Plot of residual of the prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true, true - predict2, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('True', fontsize=10)
plt.ylabel('Residual', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.suptitle('Multi-input model Residual')
fig.savefig('Plots/Mexico/US_WE_noGOES_model_MIP_Resid_true.png', dpi=300)

# G-matrix comparison
G = np.stack((predict1.ravel(), predict2.ravel(), np.ones_like(predict1.ravel())), axis=1)
print(G[:5, :])
mhat, res, rank, s = np.linalg.lstsq(G, true)
print(mhat, res, rank, s)
y_pred = np.dot(G, mhat)

print('')
print("G-matrix")
# The mean squared error
print('Mean squared error: %.10f' % mean_squared_error(true, y_pred))
# The R2 score
print('R2: %.5f' % r2_score(true, y_pred))
# The RMSE
rmse = np.sqrt(mean_squared_error(true, y_pred))
print('RMSE: %.5f' % rmse)
errors = y_pred - true
print('Average error: %.5f' % np.mean(abs(errors)))
print('')

# Plot of residual of the prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true, true - y_pred, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('True', fontsize=10)
plt.ylabel('Residual', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.suptitle('G-matrix Residual')
fig.savefig('Plots/Mexico/G-matrix_Resid_true.png', dpi=300)

# Plot of Observation vs Prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true, y_pred, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Observed', fontsize=10)
plt.ylabel('Predicted', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.suptitle('G-matrix obs vs pred')
fig.savefig('Plots/Mexico/G-matrix_Ob_v_Pred.png', dpi=300)