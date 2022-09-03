import sys
import os

import matplotlib.pyplot as plt
import pandas as pd

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *
from sklearn.utils import shuffle

# Read in data
GNSS = pd.read_feather('../../GNSS_US/US/US_Inter_PTE_vert_fixed_hgtlvs.ftr')
GNSS = GNSS.dropna()
# test = GNSS[GNSS['start_date'] > '2020-12-31']
test = GNSS[(GNSS['int_ZTD'] < 0.5) & (GNSS['int_ZTD'] > -0.5)]

test_x = test[test.columns[pd.Series(test.columns).str.startswith(('Lat', 'Hgt_m', 'date1_', 'date2_', 'Slope'))]]
test_y = test[['int_ZTD']].values

from joblib import load

scaler_x = load('Scaler/GPS_int_PTE_slope_model_MinMax_scaler_x.bin')
scaler_y = load('Scaler/GPS_int_PTE_slope_model_MinMax_scaler_y.bin')

model = tf.keras.models.load_model('Model/GPS_int_PTE_slope_model')

# Predict different model
predict = scaler_y.inverse_transform(model.predict(scaler_x.transform(test_x)))
# predict = model.predict(x_test)
true = test_y
print(predict.shape)
print(true.shape)

print_metric(true, predict, 'GPS_int_PTE_slope_model')

model_name = 'GPS_int_PTE_slope_model'
save_loc = 'Plots'

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true, predict, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Observed', fontsize=10)
plt.ylabel('Predicted', fontsize=10)
plt.ylim(-0.2, 0.2)
cbar.ax.tick_params(labelsize=10)
fig.suptitle(model_name + ' obs vs pred')
fig.savefig(save_loc + '/' + model_name + '_Ob_v_Pred.png', dpi=300)
plt.clf()

# Plot of residual of the prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true, true - predict, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('True', fontsize=10)
plt.ylabel('Residual', fontsize=10)
plt.ylim(-0.1, 0.1)
cbar.ax.tick_params(labelsize=10)
fig.suptitle(model_name + ' Residual')
fig.savefig(save_loc + '/' + model_name + '_Resid_true.png', dpi=300)
plt.clf()
