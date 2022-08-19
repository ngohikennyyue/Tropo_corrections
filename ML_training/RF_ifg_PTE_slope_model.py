# Importing the libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor
current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
sys.path.append(parent)
from sklearn.utils import shuffle
from extract_func.Extract_PTE_function import *

# Read in data
dat = pd.read_feather('../../InSAR/Large_scale/ML_data/train_data.ftr')
dat = dat.dropna()
test_dat = pd.read_feather('../../InSAR/Large_scale/ML_data/test_data.ftr')
test_dat = test_dat.dropna()

X = dat[dat.columns[pd.Series(dat.columns).str.startswith(('Lat', 'Hgt_m', 'date1_', 'date2_', 'slope'))]]
y = dat[['ifg']]
X, y = shuffle(X, y)

test_X = test_dat[test_dat.columns[pd.Series(test_dat.columns).str.startswith(('Lat', 'Hgt_m', 'date1_', 'date2_',
                                                                               'slope'))]]
test_y = test_dat[['ifg']]

X, scaler_x = standardized(X, 'MinMax')
x_test = scaler_x.transform(test_X)
y, scaler_y = standardized(y, 'MinMax')
y_test = test_y

# create regressor object
regressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)

# fit the regressor with x and y data
regressor.fit(X, y.ravel())

predict = regressor.predict(x_test)
true = y_test.values

from sklearn.metrics import mean_squared_error, r2_score

print("ANN model")
# The mean squared error
print('Mean squared error: %.10f' % mean_squared_error(true, predict))

# The R2 score
print('R2: %.5f' % r2_score(true, predict))

# The RMSE
rmse = np.sqrt(mean_squared_error(true, predict))
print('RMSE: %.5f' % rmse)

errors = predict - true
print('Average errror: %.5f' % np.mean(abs(errors)))

# Plot of Observation vs Prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true, predict, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Observed', fontsize=10)
plt.ylabel('Predicted', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.savefig('Plots/RF_ifg_PTE_slope_model_Ob_v_Pred.png', dpi=300)

# Plot of residual of the prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true, true - predict, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('True', fontsize=10)
plt.ylabel('Residual', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.savefig('Plots/RF_ifg_PTE_slope_model_Resid_true.png', dpi=300)

print('Finished Training')