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
parent = os.path.dirname(parent)
parent = os.path.dirname(parent)
sys.path.append(parent)
from sklearn.utils import shuffle
from extract_func.Extract_PTE_function import *

# Read in data
# Read in data
dat = pd.read_csv('../../../InSAR/Large_scale/Hawaii/Hawaii_train_ref_ifg_PTE_fixed_hgtlvs.csv')
dat = dat.dropna()
test_dat = pd.read_csv('../../../InSAR/Large_scale/Hawaii/Hawaii_test_ref_ifg_PTE_fixed_hgtlvs.csv')
test_dat = test_dat.dropna()

X = dat[dat.columns[pd.Series(dat.columns).str.startswith(('Lat', 'Hgt_m', 'date1_', 'date2_', 'slope'))]]
y = dat['ifg'].values
# X, y = shuffle(X, y)

test_X = test_dat[test_dat.columns[pd.Series(test_dat.columns).str.startswith(('Lat', 'Hgt_m', 'date1_', 'date2_',
                                                                               'slope'))]]
test_y = test_dat['ifg'].values

# X, scaler_x = standardized(X, 'MinMax')
# x_test = scaler_x.transform(test_X)
# y, scaler_y = standardized(y, 'MinMax')
# y_test = test_y
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000, random_state=1, n_jobs=-1)
# fit the regressor with x and y data
forest.fit(X, y)
y_train_pred = forest.predict(X)
y_test_pred = forest.predict(test_X)

from sklearn.metrics import mean_squared_error, r2_score
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y, y_train_pred),
                                       mean_squared_error(test_y, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y, y_train_pred),
                                       r2_score(test_y, y_test_pred)))
forest.save('Model/RF_ifg_HI_model')

plt.figure(figsize=(10, 8))
plt.scatter(y_train_pred, y_train_pred - y,
            c='grey', marker='o',
            s=35, alpha=0.65, edgecolor='k', label='Training data')
plt.scatter(y_test_pred, y_test_pred - test_y,
            c='lightgreen', marker='s',
            s=35, alpha=0.7, edgecolor='k', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-0.2, xmax=0.2, lw=2, color='red')
plt.xlim([-0.2, 0.2])
plt.savefig('Plots/RF_ifg_HI_residual.png', dpi=300)