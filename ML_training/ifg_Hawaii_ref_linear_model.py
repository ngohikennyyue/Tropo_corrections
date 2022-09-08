from sklearn.linear_model import LinearRegression
import sys
import os

import pandas as pd
current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
parent = os.path.dirname(parent)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *

# Read in data
dat = pd.read_csv('../../../InSAR/Large_scale/Hawaii/Hawaii_train_ref_ifg_PTE_fixed_hgtlvs.csv')
dat = dat.dropna()
test_dat = pd.read_csv('../../../InSAR/Large_scale/Hawaii/Hawaii_test_ref_ifg_PTE_fixed_hgtlvs.csv')
test_dat = test_dat.dropna()

X = dat[dat.columns[pd.Series(dat.columns).str.startswith(('Lat', 'Hgt_m', 'date1_', 'date2_', 'slope'))]]
y = dat[['ifg']]
# X, y = shuffle(X, y)
print(X.head())
x_test = test_dat[test_dat.columns[pd.Series(test_dat.columns).str.startswith(('Lat', 'Hgt_m', 'date1_', 'date2_',
                                                                               'slope'))]]
y_test = test_dat[['ifg']]

# MinMax Scaler
x_train, scaler_x = standardized(X, 'MinMax')
x_test = scaler_x.transform(x_test)
y_train, scaler_y = standardized(y, 'MinMax')
y_test = scaler_y.transform(y_test)

from joblib import dump, load
dump(scaler_x, 'Scaler/ifg_Hawaii_ref_linear_model_MinMax_scaler_x.bin', compress=True)
dump(scaler_y, 'Scaler/ifg_Hawaii_ref_linear_model_MinMax_scaler_y.bin', compress=True)

regr = LinearRegression()

regr.fit(x_train, y_train)
print('Test score (Test):', regr.score(x_test, y_test))

from sklearn.metrics import mean_absolute_error, mean_squared_error
predict = scaler_y.inverse_transform(regr.predict(x_test))
true = scaler_y.inverse_transform(y_test)
predict_true = scaler_y.inverse_transform(regr.predict(x_train))
true_true = scaler_y.inverse_transform(y_train)

print_metric(true, predict, 'Test')
print_metric(true_true, predict_true, 'Train')
print('Coef: ', regr.coef_)

plot_graphs(true, predict, 'Linear_test', 'Plots/linear_model')
plot_graphs(true_true, predict_true, 'Linear_true', 'Plots/linear_model')