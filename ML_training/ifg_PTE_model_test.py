import sys
import os

import pandas as pd

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *
from sklearn.utils import shuffle

# num_threads = 20
# os.environ["OMP_NUM_THREADS"] = "20"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "20"
# os.environ["TF_NUM_INTEROP_THREADS"] = "20"
#
# tf.config.threading.set_inter_op_parallelism_threads(num_threads)
# tf.config.threading.set_intra_op_parallelism_threads(num_threads)
# tf.config.set_soft_device_placement(True)

# Read in data
dat = pd.read_feather('../../InSAR/Large_scale/ML_data/train_data.ftr')
dat = dat.dropna()
test_dat = pd.read_feather('../../InSAR/Large_scale/ML_data/test_data.ftr')
test_dat = test_dat.dropna()

X = dat[dat.columns[pd.Series(dat.columns).str.startswith(('Lat', 'Hgt_m', 'date1_', 'date2_', 'slope'))]]
y = dat[['ifg']]
# X, y = shuffle(X, y)
print(X.head())
test_X = test_dat[test_dat.columns[pd.Series(test_dat.columns).str.startswith(('Lat', 'Hgt_m', 'date1_', 'date2_',
                                                                               'slope'))]]
test_y = test_dat[['ifg']]

from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)


from joblib import dump, load

scaler_x = load('Scaler/ifg_PTE_slope_full_model_MinMax_scaler_x.bin')
scaler_y = load('Scaler/ifg_PTE_slope_full_model_MinMax_scaler_y.bin')

X = scaler_x.transform(X)

# Saving model
full_model = tf.keras.models.load_model('Model/ifg_PTE_full_model')
small_model = tf.keras.models.load_model('Model/ifg_PTE_full_small_model')

predict_full_model = scaler_y.inverse_transform(full_model.predict(X))
predict_small_model = scaler_y.inverse_transform(small_model.predict(X))
predict_full_model_test = scaler_y.inverse_transform(full_model.predict(scaler_x.transform(test_X)))
predict_small_model_test = scaler_y.inverse_transform(small_model.predict(scaler_x.transform(test_X)))

true = y.values
true_test = test_y.values

print_metric(true, predict_full_model, 'Full_model')
plot_graphs(true, predict_full_model, 'Full_model', 'Plots/Test_model/')
print_metric(true_test, predict_full_model_test, 'Full_model_test_data')
plot_graphs(true_test, predict_full_model_test, 'Full_model_test_data', 'Plots/Test_model/')
print_metric(true, predict_small_model, 'Small_model')
plot_graphs(true, predict_small_model, 'Small_model', 'Plots/Test_model/')
print_metric(true_test, predict_small_model_test, 'Small_model_test_data')
plot_graphs(true_test, predict_small_model_test, 'Small_model_test_data', 'Plots/Test_model/')

print('Finished Training')
