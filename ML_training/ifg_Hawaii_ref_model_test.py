import sys
import os

import pandas as pd

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
parent = os.path.dirname(parent)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *
from sklearn.utils import shuffle


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


from joblib import load

scaler_x = load('Scaler/ifg_Hawaii_ref_model_MinMax_scaler_x.bin')
scaler_y = load('Scaler/ifg_Hawaii_ref_model_MinMax_scaler_y.bin')

batchsize = [64, 128, 256, 512]
for batch in batchsize:
    # Load model
    model = tf.keras.models.load_model('Model/ifg_Hawaii_ref_model_batchsize_{}'.format(batch))

    # Predict different model
    predict_true = scaler_y.inverse_transform(model.predict(scaler_x.transform(X)))
    predict_test = scaler_y.inverse_transform(model.predict(scaler_x.transform(x_test)))
    true_true = y.values
    true_test = y_test.values

    print_metric(true_true, predict_true, 'True_ifg_Hawaii_ref_model_batchsize_{}'.format(batch))
    plot_graphs(true_true, predict_true, 'True_ifg_Hawaii_ref_model_batchsize_{}'.format(batch), 'Plots/ref_model')
    print_metric(true_test, predict_test, 'Test_ifg_Hawaii_ref_model_batchsize_{}'.format(batch))
    plot_graphs(true_test, predict_test, 'Test_ifg_Hawaii_ref_model_batchsize_{}'.format(batch), 'Plots/ref_model')

print('Finished Training')
