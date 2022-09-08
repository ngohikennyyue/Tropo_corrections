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

X = dat.iloc[:, dat.columns.str.startswith(('Lon', 'Lat', 'Hgt_m', 'date1_', 'date2_', 'slope'))]
y = dat[['ifg']]
# X, y = shuffle(X, y)
print(X.head())
x_test = test_dat.iloc[:, test_dat.columns.str.startswith(('Lon', 'Lat', 'Hgt_m', 'date1_', 'date2_', 'slope'))]
y_test = test_dat[['ifg']]


from joblib import load

scaler_x = load('Scaler/ifg_Hawaii_ref_model_MinMax_scaler_x.bin')
scaler_y = load('Scaler/ifg_Hawaii_ref_model_MinMax_scaler_y.bin')

batchsize = [512]
for batch in batchsize:
    # Load model
    model = tf.keras.models.load_model('Model/ifg_Hawaii_ref_model_batchsize_{}'.format(batch))

    # Predict different model
    predict_true = scaler_y.inverse_transform(model.predict(scaler_x.transform(X)))
    predict_test = scaler_y.inverse_transform(model.predict(scaler_x.transform(x_test)))
    true_true = y.values
    true_test = y_test.values

    print_metric(true_true, predict_true, 'True_ifg_Hawaii_ref_model_batchsize_{}'.format(batch))

    model = 'True_ifg_Hawaii_ref_model_batchsize_{}'.format(batch)
    save_loc = 'Plots/Train_Test_valid'
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(true_true, predict_true, cmap=white_viridis)
    cbar = fig.colorbar(density)
    cbar.set_label(label='Number of points per pixel', size=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('Observed', fontsize=10)
    plt.ylabel('Predicted', fontsize=10)
    plt.xlim([-.1, .1])
    plt.ylim([-.1, .1])
    plt.plot([-.1, .1], [-.1, .1], 'k-')
    cbar.ax.tick_params(labelsize=10)
    fig.suptitle(model + ' obs vs pred')
    fig.savefig(save_loc + '/' + model + '_Ob_v_Pred.png', dpi=300)
    plt.clf()

    # Plot of residual of the prediction
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(true_true, true_true - predict_true, cmap=white_viridis)
    cbar = fig.colorbar(density)
    cbar.set_label(label='Number of points per pixel', size=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('True', fontsize=10)
    plt.ylabel('Residual', fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    fig.suptitle(model + ' Residual')
    fig.savefig(save_loc + '/' + model + '_Resid_true.png', dpi=300)
    plt.clf()

    print_metric(true_test, predict_test, 'Test_ifg_Hawaii_ref_model_batchsize_{}'.format(batch))
    model = 'Test_ifg_Hawaii_ref_model_batchsize_{}'.format(batch)
    save_loc = 'Plots/Train_Test_valid'
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(true_test, predict_test, cmap=white_viridis)
    cbar = fig.colorbar(density)
    cbar.set_label(label='Number of points per pixel', size=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('Observed', fontsize=10)
    plt.ylabel('Predicted', fontsize=10)
    plt.xlim([-.1, .1])
    plt.ylim([-.1, .1])
    plt.plot([-.1, .1], [-.1, .1], 'k-')
    cbar.ax.tick_params(labelsize=10)
    fig.suptitle(model + ' obs vs pred')
    fig.savefig(save_loc + '/' + model + '_Ob_v_Pred.png', dpi=300)
    plt.clf()

    # Plot of residual of the prediction
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(true_test, true_test - predict_test, cmap=white_viridis)
    cbar = fig.colorbar(density)
    cbar.set_label(label='Number of points per pixel', size=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('True', fontsize=10)
    plt.ylabel('Residual', fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    fig.suptitle(model + ' Residual')
    fig.savefig(save_loc + '/' + model + '_Resid_true.png', dpi=300)
    plt.clf()
print('Finished Training')
