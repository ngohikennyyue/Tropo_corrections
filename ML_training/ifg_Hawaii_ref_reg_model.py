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

num_threads = 20
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["TF_NUM_INTRAOP_THREADS"] = "20"
os.environ["TF_NUM_INTEROP_THREADS"] = "20"

tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)

# Read in data
dat = pd.read_csv('../../../InSAR/Large_scale/Hawaii/Hawaii_train_ref_ifg_PTE_fixed_hgtlvs.csv')
dat = dat.dropna()
test_dat = pd.read_csv('../../../InSAR/Large_scale/Hawaii/Hawaii_test_ref_ifg_PTE_fixed_hgtlvs.csv')
test_dat = test_dat.dropna()

X = dat[dat.columns[pd.Series(dat.columns).str.startswith(('Lon', 'Lat', 'Hgt_m', 'date1_', 'date2_', 'slope'))]]
y = dat[['ifg']]
# X, y = shuffle(X, y)
print(X.head())
x_test = test_dat[test_dat.columns[pd.Series(test_dat.columns).str.startswith(('Lon', 'Lat', 'Hgt_m', 'date1_', 'date2_',
                                                                               'slope'))]]
y_test = test_dat[['ifg']]

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

x_train, scaler_x = standardized(x_train, 'MinMax')
x_test = scaler_x.transform(x_test)
x_valid = scaler_x.transform(x_valid)
y_train, scaler_y = standardized(y_train, 'MinMax')
y_valid = scaler_y.transform(y_valid)

from joblib import dump

dump(scaler_x, 'Scaler/ifg_Hawaii_ref_reg_model_MinMax_scaler_x.bin', compress=True)
dump(scaler_y, 'Scaler/ifg_Hawaii_ref_reg_model_MinMax_scaler_y.bin', compress=True)

batchsize = [512, 256, 128, 64]
for batch in batchsize:
    es = EarlyStopping(verbose=1, patience=10)
    # Initialiizinig the ANN
    model = tf.keras.models.Sequential()
    # Input layer
    model.add(tf.keras.layers.Input(shape=(310,)))
    # Adding first hidden layer
    model.add(tf.keras.layers.Dense(units=106, activation=PReLU(), kernel_regularizer=tf.keras.regularizers.l1(0.01)))
    # Adding hidden layer
    model.add(tf.keras.layers.Dense(units=106, activation=PReLU(), kernel_regularizer=tf.keras.regularizers.l1(0.01)))
    # Adding hidden layer
    model.add(tf.keras.layers.Dense(units=55, activation=PReLU(), kernel_regularizer=tf.keras.regularizers.l1(0.01)))
    # Adding hidden layer
    model.add(tf.keras.layers.Dense(units=55, activation=PReLU(), kernel_regularizer=tf.keras.regularizers.l1(0.01)))
    # Adding the output layer
    model.add(tf.keras.layers.Dense(units=1, activation=PReLU()))
    # Compiling the ANN
    model.compile(optimizer='adam', loss=['MSE'], metrics=['MAE'])
    # Print model summary
    print(model.summary())
    # Train the ANN on the Training set
    model.fit(x_train, y_train, batch_size=batch, epochs=200, validation_data=[x_valid, y_valid], callbacks=[es],
              verbose=0)

    # Plot history: MSE
    plt.plot(model.history.history['loss'], label='MSE (training data)')
    plt.plot(model.history.history['val_loss'], label='MSE (validation data)')
    plt.title('MSE')
    plt.ylabel('MSE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.savefig('Plots/reg_model/ifg_Hawaii_ref_model_batchsize_{}_MSE_history.png'.format(batch), dpi=300)
    plt.clf()

    # Plot history: MAE
    plt.plot(model.history.history['MAE'], label='MAE (training data)')
    plt.plot(model.history.history['val_MAE'], label='MAE (validation data)')
    plt.title('MAE')
    plt.ylabel('MAE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.savefig('Plots/reg_model/ifg_Hawaii_ref_model_batchsize_{}_MAE_history.png'.format(batch), dpi=300)

    # Saving model
    model.save('Model/ifg_Hawaii_ref_reg_model_batchsize_{}'.format(batch))

    # Predict different model
    predict_true = scaler_y.inverse_transform(model.predict(x_train))
    predict_test = scaler_y.inverse_transform(model.predict(x_test))
    # predict = model.predict(x_test)
    true_test = y_test.values
    true_true = scaler_y.inverse_transform(y_train)

    print_metric(true_true, predict_true, 'True_ifg_Hawaii_ref_reg_model_batchsize_{}'.format(batch))

    model = 'True_ifg_Hawaii_ref_reg_model_batchsize_{}'.format(batch)
    save_loc = 'Plots/reg_model'
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

    print_metric(true_test, predict_test, 'Test_ifg_Hawaii_ref_reg_model_batchsize_{}'.format(batch))
    model = 'Test_ifg_Hawaii_ref_reg_model_batchsize_{}'.format(batch)
    save_loc = 'Plots/reg_model'
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
