import sys
import os

import pandas as pd

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
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
GNSS = pd.read_feather('../../GNSS_US/US/US_Inter_PTE_vert_fixed_hgtlvs.ftr')
GNSS = GNSS.dropna()
info = GNSS.iloc[:, GNSS.columns.str.startswith(('ID', 'start_date', 'end_date', 'int_ZTD', 'Lon', 'Lat',
                                                 'Hgt_m', 'Slope'))]
date1_P_T = pd.DataFrame(GNSS.iloc[:, GNSS.columns.str.startswith('date1_P_')].values /
                         GNSS.iloc[:, GNSS.columns.str.startswith('date1_T_')].values)
date1_P_T.columns = ['date1_P_T_' + str(i) for i in range(1, 51)]
date2_P_T = pd.DataFrame(GNSS.iloc[:, GNSS.columns.str.startswith('date2_P_')].values /
                         GNSS.iloc[:, GNSS.columns.str.startswith('date2_T_')].values)
date2_P_T.columns = ['date2_P_T_' + str(i) for i in range(1, 51)]
date1_e = GNSS.iloc[:, GNSS.columns.str.startswith('date1_e_')]
date2_e = GNSS.iloc[:, GNSS.columns.str.startswith('date2_e_')]
GNSS = pd.concat((info, date1_P_T, date1_e, date2_P_T, date2_e), axis=1)

train = GNSS[GNSS['start_date'] < '2021-01-01']
test = GNSS[GNSS['start_date'] > '2020-12-31']

train_x = train[
    train.columns[pd.Series(train.columns).str.startswith(('Lon', 'Lat', 'Hgt_m', 'date1_', 'date2_', 'Slope'))]]
train_y = train[['int_ZTD']]
print(train_x.head())
test_x = test[
    test.columns[pd.Series(test.columns).str.startswith(('Lon', 'Lat', 'Hgt_m', 'date1_', 'date2_', 'Slope'))]]
test_y = test[['int_ZTD']]

from sklearn.model_selection import train_test_split

train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=40)

train_x, scaler_x = standardized(train_x, 'MinMax')
test_x = scaler_x.transform(test_x)
valid_x = scaler_x.transform(valid_x)
train_y, scaler_y = standardized(train_y, 'MinMax')
valid_y = scaler_y.transform(valid_y)
from joblib import dump

dump(scaler_x, 'Scaler/GPS_int_PTE_slope_model_MinMax_scaler_x.bin', compress=True)
dump(scaler_y, 'Scaler/GPS_int_PTE_slope_model_MinMax_scaler_y.bin', compress=True)

es = EarlyStopping(verbose=1, patience=10)

# Initialiizinig the ANN
model = tf.keras.models.Sequential()
# Input layer
model.add(tf.keras.layers.Input(shape=(204,)))
# Adding first hidden layer
model.add(tf.keras.layers.Dense(units=104, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding first hidden layer
model.add(tf.keras.layers.Dense(units=104, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=54, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=54, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding the output layer
model.add(tf.keras.layers.Dense(units=1, activation='linear'))
# Compiling the ANN
model.compile(optimizer='adam', loss=['MSE'], metrics=['MAE'])
# Print model summary
print(model.summary())
# Train the ANN on the Training set
model.fit(train_x, train_y, batch_size=512, epochs=200, validation_data=[valid_x, valid_y], callbacks=[es], verbose=0)

# Plot history: MSE
plt.plot(model.history.history['loss'], label='MSE (training data)')
plt.plot(model.history.history['val_loss'], label='MSE (validation data)')
plt.title('MSE for noise prediction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/GPS_int_PTE_slope_model_MSE_history.png', dpi=300)
plt.clf()

# Plot history: MAE
plt.plot(model.history.history['MAE'], label='MAE (training data)')
plt.plot(model.history.history['val_MAE'], label='MAE (validation data)')
plt.title('MAE for noise prediction')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/GPS_int_PTE_slope_model_MAE_history.png', dpi=300)

# Saving model
model.save('Model/GPS_int_PTE_slope_model')

# Predict different model
predict_true = scaler_y.inverse_transform(model.predict(train_x))
predict = scaler_y.inverse_transform(model.predict(test_x))
# predict = model.predict(x_test)
true = test_y.values
true_true = scaler_y.inverse_transform(train_y)

print_metric(true, predict, 'Test_GPS_int_PTE_slope_model')
model_name = 'Test_GPS_int_PTE_slope_model'
save_loc = 'Plots/New_inputs'

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

print_metric(true_true, predict_true, 'True_GPS_int_PTE_slope_model')
model_name = 'True_GPS_int_PTE_slope_model'
save_loc = 'Plots/New_inputs'

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true_true, predict_true, cmap=white_viridis)
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
density = ax.scatter_density(true_true, true_true - predict_true, cmap=white_viridis)
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
print('Finished Training')
