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
train = GNSS[GNSS['start_date'] < '2021-01-01']
test = GNSS[GNSS['start_date'] > '2020-12-31']

train_x = train[train.columns[pd.Series(train.columns).str.startswith(('Lon', 'Lat', 'Hgt_m', 'date1_', 'date2_', 'Slope'))]]
train_y = train[['int_ZTD']]
print(train_x.head())
test_x = test[test.columns[pd.Series(test.columns).str.startswith(('Lon', 'Lat', 'Hgt_m', 'date1_', 'date2_', 'Slope'))]]
test_y = test[['int_ZTD']]

from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

train_x, scaler_x = standardized(train_x, 'MinMax')
test_x = scaler_x.transform(test_x)
train_y, scaler_y = standardized(train_y, 'MinMax')
y_test = scaler_y.transform(test_y)
from joblib import dump

dump(scaler_x, 'Scaler/GPS_int_PTE_slope_model_MinMax_scaler_x.bin', compress=True)
dump(scaler_y, 'Scaler/GPS_int_PTE_slope_model_MinMax_scaler_y.bin', compress=True)

es = EarlyStopping(verbose=1, patience=8)

# Initialiizinig the ANN
model = tf.keras.models.Sequential()
# Input layer
model.add(tf.keras.layers.Input(shape=(304,)))
# Adding first hidden layer
model.add(tf.keras.layers.Dense(units=150, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding first hidden layer
model.add(tf.keras.layers.Dense(units=70, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=30, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=15, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding the output layer
model.add(tf.keras.layers.Dense(units=1, activation='linear'))
# Compiling the ANN
model.compile(optimizer='adam', loss=['MSE'], metrics=['MAE'])
# Print model summary
print(model.summary())
# Train the ANN on the Training set
model.fit(train_x, train_y, batch_size=6000, epochs=200, validation_data=[test_x, y_test], callbacks=[es], verbose=0)

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
predict = scaler_y.inverse_transform(model.predict(test_x))
# predict = model.predict(x_test)
true = test_y.values

print_metric(true, predict, 'GPS_int_PTE_slope_model')
plot_graphs(true, predict, 'GPS_int_PTE_slope_model', 'Plots')

print('Finished Training')
