import sys
import os

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *
from tensorflow.keras.utils import plot_model

num_threads = 20
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["TF_NUM_INTRAOP_THREADS"] = "20"
os.environ["TF_NUM_INTEROP_THREADS"] = "20"

tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)

# Read in data
dat = pd.read_csv('../../GNSS_US/US/ZTD/US_GNSS_fixed_hgtlvs_GNSS_GOES.csv')
dat = dat.dropna()
dat = dat[dat['CMI_C01'] > 0]
train = dat[dat['Date'] <= '2020-12-31'].dropna()
test = dat[dat['Date'] > '2020-12-31'].dropna()
train_x = train.iloc[:, train.columns.str.startswith(('Lat', 'Hgt_m', 'P_', 'T_', 'e_', 'CMI_C', 'GREEN'))]
train_y = train[['ZTD']]
test_x = test.iloc[:, test.columns.str.startswith(('Lat', 'Hgt_m', 'P_', 'T_', 'e_', 'CMI_C', 'GREEN'))]
test_y = test[['ZTD']]
# X = dat[dat.columns[pd.Series(dat.columns).str.startswith(('Lat', 'Hgt_m', 'P_', 'T_', 'e_', 'CMI_C', 'GREEN'))]]
# y = dat[['ZTD']]

print(train_x.head())

# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

train_x, scaler_x = standardized(train_x, 'MinMax')
test_x = scaler_x.transform(test_x)
# y_train, scaler_y = standardized(y_train, 'MinMax')

from joblib import dump, load

dump(scaler_x, 'Scaler/GPS_GOES_MinMax_scaler_x.bin', compress=True)
# dump(scaler_y, 'Scaler/GOES_MinMax_scaler_y.bin', compress=True)

es = EarlyStopping(verbose=1, patience=10)

# Initialiizinig the ANN
model = tf.keras.models.Sequential()
# Input layer
model.add(tf.keras.layers.Input(shape=(163,)))
# Adding first hidden layer
model.add(tf.keras.layers.Dense(units=163, activation=PReLU(), kernel_initializer='he_normal'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=163, activation=PReLU(), kernel_initializer='he_normal'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=80, activation=PReLU(), kernel_initializer='he_normal'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=80, activation=PReLU(), kernel_initializer='he_normal'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=40, activation=PReLU(), kernel_initializer='he_normal'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=40, activation=PReLU(), kernel_initializer='he_normal'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=20, activation=PReLU(), kernel_initializer='he_normal'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=20, activation=PReLU(), kernel_initializer='he_normal'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=10, activation=PReLU(), kernel_initializer='he_normal'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=10, activation=PReLU(), kernel_initializer='he_normal'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=5, activation=PReLU(), kernel_initializer='he_normal'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=5, activation=PReLU(), kernel_initializer='he_normal'))
# Adding the output layer
model.add(tf.keras.layers.Dense(units=1, activation='linear'))
# Compiling the ANN
model.compile(optimizer='adam', loss=['MSE'], metrics=['MAE'])

# Train the ANN on the Training set
model.fit(train_x, train_y, batch_size=512, epochs=150, validation_data=[test_x, test_y], callbacks=[es], verbose=0)
# plot_model(model, 'Plots/slope_model.png', show_shapes=True)
# Plot history: MSE
plt.plot(model.history.history['loss'], label='MSE (training data)')
plt.plot(model.history.history['val_loss'], label='MSE (validation data)')
plt.title('MSE for noise prediction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/GPS_GOES_model_MSE_history.png', dpi=300)
plt.clf()

# Plot history: MAE
plt.plot(model.history.history['MAE'], label='MAE (training data)')
plt.plot(model.history.history['val_MAE'], label='MAE (validation data)')
plt.title('MAE for noise prediction')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/GPS_GOES_model_MAE_history.png', dpi=300)

# Saving model
model.save('Model/GPS_PTE_fixed_hgtlvs_GOES_model')

# Predict different model
predict = model.predict(test_x)
true = test_y.values

print_metric(true, predict, 'GPS_GOES model')
plot_graphs(true, predict, 'GPS_GOES_model', 'Plots')

print('Finished Training')
