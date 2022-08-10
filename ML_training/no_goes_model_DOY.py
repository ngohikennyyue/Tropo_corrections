import sys
import os

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
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
GOES_dat = pd.read_csv('../../GNSS_US/US/PTE_vert_fixed_hgtlvs.csv')
GOES_dat = GOES_dat.dropna()
GOES_dat = GOES_dat[GOES_dat['sigZTD'] < 0.01]
X = GOES_dat[GOES_dat.columns[pd.Series(GOES_dat.columns).str.startswith(('DOY', 'Lat', 'Hgt_m', 'P_', 'T_', 'e_'))]]
y = GOES_dat[['ZTD']]

print(X.head())

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

x_train, scaler_x = standardized(x_train, 'MinMax')
x_test = scaler_x.transform(x_test)
y_train, scaler_y = standardized(y_train, 'MinMax')

from joblib import dump, load

dump(scaler_x, 'Scaler/DOY_noGOES_MinMax_scaler_x.bin', compress=True)
dump(scaler_y, 'Scaler/DOY_noGOES_MinMax_scaler_y.bin', compress=True)

# es = EarlyStopping(verbose=1,patience=10)

# Initialiizinig the ANN
model = tf.keras.models.Sequential()
# Input layer
model.add(tf.keras.layers.Input(shape=(156,)))
# Adding first hidden layer
model.add(tf.keras.layers.Dense(units=156, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=80, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=40, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=20, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=10, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=5, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding the output layer
model.add(tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer='he_uniform'))
# Compilling the ANN
model.compile(optimizer='adam', loss=['MSE'], metrics=['MAE'])

# Train the ANN on the Training set
model.fit(x_train, y_train, batch_size=15000, epochs=150, validation_split=0.2, verbose=0)
plot_model(model, 'Plots/noGOES_DOY_model.png', show_shapes=True)

# Plot history: MSE
plt.plot(model.history.history['loss'], label='MSE (training data)')
plt.plot(model.history.history['val_loss'], label='MSE (validation data)')
plt.title('MSE for noise prediction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/MSE_history_noGOES_DOY.png', dpi=300)
plt.clf()

# Plot history: MAE
plt.plot(model.history.history['MAE'], label='MAE (training data)')
plt.plot(model.history.history['val_MAE'], label='MAE (validation data)')
plt.title('MAE for noise prediction')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/MAE_history_noGOES_DOY.png', dpi=300)

# Saving model
model.save('Model/noGOES_DOY_PTE_fixed_hgtlvs_model')

# Predict different model
predict = scaler_y.inverse_transform(model.predict(x_test))
true = y_test.values

print_metric(true, predict, 'noGOES model with DOY')

plot_graphs(true, predict, 'noGOES_model_DOY', 'Plots')

print('Finished Training')
