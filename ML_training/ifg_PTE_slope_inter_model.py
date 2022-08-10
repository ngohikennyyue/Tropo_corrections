import sys
import os

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *

num_threads = 20
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["TF_NUM_INTRAOP_THREADS"] = "20"
os.environ["TF_NUM_INTEROP_THREADS"] = "20"

tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)

# Read in data
dat = pd.read_csv('../../InSAR/Hawaii/Hawaii_ifg_PTE_fixed_hgtlvs_inter.csv')
dat = dat.dropna()

X = dat[dat.columns[pd.Series(dat.columns).str.startswith(('Lat', 'Hgt_m', 'P_', 'T_', 'e_', 'slope'))]]
y = dat[['ifg']]

print(X.head())

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

x_train, scaler_x = standardized(x_train, 'MinMax')
x_test = scaler_x.transform(x_test)
y_train, scaler_y = standardized(y_train, 'MinMax')

from joblib import dump, load

dump(scaler_x, 'Scaler/ifg_PTE_slope_inter_MinMax_scaler_x.bin', compress=True)
dump(scaler_y, 'Scaler/ifg_PTE_slope_inter_MinMax_scaler_y.bin', compress=True)

es = EarlyStopping(verbose=1, patience=15)

# Initialiizinig the ANN
model = tf.keras.models.Sequential()
# Input layer
model.add(tf.keras.layers.Input(shape=(156,)))
# Adding first hidden layer
model.add(tf.keras.layers.Dense(units=156, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding first hidden layer
model.add(tf.keras.layers.Dense(units=156, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=80, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=80, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=40, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=40, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=20, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=20, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=10, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=10, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=5, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding the output layer
model.add(tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer='he_uniform'))
# Compiling the ANN
model.compile(optimizer='adam', loss=['MSE'], metrics=['MAE'])

# Train the ANN on the Training set
model.fit(x_train, y_train, batch_size=1500, epochs=200, validation_split=0.2, callbacks=[es], verbose=0)

# Plot history: MSE
plt.plot(model.history.history['loss'], label='MSE (training data)')
plt.plot(model.history.history['val_loss'], label='MSE (validation data)')
plt.title('MSE for noise prediction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/ifg_PTE_slope_inter_MSE_history.png', dpi=300)
plt.clf()

# Plot history: MAE
plt.plot(model.history.history['MAE'], label='MAE (training data)')
plt.plot(model.history.history['val_MAE'], label='MAE (validation data)')
plt.title('MAE for noise prediction')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/ifg_PTE_slope_inter_MAE_history.png', dpi=300)

# Saving model
model.save('Model/ifg_PTE_slope_inter_model')

# Predict different model
predict = scaler_y.inverse_transform(model.predict(x_test))
true = y_test.values

from sklearn.metrics import mean_squared_error, r2_score

print("ANN model")
# The mean squared error
print('Mean squared error: %.10f' % mean_squared_error(true, predict))

# The R2 score
print('R2: %.5f' % r2_score(true, predict))

# The RMSE
rmse = np.sqrt(mean_squared_error(true, predict))
print('RMSE: %.5f' % rmse)

errors = predict - true
print('Average errror: %.5f' % np.mean(abs(errors)))

# Plot of Observation vs Prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true, predict, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Observed', fontsize=10)
plt.ylabel('Predicted', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.savefig('Plots/ifg_PTE_slope_inter_Ob_v_Pred.png', dpi=300)

# Plot of residual of the prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true, true - predict, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('True', fontsize=10)
plt.ylabel('Residual', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.savefig('Plots/ifg_PTE_slope_inter_Resid_true.png', dpi=300)

print('Finished Training')
