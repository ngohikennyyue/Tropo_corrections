import os
import math
import numpy as np
import datetime
import matplotlib.pyplot as plt
import glob
import pandas as pd
import rasterio
import xarray as xr
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.spatial.distance import pdist, cdist
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import PReLU, LeakyReLU, ReLU
from extract_func.Extract_PTE_func import *
hgtlvs = [ -100, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400
          ,2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000
          ,5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000
          ,14000, 15000, 16000, 17000, 18000, 19000, 20000, 25000, 30000, 35000, 40000]

GOES_dat = pd.read_csv('GNSS_US/GNSS_US_WE_fixed_hgtlvs_cloud.csv')
X = GOES_dat.iloc[:,9:]
y = GOES_dat[['ZTD']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

x_train, scaler_x = standardized(x_train, 'MinMax')
x_test = scaler_x.transform(x_test)
y_train, scaler_y = standardized(y_train, 'MinMax')

from joblib import dump, load
dump(scaler_x, 'Scaler/US_WE_MinMax_scaler_x.bin', compress=True)
dump(scaler_y, 'Scaler/US_WE_MinMax_scaler_y.bin', compress=True)

es = EarlyStopping(verbose=1,patience=10)

# Initialiizinig the ANN
model = tf.keras.models.Sequential()
# Input layer
model.add(tf.keras.layers.Input(shape=(158,)))
# Adding first hidden layer
model.add(tf.keras.layers.Dense(units=158,activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=80,activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=40,activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=20,activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=10,activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=5,activation=PReLU(), kernel_initializer='he_uniform'))
# Adding the output layer
model.add(tf.keras.layers.Dense(units=1,activation='linear'))
# Compilling the ANN
model.compile(optimizer='adam',loss=['MSE'],metrics= ['MAE'])

# Train the ANN on the Training set
model.fit(x_train, y_train, batch_size=128, epochs=150, validation_split=0.2, callbacks=[es])

# Plot history: MSE
plt.plot(model.history.history['loss'], label='MSE (training data)')
plt.plot(model.history.history['val_loss'], label='MSE (validation data)')
plt.title('MSE for noise prediction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('MSE_history.png', dpi=300)

# Plot history: MAE
plt.plot(model.history.history['MAE'], label='MAE (training data)')
plt.plot(model.history.history['val_MAE'], label='MAE (validation data)')
plt.title('MAE for noise prediction')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('MAE_history.png', dpi=300)

# Saving model
model.save('Model/US_WE_PTE_fixed_hgtlvs_cloud_model')

# Predict different model
predict = scaler_y.inverse_transform(model.predict(x_test))
true = y_test.values

from sklearn.metrics import mean_squared_error, r2_score
print("ANN model")
# The mean squared error
print('Mean squared error: %.10f'% mean_squared_error(true, predict))

# The R2 score
print('R2: %.5f'% r2_score(true,predict))

# The RMSE
rmse = np.sqrt(mean_squared_error(true, predict))
print('RMSE: %.5f' % rmse)

errors = predict - true
print('Average errror: %.5f' %np.mean(abs(errors)))

# Plot of Observation vs Prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true, predict, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel',size=10)
ax.tick_params(axis='both',which='major',labelsize=10)
plt.xlabel('Observed',fontsize=10)
plt.ylabel('Predicted',fontsize=10)
cbar.ax.tick_params(labelsize=10)
plt.figsave('Ob_v_Pred.png', dpi=300)

# Plot of residual of the prediction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true, true-predict, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel',size=10)
ax.tick_params(axis='both',which='major',labelsize=10)
plt.xlabel('True',fontsize=10)
plt.ylabel('Residual',fontsize=10)
cbar.ax.tick_params(labelsize=10)
plt.figsave('Resid_true.png', dpi=300)


