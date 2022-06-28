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
# Adding the output layer
model.add(tf.keras.layers.Dense(units=1,activation='linear'))
# Compilling the ANN
model.compile(optimizer='adam',loss=['MSE'],metrics= ['MAE'])

# Trarin the ANN on the Training set
model.fit(x_train, y_train, batch_size=128, epochs=150, validation_split=0.2, callbacks=[es])

