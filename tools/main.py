import pandas as pd
from Extraction_func import convert_rad, standardized, extract_param_GNSS
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import PReLU, LeakyReLU, ReLU
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.metrics import mean_squared_error, r2_score

hk_data = pd.read_csv('/mnt/stor/geob/jlmd9g/Kenny/GNSS/Subtrop/HK/GNSS_subtrop_hgtlvs_fixed.csv', index_col=False)
eur_data = pd.read_csv('/mnt/stor/geob/jlmd9g/Kenny/GNSS/Europe/GNSS_Station_Europe_fixed_hgtlvs_vertical.csv',
                       index_col=False)
us_east = pd.read_csv('/mnt/stor/geob/jlmd9g/Kenny/GNSS/US_east/GNSS_Station_US_E_fixed_hgtlvs.csv', index_col=False)
us_west = pd.read_csv('/mnt/stor/geob/jlmd9g/Kenny/GNSS/US_wast/GNSS_Station_US_W_fixed_hgtlvs.csv', index_col=False)

comb_dat = pd.concat([eur_data, us_west, us_east], ignore_index=True)

# Hydrostatic Delay
X = comb_dat.loc[:, comb_dat.columns.str.startswith(tuple(['Lat', 'Hgt_m', 'P_', 'T_']))]
y = comb_dat[['hydrostatic_delay']]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# MinMax Scaler
x_train, scaler_x = standardized(x_train, 'MinMax')
x_test = scaler_x.transform(x_test)
y_train, scaler_y = standardized(y_train, 'MinMax')
# y_test = scaler_y.transform(y_test)

# Store scaler
dump(scaler_x, '/mnt/stor/geob/jlmd9g/Kenny/Scaler/US_Eur_Hydro_MinMax_scaler_x.bin', compress=True)
dump(scaler_y, '/mnt/stor/geob/jlmd9g/Kenny/Scaler/US_Eur_Hydro_MinMax_scaler_y.bin', compress=True)

# Early Stopping to prevent over-fitting
es = EarlyStopping(verbose=1, patience=5)

# Initializing the ANN
model = tf.keras.models.Sequential()
# Input layer
model.add(tf.keras.layers.Input(shape=(52,)))
# Adding first hidden layer
model.add(tf.keras.layers.Dense(units=52, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=52, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=26, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=13, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=6, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding the output layer
model.add(tf.keras.layers.Dense(units=1, activation='linear'))
# Compiling the ANN
model.compile(optimizer='adam', loss=['MSE'], metrics=['MAE'])

# Train the ANN on the Training set
model.fit(x_train, y_train, batch_size=64, epochs=100, callbacks=[es], validation_split=0.2)

# Test model
predict = scaler_y.inverse_transform(model.predict(x_test))
true = y_test.values


print("ANN model")
# The mean squared error
print('Mean squared error: %.10f'
      % mean_squared_error(true, predict))

# The R2 score
print('R2: %.5f'
     % r2_score(true,predict))

# The RMSE
rmse = np.sqrt(mean_squared_error(true, predict))
print('RMSE: %.5f' % rmse)

errors = predict - true
# mape = 100 * np.mean(errors / true)
# accuracy = 100 - abs(mape)
print('Average error: %.5f' %np.mean(abs(errors)))

