from extract_func.Extract_PTE_function import *

hgtlvs = [-100, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400
    , 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000
    , 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000
    , 14000, 15000, 16000, 17000, 18000, 19000, 20000, 25000, 30000, 35000, 40000]

GOES_dat = pd.read_csv('../GNSS_US/GNSS_US_WE_fixed_hgtlvs_cloud.csv')
# X = GOES_dat.iloc[:,7:]
# y = GOES_dat[['ZTD']]
train = GOES_dat[GOES_dat['Date'] > '2017-12-31']
test = GOES_dat[GOES_dat['Date'] < '2018-01-01']

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
x_train = train.iloc[:, 7:]
x_test = test.iloc[:, 7:]
y_train = train[['ZTD']]
y_test = test[['ZTD']]

x_train, scaler_x = standardized(x_train, 'MinMax')
x_test = scaler_x.transform(x_test)
y_train, scaler_y = standardized(y_train, 'MinMax')

# from joblib import dump, load
#
# dump(scaler_x, 'Scaler/US_WE_SAC_MinMax_scaler_x.bin', compress=True)
# dump(scaler_y, 'Scaler/US_WE_SAC_MinMax_scaler_y.bin', compress=True)

es = EarlyStopping(verbose=1, patience=10)

# Initializing the ANN
model = tf.keras.models.Sequential()
# Input layer
model.add(tf.keras.layers.Input(shape=(160,)))
# Adding first hidden layer
model.add(tf.keras.layers.Dense(units=160, activation=PReLU(), kernel_initializer='he_normal'))
#  Add dropout
model.add(tf.keras.layers.Dropout(0.1))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=80, activation=PReLU(), kernel_initializer='he_uniform'))
#  Add dropout
model.add(tf.keras.layers.Dropout(0.1))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=40, activation=PReLU(), kernel_initializer='he_uniform'))
#  Add dropout
model.add(tf.keras.layers.Dropout(0.1))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=20, activation=PReLU(), kernel_initializer='he_uniform'))
#  Add dropout
model.add(tf.keras.layers.Dropout(0.1))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=10, activation=PReLU(), kernel_initializer='he_uniform'))
#  Add dropout
model.add(tf.keras.layers.Dropout(0.1))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=5, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding the output layer
model.add(tf.keras.layers.Dense(units=1, activation='linear'))
# Compiling the ANN
model.compile(optimizer='adam', loss=['MSE'], metrics=['MAE'])

# Train the ANN on the Training set
model.fit(x_train, y_train, batch_size=1500, epochs=150, validation_split=0.2, callbacks=[es],verbose=0)

# Plot history: MSE
plt.plot(model.history.history['loss'], label='MSE (training data)')
plt.plot(model.history.history['val_loss'], label='MSE (validation data)')
plt.title('MSE for noise prediction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/SAC_MSE_history.png', dpi=300)
plt.clf()

# Plot history: MAE
plt.plot(model.history.history['MAE'], label='MAE (training data)')
plt.plot(model.history.history['val_MAE'], label='MAE (validation data)')
plt.title('MAE for noise prediction')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/SAC_MAE_history.png', dpi=300)
plt.clf()

# Saving model
model.save('Model/US_WE_PTE_SAC_fixed_hgtlvs_cloud_model')

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
fig.savefig('Plots/SAC_Ob_v_Pred.png', dpi=300)
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
cbar.ax.tick_params(labelsize=10)
fig.savefig('Plots/SAC_Resid_true.png', dpi=300)
plt.clf()
