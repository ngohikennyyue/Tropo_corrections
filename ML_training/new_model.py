from extract_func.Extract_PTE_function import *
import tensorflow as tf

data = pd.read_csv('../GNSS_US/GNSS_US_WE_fixed_hgtlvs_cloud.csv')
# X = GOES_dat.iloc[:,9:]
# y = GOES_dat[['ZTD']]
train = data[data['Date'] > '2017-12-31']
test = data[data['Date'] < '2018-01-01']
# Preprocessing data
Lat = train['Lat']
Hgt = train['Hgt_m']
P = train[train.columns[pd.Series(train.columns).str.startswith('P')]]
T = train[train.columns[pd.Series(train.columns).str.startswith('T')]]
e = train[train.columns[pd.Series(train.columns).str.startswith('e')]]

print('Train Lat max value: ', Lat.values.max())
print('Train Hgt max value: ', Hgt.values.max())
print('Train P max value: ', P.values.max())
print('Train T max value: ', T.values.max())
print('Train e max value: ', e.values.max())

Lat /= Lat.values.max()
Hgt /= Hgt.values.max()
P /= P.values.max()
T /= T.values.max()
e /= e.values.max()
train = pd.concat((Lat, Hgt, P, T, e, train['ZTD']), axis=1)

Lat = test['Lat']
Hgt = test['Hgt_m']
P = test[test.columns[pd.Series(test.columns).str.startswith('P')]]
T = test[test.columns[pd.Series(test.columns).str.startswith('T')]]
e = test[test.columns[pd.Series(test.columns).str.startswith('e')]]

print('Test Lat max value: ', Lat.values.max())
print('Test Hgt max value: ', Hgt.values.max())
print('Test P max value: ', P.values.max())
print('Test T max value: ', T.values.max())
print('Test e max value: ', e.values.max())

Lat /= Lat.values.max()
Hgt /= Hgt.values.max()
P /= P.values.max()
T /= T.values.max()
e /= e.values.max()
test = pd.concat((Lat, Hgt, P, T, e, test['ZTD']), axis=1)

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
x_train = train.iloc[:, :-1]
x_test = test.iloc[:, :-1]
y_train = train[['ZTD']]
y_test = test[['ZTD']]

# x_train, scaler_x = standardized(x_train, 'MinMax')
# x_test = scaler_x.transform(x_test)
# y_train, scaler_y = standardized(y_train, 'Standard')

from joblib import dump, load

# dump(scaler_x, 'Scaler/US_WE_interp_Standard_scaler_x.bin', compress=True)
# dump(scaler_y, 'Scaler/US_WE_interp_Standard_scaler_y.bin', compress=True)

es = EarlyStopping(verbose=1, patience=10)
inputs = keras.Input(shape=(784,))
# Initialiizinig the ANN
model = tf.keras.models.Sequential()
# Input layer
model.add(tf.keras.layers.Input(shape=(155,)))
# Adding first hidden layer
model.add(tf.keras.layers.Dense(units=155, activation=PReLU()))
#  Add dropout
model.add(tf.keras.layers.Dropout(0.1))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=70, activation=PReLU(), kernel_initializer='he_uniform'))
#  Add dropout
model.add(tf.keras.layers.Dropout(0.1))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=155, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding the output layer
model.add(tf.keras.layers.Dense(units=1, activation='linear'))
# Compiling the ANN
model.compile(optimizer='adam', loss=['MSE'], metrics=['MAE'])

# Train the ANN on the Training set
model.fit(x_train, y_train, batch_size=1000, epochs=150, validation_split=0.2, callbacks=[es], verbose=0)

# Plot history: MSE
plt.plot(model.history.history['loss'], label='Loss (training data)')
plt.plot(model.history.history['val_loss'], label='Loss (validation data)')
plt.title('MSE for noise prediction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/New_model_MSE_history.png', dpi=300)
plt.clf()

# Plot history: MAE
plt.plot(model.history.history['MAE'], label='MAE (training data)')
plt.plot(model.history.history['val_MAE'], label='MAE (validation data)')
plt.title('MAE for noise prediction')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/New_model_MAE_history.png', dpi=300)
plt.clf()

# Saving model
model.save('Model/NEW_Mod_US_WE_PTE_fixed_hgtlvs_cloud_model')

# Predict different model
# predict = scaler_y.inverse_transform(model.predict(x_test))
# true = y_test.values
predict = model.predict(x_test)
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
print('Average error: %.5f' % np.mean(abs(errors)))

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
fig.savefig('Plots/New_model_Ob_v_Pred.png', dpi=300)

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
fig.savefig('Plots/New_model_Resid_true.png', dpi=300)
