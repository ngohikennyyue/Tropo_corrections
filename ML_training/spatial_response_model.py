from extract_func.Extract_PTE_function import *

data = pd.read_csv('../GNSS_US/GNSS_US_WE_interp.csv')
# X = GOES_dat.iloc[:,9:]
# y = GOES_dat[['ZTD']]
train = data[data['Date'] > '2017-12-31']
test = data[data['Date'] < '2018-01-01']

x_train = train.iloc[:, 7:10]
x_test = test.iloc[:, 7:10]
y_train = train[['ZTD']]
y_test = test[['ZTD']]

x_train, scaler_x = standardized(x_train, 'MinMax')
x_test = scaler_x.transform(x_test)
# y_train, scaler_y = standardized(y_train, 'MinMax')

es = EarlyStopping(verbose=1, patience=10)

# Initialiizinig the ANN
model = tf.keras.models.Sequential()
# Input layer
model.add(tf.keras.layers.Input(shape=(3,)))
# Adding first hidden layer
model.add(tf.keras.layers.Dense(units=8, activation=PReLU(), kernel_initializer='he_normal'))
# Adding the output layer
model.add(tf.keras.layers.Dense(units=1, activation='linear'))
# Compiling the ANN
model.compile(optimizer='adam', loss=['MSE'], metrics=['MAE'])

# Train the ANN on the Training set
model.fit(x_train, y_train, batch_size=150, epochs=150, validation_split=0.2, callbacks=[es], verbose=0)

# Plot history: MSE
plt.plot(model.history.history['loss'], label='Loss (training data)')
plt.plot(model.history.history['val_loss'], label='Loss (validation data)')
plt.title('MSE for noise prediction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/SRM_MSE_history.png', dpi=300)
plt.clf()

# Plot history: MAE
plt.plot(model.history.history['MAE'], label='MAE (training data)')
plt.plot(model.history.history['val_MAE'], label='MAE (validation data)')
plt.title('MAE for noise prediction')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/SRM_MAE_history.png', dpi=300)
plt.clf()

# Saving model
# model.save('Model/US_WE_PTE_fixed_hgtlvs_cloud_model')

# Predict different model
predict = model.predict(x_test)
true = y_test.values
# predict = model.predict(x_test)
# true = y_test.values

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
fig.savefig('Plots/SRM_Ob_v_Pred.png', dpi=300)

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
fig.savefig('Plots/SRM_Resid_true.png', dpi=300)
