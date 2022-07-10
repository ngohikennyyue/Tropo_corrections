import numpy as np
from extract_func.Extract_PTE_function import *
from sklearn import datasets
from sklearn.linear_model import LinearRegression

GOES_dat = pd.read_csv('../GNSS_US/GNSS_US_WE_interp.csv')
# X = GOES_dat.iloc[:,7:]
# y = GOES_dat[['ZTD']]
train = GOES_dat[GOES_dat['Date'] > '2017-12-31']
test = GOES_dat[GOES_dat['Date'] < '2018-01-01']
print(len(train), len(test))

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
x_train = train.iloc[:, np.r_[7:8,9:13]]
x_test = test.iloc[:, np.r_[7:8,9:13]]
y_train = train[['ZTD']]
y_test = test[['ZTD']]

x_train, scaler_x = standardized(x_train, 'MinMax')
x_test = scaler_x.transform(x_test)
y_train, scaler_y = standardized(y_train, 'MinMax')
y_test = scaler_y.transform(y_test)

# from joblib import dump, load
# dump(scaler_x, 'Scaler/US_WE_SAC_Standard_scaler_x.bin', compress=True)
# dump(scaler_y, 'Scaler/US_WE_SAC_Standard_scaler_y.bin', compress=True)

regr = LinearRegression()

regr.fit(x_train, y_train)
print('Test score (Test):', regr.score(x_test, y_test))

from sklearn.metrics import mean_absolute_error, mean_squared_error
predict = scaler_y.inverse_transform(regr.predict(x_test))
true = scaler_y.inverse_transform(y_test)
mae = mean_absolute_error(true, predict)
mse = mean_squared_error(true, predict)
rmse = np.sqrt(mse)

print('MAE: ', mae)
print('MSE: ', mse)
print('RMSE: ', rmse)
print('Coef: ', regr.coef_)

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
fig.savefig('Plots/linear_Ob_v_Pred.png', dpi=300)
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
fig.savefig('Plots/linear_Resid_true.png', dpi=300)
plt.clf()
