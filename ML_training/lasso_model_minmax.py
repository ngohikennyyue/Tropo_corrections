from extract_func.Extract_PTE_function import *
from sklearn.linear_model import Lasso, Ridge, ElasticNet

GOES_dat = pd.read_csv('../GNSS_US/GNSS_US_WE_fixed_hgtlvs_cloud.csv')
# X = GOES_dat.iloc[:, 7:]
# y = GOES_dat[['ZTD']]
train = GOES_dat[GOES_dat['Date'] > '2017-12-31']
test = GOES_dat[GOES_dat['Date'] < '2018-01-01']
#
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)
x_train = train.iloc[:, 7:]
x_test = test.iloc[:, 7:]
y_train = train[['ZTD']]
y_test = test[['ZTD']]

x_train, scaler_x = standardized(x_train, 'MinMax')
x_test = scaler_x.transform(x_test)
# print(x_train)
# print('')
# print(x_test)
# y_train, scaler_y = standardized(y_train, 'MinMax')
# y_test = scaler_y.transform(y_test)

# from joblib import dump, load
# dump(scaler_x, 'Scaler/US_WE_SAC_Standard_scaler_x.bin', compress=True)
# dump(scaler_y, 'Scaler/US_WE_SAC_Standard_scaler_y.bin', compress=True)

# Lasso Regression implementation
lasso = Lasso(alpha=0.01,random_state=2011)
ridge = Ridge()
En = ElasticNet()
# Fit the Lasso model
lasso.fit(x_train, y_train)
ridge.fit(x_train, y_train)
En.fit(x_train, y_train)
# Create the model score
print('Lasso MinMax')
print('Test score (Test):', lasso.score(x_test, y_test))
print('Test score (Train):', lasso.score(x_train, y_train))
print('Coefficient:', lasso.coef_)
print('')
print('Ridge MinMax')
print('Test score (Test):', ridge.score(x_test, y_test))
print('Test score (Train):', ridge.score(x_train, y_train))
print('Coefficient:', ridge.coef_)
print('')
print('Elastic Net MinMax')
print('Test score (Test):', En.score(x_test, y_test))
print('Test score (Train):', En.score(x_train, y_train))
print('Coefficient:', En.coef_)
print('')
