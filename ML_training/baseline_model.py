from random import seed
from random import randrange
import pandas as pd
import numpy as np


# zero rule algorithm for regression
def zero_rule_algorithm_regression(train, test):
    prediction = sum(train) / float(len(train))
    predicted = [prediction for i in range(len(test))]
    return predicted


seed(1)

data = pd.read_csv('../GNSS_US/GNSS_US_WE_interp.csv')
# X = GOES_dat.iloc[:,9:]
# y = GOES_dat[['ZTD']]
train = data[data['Date'] > '2017-12-31']
test = data[data['Date'] < '2018-01-01']

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
# x_train = train.iloc[:, 7:]
# x_test = test.iloc[:, 7:]
y_train = train[['ZTD']]
y_test = test[['ZTD']]

prediction = zero_rule_algorithm_regression(y_train, y_test)
print('Expected: ', np.mean(y_train))
print('Prediction (test):', prediction)
