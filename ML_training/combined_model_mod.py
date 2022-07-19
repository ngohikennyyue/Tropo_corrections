import sys
import os

import numpy as np
import tensorflow as tf

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, concatenate, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

num_threads = 7
os.environ["OMP_NUM_THREADS"] = "7"
os.environ["TF_NUM_INTRAOP_THREADS"] = "7"
os.environ["TF_NUM_INTEROP_THREADS"] = "7"

tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)


def process_PTE_data(train, test, variable, scale=False):
    if not isinstance(variable, (str, list, tuple)):
        raise TypeError("Variable not of type str")
    else:
        if scale:
            cs = MinMaxScaler()
            trainX = cs.fit_transform(train[train.columns[pd.Series(train.columns).str.startswith(variable)]])
            testX = cs.transform(test[test.columns[pd.Series(test.columns).str.startswith(variable)]])
            return trainX, testX, cs
        else:
            trainX = train[train.columns[pd.Series(train.columns).str.startswith(variable)]]
            testX = test[test.columns[pd.Series(test.columns).str.startswith(variable)]]
            return trainX, testX


def create_mlp(dim, nodes, regress=False):
    model = Sequential()
    for i, n in enumerate(nodes):
        if i == 0:
            model.add(Dense(n, input_dim=dim, activation='relu'))
        else:
            model.add(Dense(n, activation='relu'))
    if regress:
        model.add(Dense(1, activation='linear'))

    return model


print('Read data...')
data = pd.read_csv('../../GNSS_US/US/NewPTE_vert_fixed_hgtlvs.csv')
data = data.dropna()
data = data[data['sigZTD'] < 0.1]
train, test = train_test_split(data, test_size=0.3, random_state=40)
# train = data[data['Date'] > '2017-12-31']
# test = data[data['Date'] < '2018-01-01']

# Scale Target
# cs = MinMaxScaler()
# trainY = cs.fit_transform(train[['ZTD']])
# testY = cs.transform(test[['ZTD']])
trainY = train[['ZTD']].values
testY = test[['ZTD']].values

print('Create training and testing sets...')
trainP, testP = process_PTE_data(train, test, ('Lat', 'Hgt_m', 'P_'))
trainT, testT = process_PTE_data(train, test, ('Lat', 'Hgt_m', 'T_'))
trainE, testE = process_PTE_data(train, test, ('Lat', 'Hgt_m', 'e_'))
trainX, testX = process_PTE_data(train, test, ('Lat', 'Hgt_m', 'P_', 'T_', 'e_'))

# Load model
Norm_model = tf.keras.models.load_model('../Model/Full_US_PTE_fixed_hgtlvs_model')
Multi_model = tf.keras.models.load_model(
    '../Multiple_Input_Model/Model/Test_New_model3_US_PTE_fixed_hgtlvs_cloud_model')
from joblib import dump, load

# Load scaler
scaler_x = load('../Scaler/US_WE_noGOES_MinMax_scaler_x.bin')
scaler_y = load('../Scaler/US_WE_noGOES_MinMax_scaler_y.bin')
scalerP = load('../Multiple_Input_Model/Scaler/Test_New_model3_pScaler_x.bin')
scalerT = load('../Multiple_Input_Model/Scaler/Test_New_model3_tScaler_x.bin')
scalerE = load('../Multiple_Input_Model/Scaler/Test_New_model3_eScaler_x.bin')
scaler_y1 = load('../Multiple_Input_Model/Scaler/Test_New_model3_scaler_y.bin')

pred_1 = scaler_y.inverse_transform(Norm_model.predict(scaler_x.transform(trainX)))
pred_2 = scaler_y1.inverse_transform(
    Multi_model.predict([scalerP.transform(trainP), scalerT.transform(trainT), scalerE.transform(trainE)]))

input = np.hstack((pred_1.reshape(-1, 1), pred_2.reshape(-1, 1)))

# Initialiizinig the ANN
model = tf.keras.models.Sequential()
# Input layer
model.add(tf.keras.layers.Input(shape=(2,)))
# Adding first hidden layer
model.add(tf.keras.layers.Dense(units=2, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding the output layer
model.add(tf.keras.layers.Dense(units=1, activation='linear'))
# Compilling the ANN
model.compile(optimizer='adam', loss=['MSE'], metrics=['MAE'])

# Train the ANN on the Training set
model.fit(input, trainY, batch_size=1500, epochs=150, validation_split=0.2, verbose=0)

# Plot history: MSE
plt.plot(model.history.history['loss'], label='Loss (training data)')
plt.plot(model.history.history['val_loss'], label='Loss (validation data)')
plt.title('MSE for noise prediction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/combined_model_mod_MSE_history.png', dpi=300)
plt.clf()

# Saving model
model.save('Model/combined_model_mod_US_PTE_fixed_hgtlvs_model')

# Predict different model
test_1 = scaler_y.inverse_transform(Norm_model.predict(scaler_x.transform(testX)))
test_2 = scaler_y1.inverse_transform(
    Multi_model.predict([scalerP.transform(testP), scalerT.transform(testT), scalerE.transform(testE)]))
test_join = np.hstack((test_1.reshape(-1, 1), test_2.reshape(-1, 1)))

predict = model.predict(test_join)
true = testY

print(predict[:5], true[:5])
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
fig.suptitle('Observe vs Predict')
fig.savefig('Plots/combined_model_mod_Ob_v_Pred.png', dpi=300)
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
fig.suptitle('Residual vs true')
fig.savefig('Plots/combined_model_mod_Resid_true.png', dpi=300)
plt.clf()
