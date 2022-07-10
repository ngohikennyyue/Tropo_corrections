import sys
import os
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


def process_PTE_data(train, test, variable):
    if not isinstance(variable, (str, list, tuple)):
        raise TypeError("Variable not of type str")
    else:
        cs = MinMaxScaler()
        trainX = cs.fit_transform(train[train.columns[pd.Series(train.columns).str.startswith(variable)]])
        testX = cs.transform(test[test.columns[pd.Series(test.columns).str.startswith(variable)]])
        return trainX, testX, cs


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
data = pd.read_csv('../../GNSS_US/GNSS_US_WE_fixed_hgtlvs_cloud.csv')
train, test = train_test_split(data, test_size=0.2, random_state=40)
# train = data[data['Date'] > '2017-12-31']
# test = data[data['Date'] < '2018-01-01']

# Scale Target
cs = MinMaxScaler()
trainY = cs.fit_transform(train[['ZTD']])
testY = cs.transform(test[['ZTD']])

print('Create training and testing sets...')
trainA, testA, pteScaler = process_PTE_data(train, test, ('Lat', 'Lon', 'Hgt_m', 'P_', 'T_', 'e_'))
trainB, testB, cmiScaler = process_PTE_data(train, test, ('Lat', 'Lon', 'Hgt_m', 'CMI_C'))
# cs = MinMaxScaler()
# trainLoc = cs.fit_transform(train[['Lon', 'Lat', 'Hgt_m']])
# testLoc = cs.transform(test[['Lon', 'Lat', 'Hgt_m']])
# trainP, testP = process_PTE_data(train, test, 'P_')
# trainT, testT = process_PTE_data(train, test, 'T_')
# trainE, testE = process_PTE_data(train, test, 'e_')
# trainGOES, testGOES = process_PTE_data(train, test, 'CMI_')

from joblib import dump, load

dump(pteScaler, 'Scaler/MIML_pteScaler_x.bin', compress=True)
dump(cmiScaler, 'Scaler/MIML_cmiScaler_x.bin', compress=True)
dump(cs, 'Scaler/MIML_scaler_y.bin', compress=True)

# Model
InputA = layers.Input(156, )
InputB = layers.Input(7, )
# InputLoc = layers.Input(3, )
# InputP = layers.Input(51, )
# InputT = layers.Input(51, )
# InputE = layers.Input(51, )
# InputGOES = layers.Input(4, )

# P model
# p_model = create_mlp(InputP.shape[1], [25, 5])
# x = Dense(25, activation='relu')(InputP)
# x = Dense(10, activation='relu')(x)
# x = Dense(5, activation='relu')(x)
# x = Model(inputs=InputP, outputs=x)

# T model
# t_model = create_mlp(InputT.shape[1], [25, 5])
# y = Dense(25, activation='relu')(InputT)
# y = Dense(10, activation='relu')(y)
# y = Dense(5, activation='relu')(y)
# y = Model(inputs=InputT, outputs=y)

# E model
# e_model = create_mlp(InputE.shape[1], [25, 5])
# z = Dense(25, activation='relu')(InputE)
# z = Dense(10, activation='relu')(z)
# z = Dense(5, activation='relu')(z)
# z = Model(inputs=InputE, outputs=z)

model_a = create_mlp(InputA.shape[1], [80, 40, 10])
model_b = create_mlp(InputB.shape[1], [4, 4])

# loc = Dense(3, activation='linear')(InputLoc)
# loc = Model(inputs=InputLoc, outputs=loc)
# GOES = Dense(4, activation='linear')(InputGOES)
# GOES = Model(inputs=InputGOES, outputs=GOES)
# combined = concatenate([loc.output, p_model.output, t_model.output, e_model.output, GOES.output])
combined = concatenate([model_a.output, model_b.output])
xy = Dense(14, activation='relu')(combined)
xy = Dropout(0.1)(xy)
xy = Dense(14, activation='relu')(xy)
xy = Dense(1, activation='linear')(xy)
# model = Model(inputs=[loc.input, p_model.input, t_model.input, e_model.input, GOES.input], outputs=xy)
model = Model(inputs=[model_a.input, model_b.input], outputs=xy, name='ZTD_pred_model')
plot_model(model, 'Plots/ZTD_pred_model.png', show_shapes=True)

print(model.summary())

es = EarlyStopping(verbose=1, patience=10)
# Compile model
opt = Adam(learning_rate=1e-5)
model.compile(optimizer=opt, loss=['MSE'])
print('Model compiled...')
# Train the ANN on the Training set
# model.fit(x=[trainLoc, trainP, trainT, trainE, trainGOES], y=trainY, batch_size=1500, epochs=150,
#           validation_data=([testLoc, testP, testT, testE, testGOES], testY), callbacks=[es], verbose=0)
model.fit(x=[trainA, trainB], y=trainY, batch_size=64, epochs=150, validation_data=([testA, testB], testY),
          callbacks=[es], verbose=0)
# Plot history: MSE
plt.plot(model.history.history['loss'], label='Loss (training data)')
plt.plot(model.history.history['val_loss'], label='Loss (validation data)')
plt.title('MSE for noise prediction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/Test_New_model_MSE_history.png', dpi=300)
plt.clf()

# Plot history: MAE
# plt.plot(model.history.history['MAE'], label='MAE (training data)')
# plt.plot(model.history.history['val_MAE'], label='MAE (validation data)')
# plt.title('MAE for noise prediction')
# plt.ylabel('MAE value')
# plt.xlabel('No. epoch')
# plt.legend(loc="upper left")
# plt.savefig('Plots/New_model_MAE_history.png', dpi=300)
# plt.clf()

# Saving model
model.save('Model/Test_NEW_Mod_US_WE_PTE_fixed_hgtlvs_cloud_model')

# Predict different model
# predict = scaler_y.inverse_transform(model.predict(x_test))
# true = y_test.values
predict = cs.inverse_transform(model.predict([testA, testB]))
true = cs.inverse_transform(testY)

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
fig.savefig('Plots/Test_New_model_Ob_v_Pred.png', dpi=300)

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
fig.savefig('Plots/Test_New_model_Resid_true.png', dpi=300)
