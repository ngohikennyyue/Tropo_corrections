import sys
import os
import tensorflow as tf
current  = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, concatenate, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split

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
# trainA, testA, pteScaler = process_PTE_data(train, test, ('Lat', 'Lon', 'Hgt_m', 'P_', 'T_', 'e_'))
# trainB, testB, cmiScaler = process_PTE_data(train, test, ('Lat', 'Lon', 'Hgt_m', 'CMI_C'))
# # cs = MinMaxScaler()
# trainLoc = cs.fit_transform(train[['Lon', 'Lat', 'Hgt_m']])
# testLoc = cs.transform(test[['Lon', 'Lat', 'Hgt_m']])
trainP, testP, pScaler = process_PTE_data(train, test, ('Lat', 'Hgt_m', 'P_'))
trainT, testT, tScaler = process_PTE_data(train, test, ('Lat', 'Hgt_m', 'T_'))
trainE, testE, eScaler = process_PTE_data(train, test, ('Lat', 'Hgt_m', 'e_'))
trainGOES, testGOES, GOESScaler = process_PTE_data(train, test, ('Lat', 'Hgt_m', 'CMI_'))

from joblib import dump, load

dump(pScaler, 'Scaler/Test_New_model1_pScaler_x.bin', compress=True)
dump(tScaler, 'Scaler/Test_New_model1_tScaler_x.bin', compress=True)
dump(eScaler, 'Scaler/Test_New_model1_eScaler_x.bin', compress=True)
dump(GOESScaler, 'Scaler/Test_New_model1_GOESScaler_x.bin', compress=True)
dump(cs, 'Scaler/Test_New_model1_scaler_y.bin', compress=True)

# Model
# InputA = layers.Input(156, )
# InputB = layers.Input(7, )
# InputLoc = layers.Input(3, )
InputP = layers.Input(53, )
InputT = layers.Input(53, )
InputE = layers.Input(53, )
InputGOES = layers.Input(6, )

# P model
p_model = create_mlp(InputP.shape[1], [51, 25, 5])
# x = Dense(25, activation='relu')(InputP)
# x = Dense(10, activation='relu')(x)
# x = Dense(5, activation='relu')(x)
# x = Model(inputs=InputP, outputs=x)

# T model
t_model = create_mlp(InputT.shape[1], [51, 25, 5])
# y = Dense(25, activation='relu')(InputT)
# y = Dense(10, activation='relu')(y)
# y = Dense(5, activation='relu')(y)
# y = Model(inputs=InputT, outputs=y)

# E model
e_model = create_mlp(InputE.shape[1], [51, 25, 5])
# z = Dense(25, activation='relu')(InputE)
# z = Dense(10, activation='relu')(z)
# z = Dense(5, activation='relu')(z)
# z = Model(inputs=InputE, outputs=z)

# GOES model
goes_model = create_mlp(InputGOES.shape[1], [4, 2, 1])
# model_a = create_mlp(InputA.shape[1], [80, 40, 10])
# model_b = create_mlp(InputB.shape[1], [10, 5])

combined = concatenate([p_model.output, t_model.output, e_model.output, goes_model.output])
# combined = concatenate([model_a.output, model_b.output])
xy = Dense(16, activation='relu')(combined)
xy = Dropout(0.2)(xy)
xy = Dense(16, activation='relu')(xy)
xy = Dropout(0.1)(xy)
xy = Dense(16, activation='relu')(xy)
xy = Dense(1, activation='linear')(xy)
model = Model(inputs=[p_model.input, t_model.input, e_model.input, goes_model.input], outputs=xy,
              name='New_ZTD_pred_model')
# model = Model(inputs=[model_a.input, model_b.input], outputs=model_c.input, name='ZTD_pred_model')
plot_model(model, 'Plots/Test_New_model1_pred_model.png', show_shapes=True)

print(model.summary())

es = EarlyStopping(verbose=1, patience=10)
# Compile model
opt = Adam(learning_rate=1e-5)
model.compile(optimizer=opt, loss=['MSE'])
print('Model compiled...')
# Train the ANN on the Training set
model.fit(x=[trainP, trainT, trainE, trainGOES], y=trainY, batch_size=64, epochs=150,
          validation_data=([testP, testT, testE, testGOES], testY), callbacks=[es], verbose=0)
# model.fit(x=[trainA, trainB], y=trainY, batch_size=1000, epochs=150, validation_data=([testA, testB], testY),
#           callbacks=[es], verbose=0)
# Plot history: MSE
plt.plot(model.history.history['loss'], label='Loss (training data)')
plt.plot(model.history.history['val_loss'], label='Loss (validation data)')
plt.title('MSE for noise prediction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/Test_New_model1_MSE_history.png', dpi=300)
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
model.save('Model/Test_NEW_model1_US_WE_PTE_fixed_hgtlvs_cloud_model')

# Predict different model
# true = y_test.values
# predict = cs.inverse_transform(model.predict([testA, testB]))
predict = cs.inverse_transform(model.predict([testP, testT, testE, testGOES]))
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
fig.savefig('Plots/Test_New_model1_Ob_v_Pred.png', dpi=300)

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
fig.savefig('Plots/Test_New_model1_Resid_true.png', dpi=300)
