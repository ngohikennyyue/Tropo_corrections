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
data = pd.read_csv('../../GNSS_US/US/NewPTE_vert_fixed_hgtlvs.csv')
train, test = train_test_split(data, test_size=0.2, random_state=40)
# train = data[data['Date'] > '2017-12-31']
# test = data[data['Date'] < '2018-01-01']

# Scale Target
cs = MinMaxScaler()
trainY = cs.fit_transform(train[['ZTD']])
testY = cs.transform(test[['ZTD']])

print('Create training and testing sets...')
trainP, testP, pScaler = process_PTE_data(train, test, ('Lat', 'Hgt_m', 'P_'))
trainT, testT, tScaler = process_PTE_data(train, test, ('Lat', 'Hgt_m', 'T_'))
trainE, testE, eScaler = process_PTE_data(train, test, ('Lat', 'Hgt_m', 'e_'))

from joblib import dump, load

dump(pScaler, 'Scaler/Test_New_model3_pScaler_x.bin', compress=True)
dump(tScaler, 'Scaler/Test_New_model3_tScaler_x.bin', compress=True)
dump(eScaler, 'Scaler/Test_New_model3_eScaler_x.bin', compress=True)
dump(cs, 'Scaler/Test_New_model3_scaler_y.bin', compress=True)

# Model
InputP = layers.Input(53, )
InputT = layers.Input(53, )
InputE = layers.Input(53, )

# P model
p_model = create_mlp(InputP.shape[1], [53, 25, 10])

# T model
t_model = create_mlp(InputT.shape[1], [53, 25, 10])

# E model
e_model = create_mlp(InputE.shape[1], [53, 25, 10])


combined = concatenate([p_model.output, t_model.output, e_model.output])
xy = Dense(53, activation='relu')(combined)
xy = Dense(53, activation='relu')(xy)
xy = Dense(25, activation='relu')(xy)
xy = Dense(25, activation='relu')(xy)
xy = Dense(1, activation='linear')(xy)
model = Model(inputs=[p_model.input, t_model.input, e_model.input], outputs=xy,
              name='PTE_ZTD_pred_model')
# model = Model(inputs=[model_a.input, model_b.input], outputs=model_c.input, name='ZTD_pred_model')
plot_model(model, 'Plots/Test_New_model3_pred_model.png', show_shapes=True)

print(model.summary())

es = EarlyStopping(verbose=1, patience=10)
# Compile model
opt = Adam(learning_rate=1e-8)
model.compile(optimizer=opt, loss=['MSE'])
print('Model compiled...')
# Train the ANN on the Training set
# model.fit(x=[trainLoc, trainP, trainT, trainE, trainGOES], y=trainY, batch_size=1500, epochs=150,
#           validation_data=([testLoc, testP, testT, testE, testGOES], testY), callbacks=[es], verbose=0)
model.fit(x=[trainP, trainT, trainE], y=trainY, batch_size=64, epochs=150, validation_data=([testP, testT, testE], testY),
          callbacks=[es], verbose=0)
# Plot history: MSE
plt.plot(model.history.history['loss'], label='Loss (training data)')
plt.plot(model.history.history['val_loss'], label='Loss (validation data)')
plt.title('MSE for noise prediction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/Test_New_model3_MSE_history.png', dpi=300)
plt.clf()

# Saving model
model.save('Model/Test_New_model3_US_PTE_fixed_hgtlvs_cloud_model')

# Predict different model
# predict = scaler_y.inverse_transform(model.predict(x_test))
# true = y_test.values
predict = cs.inverse_transform(model.predict([testP, testT, testE]))
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
fig.savefig('Plots/Test_New_model3_Ob_v_Pred.png', dpi=300)

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
fig.savefig('Plots/Test_New_model3_Resid_true.png', dpi=300)
