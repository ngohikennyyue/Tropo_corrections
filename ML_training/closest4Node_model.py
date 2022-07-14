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
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import PReLU, LeakyReLU, ReLU

num_threads = 10
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["TF_NUM_INTRAOP_THREADS"] = "10"
os.environ["TF_NUM_INTEROP_THREADS"] = "10"

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


def create_mlp(dim, nodes, activation='relu', regress=False):
    model = Sequential()
    for i, n in enumerate(nodes):
        if i == 0:
            model.add(Dense(n, input_dim=dim, activation=activation))
        else:
            model.add(Dense(n, activation=activation))
    if regress:
        model.add(Dense(1, activation='linear'))

    return model


print('Read data...')
data = pd.read_csv('../../GNSS_US/US/US_PTE_closest_4Nodes_vert_fixed_hgtlvs.csv')
data = data.dropna()
data = data[data['sigZTD'] < 0.1]
train, test = train_test_split(data, test_size=0.2, random_state=40)
# train = data[data['Date'] > '2017-12-31']
# test = data[data['Date'] < '2018-01-01']

# Scale Target
cs = MinMaxScaler()
trainY = cs.fit_transform(train[['ZTD']])
testY = cs.transform(test[['ZTD']])

print('Create training and testing sets...')
trainP, testP, pScaler = process_PTE_data(train, test, ('Lat', 'Hgt_m', 'P_', 'dist_'))
trainT, testT, tScaler = process_PTE_data(train, test, ('Lat', 'Hgt_m', 'T_', 'dist_'))
trainE, testE, eScaler = process_PTE_data(train, test, ('Lat', 'Hgt_m', 'e_', 'dist_'))

from joblib import dump, load

dump(pScaler, 'Scaler/closest4Node_pScaler_x.bin', compress=True)
dump(tScaler, 'Scaler/closest4Node_tScaler_x.bin', compress=True)
dump(eScaler, 'Scaler/closest4Node_eScaler_x.bin', compress=True)
dump(cs, 'Scaler/closest4Node_scaler_y.bin', compress=True)

# Model
InputP = layers.Input(trainP.shape[1], )
InputT = layers.Input(trainT.shape[1], )
InputE = layers.Input(trainE.shape[1], )

# P model
p_model = create_mlp(InputP.shape[1], [210, 105, 51])

# T model
t_model = create_mlp(InputT.shape[1], [210, 105, 51])

# E model
e_model = create_mlp(InputE.shape[1], [210, 105, 51])

combined = concatenate([p_model.output, t_model.output, e_model.output])
# combined = concatenate([model_a.output, model_b.output])
xy = Dense(51, activation=PReLU())(combined)
xy = Dense(51, activation=PReLU())(xy)
xy = Dense(25, activation=PReLU())(xy)
xy = Dense(25, activation=PReLU())(xy)
xy = Dense(1, activation='linear')(xy)
model = Model(inputs=[p_model.input, t_model.input, e_model.input], outputs=xy,
              name='closest4Node_ZTD_pred_model')
# model = Model(inputs=[model_a.input, model_b.input], outputs=model_c.input, name='ZTD_pred_model')
plot_model(model, 'Plots/closest4Node_pred_model.png', show_shapes=True)

print(model.summary())

es = EarlyStopping(verbose=1, patience=10)
# Compile model
opt = Adam(learning_rate=1e-5)
model.compile(optimizer=opt, loss=['MSE'])
print('Model compiled...')
# Train the ANN on the Training set
model.fit(x=[trainP, trainT, trainE], y=trainY, batch_size=1000, epochs=150,
          validation_data=([testP, testT, testE], testY), callbacks=[es], verbose=0)

# Plot history: MSE
plt.plot(model.history.history['loss'], label='Loss (training data)')
plt.plot(model.history.history['val_loss'], label='Loss (validation data)')
plt.title('MSE for noise prediction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/closest4Node_MSE_history.png', dpi=300)
plt.clf()

# Saving model
model.save('Model/closest4Node_US_PTE_fixed_hgtlvs_model')

# Predict different model
# true = y_test.values
# predict = cs.inverse_transform(model.predict([testA, testB]))
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
fig.savefig('Plots/closest4Node_Ob_v_Pred.png', dpi=300)

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
fig.savefig('Plots/closest4Node_Resid_true.png', dpi=300)
