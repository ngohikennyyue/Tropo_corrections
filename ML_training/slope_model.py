import sys
import os

import matplotlib.pyplot as plt

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *
from tensorflow.keras.utils import plot_model

num_threads = 10
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["TF_NUM_INTRAOP_THREADS"] = "10"
os.environ["TF_NUM_INTEROP_THREADS"] = "10"

tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)

# Read in data
dat = pd.read_csv('../../GNSS_US/US/PTE_vert_fixed_hgtlvs_slope.csv')
dat = dat.dropna()
X = dat[dat.columns[pd.Series(dat.columns).str.startswith(('Lat', 'Lon', 'Hgt_m', 'P_', 'T_', 'e_', 'Slope'))]]
y = dat[['ZTD']]

print(X.head())

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

x_train, scaler_x = standardized(x_train, 'MinMax')
x_test = scaler_x.transform(x_test)
y_train, scaler_y = standardized(y_train, 'MinMax')
y_test = scaler_y.transform(y_test)

from joblib import dump, load

dump(scaler_x, 'Scaler/Slope_MinMax_scaler_x_1.bin', compress=True)
dump(scaler_y, 'Scaler/Slope_MinMax_scaler_y_1.bin', compress=True)
R2 = []
Rmse = []
es = EarlyStopping(verbose=1, patience=10)
Batchsize = [32, 64, 128, 1028, 2048]
for batch in Batchsize:
    # Initialiizinig the ANN
    model = tf.keras.models.Sequential()
    # Input layer
    model.add(tf.keras.layers.Input(shape=(157,)))
    # Adding first hidden layer
    model.add(tf.keras.layers.Dense(units=157, activation=PReLU(), kernel_initializer='he_uniform'))
    # Adding hidden layer
    model.add(tf.keras.layers.Dense(units=80, activation=PReLU(), kernel_initializer='he_uniform'))
    # Adding hidden layer
    model.add(tf.keras.layers.Dense(units=40, activation=PReLU(), kernel_initializer='he_uniform'))
    # Adding hidden layer
    model.add(tf.keras.layers.Dense(units=20, activation=PReLU(), kernel_initializer='he_uniform'))
    # Adding hidden layer
    model.add(tf.keras.layers.Dense(units=10, activation=PReLU(), kernel_initializer='he_uniform'))
    # Adding hidden layer
    model.add(tf.keras.layers.Dense(units=5, activation=PReLU(), kernel_initializer='he_uniform'))
    # Adding the output layer
    model.add(tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer='he_uniform'))
    # Compiling the ANN
    model.compile(optimizer='adam', loss=['MSE'], metrics=['MAE'])

    # Train the ANN on the Training set
    model.fit(x_train, y_train, batch_size=batch, epochs=150, callbacks=[es], verbose=0, validation_data=(x_test,y_test))
    # plot_model(model, 'Plots/slope_model.png', show_shapes=True)
    # Plot history: MSE
    plt.plot(model.history.history['loss'], label='MSE (training data)')
    plt.plot(model.history.history['val_loss'], label='MSE (validation data)')
    plt.title('MSE for noise prediction')
    plt.ylabel('MSE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.savefig('Plots/Slope_model_batchsize_{}_MSE_history.png'.format(batch), dpi=300)
    plt.clf()

    # Plot history: MAE
    plt.plot(model.history.history['MAE'], label='MAE (training data)')
    plt.plot(model.history.history['val_MAE'], label='MAE (validation data)')
    plt.title('MAE for noise prediction')
    plt.ylabel('MAE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.savefig('Plots/Slope_model_batchsize_{}_MAE_history.png'.format(batch), dpi=300)
    plt.clf()

    # Saving model
    model.save('Model/PTE_fixed_hgtlvs_slope_model_batchsize_{}'.format(batch))

    # Predict different model
    predict = scaler_y.inverse_transform(model.predict(x_test))
    true = scaler_y.inverse_transform(y_test)

    from sklearn.metrics import mean_squared_error, r2_score

    R2.append(r2_score(true, predict))
    Rmse.append(np.sqrt(mean_squared_error(true, predict)))
    print_metric(true, predict, 'Slope model {}'.format(batch))

    plot_graphs(true, predict, 'Slope_model_batchsize_{}'.format(batch), 'Plots')

plt.clf()
plt.scatter(Batchsize, R2)
plt.xlabel('Batchsize')
plt.ylabel('R2')
plt.savefig('Plots/R2_batchsize.png')
plt.clf()

plt.scatter(Batchsize, Rmse)
plt.xlabel('Batchsize')
plt.ylabel('Rmse')
plt.savefig('Plots/Rmse_batchsize.png')
plt.clf()

print('Finished Training')
