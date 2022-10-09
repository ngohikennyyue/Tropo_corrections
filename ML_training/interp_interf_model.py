import sys
import os

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *
from tensorflow.keras.utils import plot_model

num_threads = 20
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["TF_NUM_INTRAOP_THREADS"] = "20"
os.environ["TF_NUM_INTEROP_THREADS"] = "20"

tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)

# Read in data
dat = pd.read_csv('../../InSAR/Large_scale/East/East_interp_interf.csv')
dat = dat.dropna()
X = dat.iloc[:, :-1]
y = dat[['ifg']]

print(X.head())

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

x_train, scaler_x = standardized(x_train, 'MinMax')
x_test = scaler_x.transform(x_test)
y_train, scaler_y = standardized(y_train, 'MinMax')

from joblib import dump, load

dump(scaler_x, 'Scaler/interp_interf_MinMax_scaler_x.bin', compress=True)
dump(scaler_y, 'Scaler/interp_interf_MinMax_scaler_y.bin', compress=True)

es = EarlyStopping(verbose=1, patience=10)

# Initialiizinig the ANN
model = tf.keras.models.Sequential()
# Input layer
model.add(tf.keras.layers.Input(shape=(9,)))
# Adding first hidden layer
model.add(tf.keras.layers.Dense(units=9, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=9, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding hidden layer
model.add(tf.keras.layers.Dense(units=9, activation=PReLU(), kernel_initializer='he_uniform'))
# Adding the output layer
model.add(tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer='he_uniform'))
# Compiling the ANN
model.compile(optimizer='adam', loss=['MSE'], metrics=['MAE'])

# Train the ANN on the Training set
model.fit(x_train, y_train, batch_size=64, epochs=200, validation_split=0.2, callbacks=[es], verbose=0)
plot_model(model, 'Plots/interp_interf_model.png', show_shapes=True)
# Plot history: MSE
plt.plot(model.history.history['loss'], label='MSE (training data)')
plt.plot(model.history.history['val_loss'], label='MSE (validation data)')
plt.title('MSE for noise prediction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/interp_interf_model_MSE_history.png', dpi=300)
plt.clf()

# Plot history: MAE
plt.plot(model.history.history['MAE'], label='MAE (training data)')
plt.plot(model.history.history['val_MAE'], label='MAE (validation data)')
plt.title('MAE for noise prediction')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/interp_interf_model_MAE_history.png', dpi=300)

# Saving model
model.save('Model/interp_interf_model')

# Predict different model
predict = scaler_y.inverse_transform(model.predict(x_test))
true = y_test.values

print_metric(true, predict, 'interp_interf model')

plot_graphs(true, predict, 'interp_interf_model', 'Plots')

print('Finished Training')
