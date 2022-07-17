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

df = pd.read_csv('../../GNSS_US/US/US_node_delay_vert_fixed_hgtlvs.csv')
df = df.dropna()
X = df[df.columns[pd.Series(df.columns).str.startswith(('Lat', 'Hgt_m', 'total_'))]]
y = df[['ZTD']].values

# Split data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
x_train, scaler_x = standardized(x_train, 'MinMax')
x_test = scaler_x.transform(x_test)
y_train, scaler_y = standardized(y_train, 'MinMax')
y_test = scaler_y.transform(y_test)

from joblib import dump, load

dump(scaler_x, 'Scaler/wet_hydro_model_scaler_x.bin', compress=True)
dump(scaler_y, 'Scaler/wet_hydro_model_scaler_y.bin', compress=True)

# Create the model
dim = layers.Input(x_train.shape[-1], )
model = create_mlp(dim.shape[-1], [53, 53, 25, 25], regress=True)
plot_model(model, 'Plots/wet_hydro_model.png', show_shapes=True)
model.compile(optimizer='adam', loss=['MSE'])

es = EarlyStopping(verbose=1, patience=10)
model.fit(x=x_train, y=y_train, batch_size=256, epochs=150, validation_data=(x_test, y_test),
          callbacks=[es], verbose=0)
# Plot history: MSE
plt.plot(model.history.history['loss'], label='Loss (training data)')
plt.plot(model.history.history['val_loss'], label='Loss (validation data)')
plt.title('MSE for noise prediction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/wet_hydro_model_MSE_history.png', dpi=300)
plt.clf()

# Saving model
model.save('Model/wet_hydro_US_PTE_fixed_hgtlvs_model')

# Predict different model
predict = scaler_y.inverse_transform(model.predict(x_test))
true = scaler_y.inverse_transform(y_test)

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
fig.suptitle('Wet and Hydro param obs vs pred')
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true, predict, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Observed', fontsize=10)
plt.ylabel('Predicted', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.savefig('Plots/wet_hydro_model_Ob_v_Pred.png', dpi=300)
plt.clf()

# Plot of residual of the prediction
fig = plt.figure()
fig.suptitle('Wet and Hydro param residual plot')
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(true, true - predict, cmap=white_viridis)
cbar = fig.colorbar(density)
cbar.set_label(label='Number of points per pixel', size=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('True', fontsize=10)
plt.ylabel('Residual', fontsize=10)
cbar.ax.tick_params(labelsize=10)
fig.savefig('Plots/wet_hydro_model_Resid_true.png', dpi=300)
plt.clf()