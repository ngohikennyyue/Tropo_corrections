import sys
import os

import pandas as pd

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *
from sklearn.utils import shuffle

num_threads = 20
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["TF_NUM_INTRAOP_THREADS"] = "20"
os.environ["TF_NUM_INTEROP_THREADS"] = "20"

tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)

# Read in data
dat = pd.read_feather('../../InSAR/Large_scale/ML_data/train_data.ftr')
dat = dat.dropna()
test_dat = pd.read_feather('../../InSAR/Large_scale/ML_data/test_data.ftr')
test_dat = test_dat.dropna()

X = dat[dat.columns[pd.Series(dat.columns).str.startswith(('Lat', 'Hgt_m', 'date1_', 'date2_', 'slope'))]]
y = dat[['ifg']]
X, y = shuffle(X, y)
print(X.head())
test_X = test_dat[test_dat.columns[pd.Series(test_dat.columns).str.startswith(('Lat', 'Hgt_m', 'date1_', 'date2_',
                                                                               'slope'))]]
test_y = test_dat[['ifg']]

from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
pca = PCA(n_components=25)
X, scaler_x = standardized(X, 'MinMax')
X = pca.fit_transform(X)
x_test = scaler_x.transform(test_X)
x_test = pca.transform(x_test)
y, scaler_y = standardized(y, 'MinMax')
y_test = test_y
print('Variance of each component:', pca.explained_variance_ratio_)
print('\n Total Variance Explained:', round(sum(list(pca.explained_variance_ratio_))*100, 2))

from joblib import dump

dump(scaler_x, 'Scaler/ifg_PTE_slope_full_small_PCA_model_MinMax_scaler_x.bin', compress=True)
dump(scaler_y, 'Scaler/ifg_PTE_slope_full_small_PCA_model_MinMax_scaler_y.bin', compress=True)
dump(pca, 'Scaler/small_model_PCA.bin', compress=True)

es = EarlyStopping(verbose=1, patience=8)

# Initialiizinig the ANN
model = tf.keras.models.Sequential()
# Input layer
model.add(tf.keras.layers.Input(shape=(20,)))
# Adding first hidden layer
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
model.fit(X, y, batch_size=64, epochs=200, validation_split=0.2, callbacks=[es], verbose=0)

# Plot history: MSE
plt.plot(model.history.history['loss'], label='MSE (training data)')
plt.plot(model.history.history['val_loss'], label='MSE (validation data)')
plt.title('MSE for noise prediction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/Small_batch_32/ifg_PTE_full_small_PCA_model_MSE_history.png', dpi=300)
plt.clf()

# Plot history: MAE
plt.plot(model.history.history['MAE'], label='MAE (training data)')
plt.plot(model.history.history['val_MAE'], label='MAE (validation data)')
plt.title('MAE for noise prediction')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Plots/Small_batch_32/ifg_PTE_full_small_PCA_model_MAE_history.png', dpi=300)
plt.clf()

# Saving model
model.save('Model/ifg_PTE_full_small_PCA_model')

# Predict different model
predict = scaler_y.inverse_transform(model.predict(x_test))
# predict = model.predict(x_test)
true = y_test.values

predict_train = scaler_y.inverse_transform(model.predict(X))
true_train = scaler_y.inverse_transform(y)

print_metric(true, predict, 'ifg_PTE_slope_full_small_PCA_model')
plot_graphs(true, predict, 'ifg_PTE_slope_full_small_PCA_model', 'Plots/Small_batch_32')
print_metric(true_train, predict_train, 'ifg_PTE_slope_full_small_PCA_model_train')
plot_graphs(true_train, predict_train, 'ifg_PTE_slope_full_small_PCA_model_train', 'Plots/Small_batch_32')

print('Finished Training')
