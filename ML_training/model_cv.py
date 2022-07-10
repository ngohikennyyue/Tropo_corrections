import os
import sys
import tensorflow as tf

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
sys.path.append(parent)

from extract_func.Extract_PTE_function import *
from sklearn.metrics import mean_squared_error, r2_score
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


print('Read in data ...')
data = pd.read_csv('../GNSS_US/GNSS_US_WE_fixed_hgtlvs_cloud.csv')
kf = KFold(n_splits=8, shuffle=True, random_state=True)
cvscores = []

for train_index, test_index in kf.split(data):
    # Scale Target
    cs = MinMaxScaler()
    trainY = cs.fit_transform(data.iloc[train_index][['ZTD']])
    testY = cs.transform(data.iloc[test_index][['ZTD']])

    print('Create training and testing sets...')
    trainA, testA, pteScaler = process_PTE_data(data.iloc[train_index], data.iloc[test_index],
                                                ('Lat', 'Hgt_m', 'P_', 'T_', 'e_'))
    trainB, testB, cmiScaler = process_PTE_data(data.iloc[train_index], data.iloc[test_index],
                                                ('Lat', 'Hgt_m', 'CMI_C'))

    # Model
    InputA = layers.Input(155, )
    InputB = layers.Input(6, )

    model_a = create_mlp(InputA.shape[1], [155, 106, 53])
    model_b = create_mlp(InputB.shape[1], [6, 6, 6])

    combined = concatenate([model_a.output, model_b.output])
    xy = Dense(53, activation='relu')(combined)
    xy = Dense(53, activation='relu')(xy)
    xy = Dense(1, activation='linear')(xy)
    model = Model(inputs=[model_a.input, model_b.input], outputs=xy, name='ZTD_pred_model')
    es = EarlyStopping(verbose=1, patience=10)
    opt = Adam(learning_rate=1e-5)
    model.compile(optimizer=opt, loss=['MSE'])
    model.fit(x=[trainA, trainB], y=trainY, batch_size=64, epochs=150, validation_data=([testA, testB], testY),
              callbacks=[es], verbose=0)

    # Predict and compare metrics
    predict = cs.inverse_transform(model.predict([testA, testB]))
    true = cs.inverse_transform(testY)

    # The mean squared error
    mse = mean_squared_error(true, predict)
    # The R2 score
    r2 = r2_score(true, predict)
    # The RMSE
    rmse = np.sqrt(mean_squared_error(true, predict))

    print('Mean squared error: %.10f' % mse, 'R2: %.5f' % r2, 'RMSE: %.5f' % rmse)
    cvscores.append([mse, r2, rmse])

print("Avg MSE: %.10f%% (+/- %.10f%%)" % (np.mean(np.array(cvscores)[:, 0]), np.std(np.array(cvscores)[:, 0])))
print("Avg R2: %.10f%% (+/- %.10f%%)" % (np.mean(np.array(cvscores)[:, 1]), np.std(np.array(cvscores)[:, 1])))
print("Avg RMSE: %.10f%% (+/- %.10f%%)" % (np.mean(np.array(cvscores)[:, 2]), np.std(np.array(cvscores)[:, 2])))
