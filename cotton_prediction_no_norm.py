from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
import keras.backend as K
from cotton_model import *
import numpy as np


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    	Frame a time series as a supervised learning dataset.
    	Arguments:
    		data: Sequence of observations as a list or NumPy array.
    		n_in: Number of lag observations as input (X).
    		n_out: Number of observations as output (y).
    		dropnan: Boolean whether or not to drop rows with NaN values.
    	Returns:
    		Pandas DataFrame of series framed for supervised learning.
    	"""

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = read_csv('data/cotton.csv', header=0)
values = dataset.values

# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
# reframed = series_to_supervised(scaled, n_in=1, n_out=1, dropnan=True)

reframed = DataFrame(values)
# drop columns we don't want to predict
# reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
# reframed.drop(reframed.columns[[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]], axis=1, inplace=True)
print(reframed.head())

n_train_hours = int(0.8 * len(values))
my_train = values[:n_train_hours, :]
my_test = values[n_train_hours:, :]
my_train_X, my_train_y = my_train[:, 1:], my_train[:, 1]
my_test_X, my_test_y = my_test[:, 1:], my_test[:, 1]
# split into train and test sets
values = reframed.values

train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs

# train_X is train without last column which is the target at time t.
# (we have other variables at time t-1 and target at time t-1)

# train_Y is train with the last column only
# -1 means that accessing column 1 count backward
train_X, train_y = train[:, 1:], train[:, 0]
test_X, test_y = test[:, 1:], test[:, 0]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


def r2_keras(y_true, y_pred):
    """Coefficient of Determination
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )



TIME_STEPS = train_X.shape[1]
INPUT_DIM = train_X.shape[2]
lstm_units = 64
# deep network
# model = create_no_attention_model(TIME_STEPS, INPUT_DIM, lstm_units=lstm_units)
#model = create_attention_model(TIME_STEPS, INPUT_DIM, lstm_units=lstm_units)
# model = create_attention_layer_model(TIME_STEPS, INPUT_DIM, lstm_units=lstm_units)
model = create_auto_encoder_model(TIME_STEPS, INPUT_DIM, lstm_units=lstm_units)
# model = create_simple_model(TIME_STEPS, INPUT_DIM, lstm_units=lstm_units)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mean_squared_error', r2_keras])

# fit network
history = model.fit(train_X, train_y, epochs=500, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
inv_yhat = yhat[:, 0]

test_y = test_y.reshape((len(test_y), 1))
inv_y = test_y[:, 0]


# calculate RMSE
median_abs_error = median_absolute_error(inv_y, inv_yhat)
msle = mean_squared_log_error(inv_y, inv_yhat)
mse = mean_squared_error(inv_y, inv_yhat)
mae = mean_absolute_error(inv_y, inv_yhat)
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f, MSE %.3f, MAE %.3f, msle %.3f' % (rmse, mse, mae, msle))