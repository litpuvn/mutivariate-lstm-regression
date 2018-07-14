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

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, Activation, Input
import keras.backend as K
from attention_utils import self_attention_3d_block
from attention import Attention
from keras import initializers, regularizers, constraints


def create_simple_model(timesteps, input_dims):
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(timesteps, input_dims)))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    return model


def create_deep_model(timesteps, input_dims):
    # deep network
    lstm_units = 100

    model = Sequential()
    model.add(LSTM(units=lstm_units,
                   input_shape=(timesteps, input_dims),
                   return_sequences=True
                   ))
    model.add(Dropout(0.1))
    model.add(LSTM(
        units=50,
        return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))
    model.add(Activation("linear"))

    return model

def create_auto_encoder_model(timesteps, input_dims, lstm_units=100):
    encoding_dim = 8

    input_layer = Input(shape=(timesteps, input_dims,))

    encoder = Dense(encoding_dim, activation="tanh",
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(input_dims / 2), activation="relu")(encoder)

    decoder = Dense(int(input_dims / 2), activation='tanh')(encoder)
    decoder = Dense(input_dims, activation='relu')(decoder)


    lstm_out = LSTM(units=lstm_units, return_sequences=False)(decoder)

    drop = Dropout(0.1)(lstm_out)
    drop = Dropout(0.1)(drop)

    # attention_mul = self_attention_3d_block(drop)

    dense_out = Dense(units=1)(drop)
    outputs = Activation("linear")(dense_out)

    autoencoder = Model(inputs=input_layer, outputs=outputs)

    return autoencoder

def create_attention_model(timesteps, input_dims, lstm_units=100):

    inputs = Input(shape=(timesteps, input_dims,))
    lstm_out = LSTM(units=lstm_units, return_sequences=True)(inputs)

    drop = Dropout(0.1)(lstm_out)

    attention_mul = self_attention_3d_block(drop)

    dense_out = Dense(units=1)(attention_mul)
    activation = Activation("linear")(dense_out)

    # output = Dense(INPUT_DIM, activation='sigmoid', name='output')(attention_mul)

    model = Model(input=[inputs], output=activation)

    return model

def create_attention_layer_model(timesteps, input_dims, lstm_units=100):

    inputs = Input(shape=(timesteps, input_dims,))
    lstm_out = LSTM(units=lstm_units, return_sequences=True)(inputs)
    drop = Dropout(0.1)(lstm_out)

    merged = Attention(timesteps)(drop)

    drop = Dropout(0.1)(merged)
    dense_out = Dense(units=1)(drop)
    activation = Activation("linear")(dense_out)

    # output = Dense(INPUT_DIM, activation='sigmoid', name='output')(attention_mul)

    model = Model(input=[inputs], output=activation)

    return model

def create_no_attention_model(timesteps, input_dims, lstm_units=100):

    inputs = Input(shape=(timesteps, input_dims,))
    lstm_out = LSTM(units=lstm_units, return_sequences=False)(inputs)

    drop = Dropout(0.1)(lstm_out)
    drop = Dropout(0.1)(drop) # to be fair with attention layer

    dense_out = Dense(units=1)(drop)
    activation = Activation("linear")(dense_out)

    # output = Dense(INPUT_DIM, activation='sigmoid', name='output')(attention_mul)

    model = Model(input=[inputs], output=activation)

    return model