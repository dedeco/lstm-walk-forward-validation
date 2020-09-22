# multivariate multi-step encoder-decoder lstm
import logging
from math import sqrt

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from numpy import array
from numpy import split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

N_TRAIN_HOURS = 365 * 4 * 8
N_TEST_HOURS = 365 * 4 * 2


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


def inverse_input(history, n_input):
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    input_x = data[-n_input:, :]
    return input_x.reshape((1, input_x.shape[0], input_x.shape[1]))


# split a multivariate dataset into train/test sets
def split_dataset(data, output):
    # split into standard days
    train, test = data[:-N_TEST_HOURS], data[-N_TEST_HOURS:]
    logging.info("split_dataset: train {} test {}".format(train.shape, test.shape))
    # restructure into windows in day predict
    if train.shape[0] % output != 0:
        slice_left = train.shape[0] - (train.shape[0] % output)
        train, test = data[:slice_left], data[slice_left:]
    if test.shape[0] % output != 0:
        train, test = data[:slice_left], data[slice_left:-(test.shape[0] % output)]
    return train, test


def restructure_into_daily_data(train, test, output):
    # restructure into windows of daily data (every 6 hours have a new data: 4 a day)
    logging.info("restructure_into_daily_data: train {} test {}".format(train.shape, test.shape))
    train = array(split(train, len(train) / output))
    test = array(split(test, len(test) / output))
    logging.info("restructure_into_daily_data: train {} test {}".format(train.shape, test.shape))
    return train, test


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        logging.info("evaluate_forecasts: real {} predict {}".format(actual.shape, predicted.shape))
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    logging.info('summarize_scores: %s: [%.3f] %s' % (name, score, s_scores))


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return array(X), array(y)


# train the model
def build_model(train, n_input, n_out):
    # prepare data
    train_x, train_y = to_supervised(train, n_input, n_out)
    # define parameters
    verbose, epochs, batch_size = 0, 1000, 500
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(300, input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(300, return_sequences=True))
    model.add(TimeDistributed(Dense(200)))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=False)
    return model


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return data, yhat
