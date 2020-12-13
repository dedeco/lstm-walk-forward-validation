import pandas
import numpy

from math import sqrt

from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error

from helpers import split_dataset, build_model, forecast, evaluate_forecasts, summarize_scores, scale, \
    restructure_into_daily_data, reshape_input_predict, to_supervised
from utils import plot_results, plot_scatter

N_TRAIN_HOURS = 365 * 4 * 8
N_TEST_HOURS = 365 * 4 * 2

import logging


def run():
    logging.basicConfig(format='%(message)s',
                        level=logging.INFO,
                        filename='results.log',
                        filemode='w')

    # load the new file
    df = read_csv('pre-processed-in-24-hours.csv',
                  index_col=0,
                  parse_dates=True)

    for cell in [108*2+1]:  # [50, 100, 150, 200, 250, 300, 400]:
        for epoch in [1000, ]:  # [1000, 2000, 3000, 4000, 5000]:
            for batch_size in [500, ]:  # [500, 1000, 1500]:
                for n_input in [1, 2, 4, 8, 12, 16]:
                    for n_out in range(1, 9):
                        logging.info(
                            "Starting... cell {0}, epoch {1}, batch_size {2}, input {3} and output {4}" \
                                .format(cell
                                        , epoch
                                        , batch_size
                                        , n_input
                                        , n_out))
                        try:
                            logging.info("Training {} {}".format(n_input, n_out))
                            # transform data
                            scaler, data_scaled = scale(df.values)

                            train, test = split_dataset(df.values, n_out)
                            train_scaled, test_scaled = split_dataset(data_scaled, n_out)

                            # restructure into window size
                            train_scaled, test_scaled = restructure_into_daily_data(train_scaled, test_scaled, n_out)
                            train, test = restructure_into_daily_data(train, test, n_out)

                            # fit model
                            model = build_model(train_scaled, n_input, n_out, cell, epoch, batch_size)

                            # after training measuring rsme
                            train_y, yhat_all_inversed = sumarize_rmse_training(model, n_input, n_out, scaler,
                                                                                train_scaled)

                            rmse = sqrt(mean_squared_error(train_y, yhat_all_inversed))
                            logging.info('Training RMSE: %.3f' % rmse)

                            # history is a list by window size
                            history_scaled = [x for x in train_scaled]
                            history = [x for x in train]

                            # walk-forward validation
                            predictions = list()
                            predictions_inverted = list()
                            for i in range(len(test_scaled)):
                                # predict the window
                                yhat_sequence = forecast(model, history_scaled, n_input)

                                # store the predictions
                                predictions.append(yhat_sequence)

                                # real observation
                                data = array(test_scaled[i, :])
                                data[:, 0] = yhat_sequence.reshape(data.shape[0])
                                inverse_data = scaler.inverse_transform(data)
                                yhat_sequence_inversed = inverse_data[:, 0]

                                predictions_inverted.append(yhat_sequence_inversed)

                                # get real observation and add to history for predicting the next week
                                history_scaled.append(test_scaled[i, :])
                                history.append(test[i, :])

                            # evaluate predictions on test
                            predictions_inverted = array(predictions_inverted)
                            score, scores = evaluate_forecasts(test[:, :, 0], predictions_inverted)

                            # summarize scores on test
                            summarize_scores('lstm', score, scores)

                            rmse = sqrt(mean_squared_error(test[:, :, 0], predictions_inverted))
                            logging.info('Test RMSE: %.3f' % rmse)

                            errors_abs = abs((test[:, :, 0] - predictions_inverted))
                            logging.info(
                                'Test ABS MIN: {} MAX: {} MEAN: {} STD: {}'.format(errors_abs.min()
                                                                                   , errors_abs.max()
                                                                                   , errors_abs.mean()
                                                                                   , errors_abs.std())
                            )

                            logging.info("predictions_inverted: {}".format(predictions_inverted.shape))
                            logging.info("test {}".format(test.shape))

                            data = {
                                'predict': predictions_inverted.reshape(
                                    predictions_inverted.shape[0] * predictions_inverted.shape[1]),
                                'real': test[:, :, 0].reshape(test[:, :, 0].shape[0] * test[:, :, 0].shape[1])}

                            data['time'] = df.index[-data["predict"].shape[0]:]

                            df_plot = pandas.DataFrame.from_dict(data)
                            df_plot.to_csv('plot_results_{0}_{1}.csv'.format(n_input, n_out))
                            plot_results(df_plot)
                            plot_scatter(df_plot)

                        except Exception as e:
                            logging.info(e)


def sumarize_rmse_training(model, n_input, n_out, scaler, train_scaled):
    train_x, train_y = to_supervised(train_scaled, n_input, n_out)
    train_yhat = model.predict(train_x, verbose=0)
    # inverse just the y to cal rsme
    data = numpy.random.uniform(0, 0, [train_y.shape[0] * train_y.shape[1], 108])
    data[:, 0] = train_yhat.reshape(train_y.shape[0] * train_y.shape[1])
    inverse_data = scaler.inverse_transform(data)
    yhat_all_inversed = inverse_data[:, 0]
    yhat_all_inversed = yhat_all_inversed.reshape(train_y.shape[0], train_y.shape[1])
    return train_y, yhat_all_inversed


if __name__ == "__main__":
    run()
