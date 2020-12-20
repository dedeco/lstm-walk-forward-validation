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

    for cell in [108 * 2 + 1]:  # [50, 100, 150, 200, 250, 300, 400]:
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

                            # history is a list by window size
                            history_scaled = [x for x in train_scaled[:n_input, :, :]]
                            history = [x for x in train[:n_input, :, :]]

                            # TRAIN walk-forward validation
                            predictions = list()
                            predictions_inverted = list()
                            for i in range(len(train_scaled)):
                                # predict the window
                                if i % 1000 == 0:
                                    logging.info('TRAIN walk-forward validation step {}'.format(i))

                                yhat_sequence = forecast(model, history_scaled, n_input)

                                # store the predictions
                                predictions.append(yhat_sequence)

                                # real observation
                                data = array(train_scaled[i, :])
                                data[:, 0] = yhat_sequence.reshape(data.shape[0])
                                inverse_data = scaler.inverse_transform(data)
                                yhat_sequence_inversed = inverse_data[:, 0]

                                predictions_inverted.append(yhat_sequence_inversed)

                                # get real observation and add to history for predicting the next week
                                history_scaled.append(train_scaled[i, :])
                                history.append(train[i, :])

                            # evaluate predictions on train
                            predictions_inverted = array(predictions_inverted)
                            score, scores = evaluate_forecasts(train[:, :, 0], predictions_inverted)

                            # summarize scores on test
                            summarize_scores('lstm', score, scores)

                            rmse = sqrt(mean_squared_error(train[:, :, 0], predictions_inverted))
                            logging.info('Train RMSE: %.3f' % rmse)

                            errors_abs = abs((train[:, :, 0] - predictions_inverted))
                            logging.info(
                                'Train ABS MIN: {} MAX: {} MEAN: {} STD: {}'.format(errors_abs.min()
                                                                                    , errors_abs.max()
                                                                                    , errors_abs.mean()
                                                                                    , errors_abs.std())
                            )

                            # history is a list by window size
                            history_scaled = [x for x in train_scaled]
                            history = [x for x in train]

                            # TEST walk-forward validation
                            predictions = list()
                            predictions_inverted = list()
                            for i in range(len(test_scaled)):
                                # predict the window

                                if i % 1000 == 0:
                                    logging.info('TEST walk-forward validation step {}'.format(i))

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


if __name__ == "__main__":
    run()
