import pandas

from math import sqrt

from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error

from helpers import split_dataset, build_model, forecast, evaluate_forecasts, summarize_scores, scale, \
    restructure_into_daily_data
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
    df = read_csv('pre-processed-in.csv',
                  index_col=0,
                  parse_dates=True)

    for n_input in [4, 8, 12, 16]:
        for n_out in range(1, 9):
            if (n_input == 4 and n_out in [1, 2, 3, 4, 5, 6, 7, 8]) or (n_input == 8 and n_out in [1, 2, 3, 4]):
                continue

            logging.info("Starting... input {} and output {}".format(n_input, n_out))

            try:

                logging.info("Training {} {}".format(n_input, n_out))

                train, test = split_dataset(df.values, n_out)

                # transform data
                scaler, train_scaled, test_scaled = scale(train, test)

                # restructure into weekly
                train_scaled, test_scaled = restructure_into_daily_data(train_scaled, test_scaled, n_out)
                train, test = restructure_into_daily_data(train, test, n_out)

                # fit model
                model = build_model(train_scaled, n_input, n_out)

                # history is a list of weekly data
                history_scaled = [x for x in train_scaled]
                history = [x for x in train]

                # walk-forward validation over each week
                predictions = list()
                predictions_inverted = list()
                for i in range(len(test_scaled)):
                    # predict the week
                    data_input, yhat_sequence = forecast(model, history_scaled, n_input)

                    # store the predictions
                    predictions.append(yhat_sequence)

                    # real observation
                    data = array(test_scaled[i, :])
                    data[:, 0] = yhat_sequence.reshape(data.shape[0])
                    inverse_data = scaler.inverse_transform(data)
                    yhat_sequence_inversed = inverse_data[:, 0]

                    predictions_inverted.append(yhat_sequence_inversed)

                    # print('predictions', yhat_sequence_inversed)

                    # get real observation and add to history for predicting the next week
                    history_scaled.append(test_scaled[i, :])
                    history.append(test[i, :])

                # evaluate predictions days for each week
                predictions_inverted = array(predictions_inverted)
                score, scores = evaluate_forecasts(test[:, :, 0], predictions_inverted)

                # summarize scores
                summarize_scores('lstm', score, scores)

                rmse = sqrt(mean_squared_error(test[:, :, 0], predictions_inverted))
                logging.info('Test RMSE: %.3f' % rmse)

                errors_abs = abs((test[:, :, 0] - predictions_inverted))
                logging.info(
                    'ABS MIN: {} MAX: {} MEAN: {} STD: {}'.format(errors_abs.min()
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
