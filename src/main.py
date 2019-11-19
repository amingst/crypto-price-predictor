from Loaders import load_data

import time
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM


class Model:
    # TODO: Move to seperate file
    # TODO: Add docs
    def __init__(self):
        self.model = Sequential()

    def build(self, forecast, dropout, activation, loss, optimizer, i_shape):
        self.model.add(Bidirectional(LSTM(forecast, return_sequences=True),
                                     input_shape=i_shape))
        self.model.add(Dropout(dropout))

        self.model.add(Bidirectional(
            LSTM((forecast*2), return_sequences=True)))
        self.model.add(Dropout(dropout))

        self.model.add(Bidirectional(LSTM(forecast, return_sequences=False)))

        self.model.add(Dense(units=1))

        self.model.add(Activation(activation))

        self.model.compile(loss=loss, optimizer=optimizer)


def main():
    x_train, y_train, x_test, y_test, prev_y, unnormalized, forecast = load_data(
        "../data/bitcoin_historical.csv", 50)

    input_shape = (forecast, x_train.shape[-1])

    model = Model()
    model.build(forecast, 0.2, 'linear', 'mse', 'adam', input_shape)


if __name__ == "__main__":
    main()
