from Loaders import load_data

import time
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM


class Model:
    # TODO: Move to seperate file
    # TODO: Add docs
    def __init__(self):
        self.model = Sequential()

    def train_time(self):
        print(self.time_to_train)

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

    def train(self, x_train, y_train, batch_size, epochs, validation):
        # TODO: Refactor time_to_train
        start_time = time.time()

        self.model.fit(x_train, y_train, batch_size=batch_size,
                       epochs=epochs, validation_split=validation)

        self.time_to_train = int(math.floor(time.time() - start_time))

    def test(self, x_test, y_test, unnormalized):
        y_predict = self.model.predict(x_test)

        y_test_actual = np.zeros_like(y_test)
        y_predict_actual = np.zeros_like(y_predict)

        for i in range(y_test.shape[0]):
            y = y_test[i]
            prediction = y_predict[i]
            y_test_actual[i] = (y+1)*unnormalized[i]
            y_predict_actual[i] = (prediction+1)*unnormalized[i]

        print(y_predict)
        print(y_test_actual)
        print(y_predict_actual)

        return y_predict, y_predict_actual, y_test_actual


def main():
    x_train, y_train, x_test, y_test, prev_y, unnormalized, forecast = load_data(
        "../data/bitcoin_historical.csv", 50)

    input_shape = (forecast, x_train.shape[-1])

    model = Model()
    model.build(forecast, 0.2, 'linear', 'mse', 'adam', input_shape)
    model.train(x_train, y_train, 1024, 5, .05)
    model.train_time()
    y_predict, y_predict_actual, y_test_actual = model.test(
        x_test, y_test, unnormalized)


if __name__ == "__main__":
    main()
