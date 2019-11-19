from Loaders import load_data

from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM


def init_model(forecast, dropout, activation, loss, optimizer, x_train):
    model = Sequential()

    model.add(Bidirectional(LSTM(forecast, return_sequences=True),
                            input_shape=(forecast, x_train.shape[-1]),))
    model.add(Dropout(dropout))

    model.add(Bidirectional(LSTM((forecast*2), return_sequences=True)))
    model.add(Dropout(dropout))

    model.add(Bidirectional(LSTM(forecast, return_sequences=False)))

    model.add(Dense(units=1))

    model.add(Activation(activation))

    model.compile(loss=loss, optimizer=optimizer)

    return model


def main():
    x_train, y_train, x_test, y_test, prev_y, unnormalized, forecast = load_data(
        "../data/bitcoin_historical.csv", 50)

    model = init_model(forecast, 0.2, 'linear', 'mse', 'adam', x_train)s


if __name__ == "__main__":
    main()
