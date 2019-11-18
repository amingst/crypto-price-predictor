from Loaders import load_data

from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

def main():
    x_train, y_train, x_test, y_test, prev_y, unnormalized, w_size = load_data(
    "../data/bitcoin_historical.csv", 50)

if __name__ == "__main__":
    main()