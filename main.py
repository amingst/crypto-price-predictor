import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM


def load_data(filename, num_days):
    raw_data = pd.read_csv(filename, dtype=float).values

    for x in range(0, raw_data.shape[0]):
        for y in range(0, raw_data.shape[1]):
            if (raw_data[x][y] == 0):
                raw_data[x][y] = raw_data[x-1][y]

    data = raw_data.tolist()

    data_3D = []
    for index in range(len(data) - num_days):
        data_3D.append(data[index: index + num_days])

    print(data_3D)


load_data("./data/bitcoin_historical.csv", 50)
