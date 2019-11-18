import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM


def remove_zeros(data_in):
    data_out = data_in

    for x in range(0, data_out.shape[0]):
        for y in range(0, data_out.shape[1]):
            if (data_out[x][y] == 0):
                data_out[x][y] = data_out[x-1][y]

    return data_out


def to_array3D(data_in, num_days):
    data_out = []

    for index in range(len(data_in) - num_days):
        data_out.append(data_in[index: index + num_days])

    return data_out


def normalize_data(data_in):
    data_init = np.array(data_in)
    normalized = np.zeros_like(data_init)
    normalized[:, 1:, :] = data_init[:, 1:, :] / data_init[:, 0:1, :] - 1

    unnormalized = data_init[2400:int(normalized.shape[0] + 1), 0:1, 20]

    return normalized, unnormalized


def load_data(filename, num_days):
    raw_data = pd.read_csv(filename, dtype=float).values
    raw_data = remove_zeros(raw_data)

    data = raw_data.tolist()
    data_3D = to_array3D(data, num_days)

    normalized, unnormalized = normalize_data(data_3D)


load_data("./data/bitcoin_historical.csv", 50)
