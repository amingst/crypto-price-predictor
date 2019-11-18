import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

# TODO: Move to seperate file


def remove_zeros(data_in):
    """ Takes an input dataframe and removes zero valued indecies.
    
    Parameters
    ----------
    data_in: pandas dataframe, required
            The input dataframe.
    
    Returns
    -------
    data_out: pandas dataframe
            The original dataframe with the zero valued indecies removed.
    """
    data_out = data_in

    for x in range(0, data_out.shape[0]):
        for y in range(0, data_out.shape[1]):
            if (data_out[x][y] == 0):
                data_out[x][y] = data_out[x-1][y]

    return data_out

# TODO: Move to seperate file


def to_array3D(data_in, num_days):
    data_out = []

    for index in range(len(data_in) - num_days):
        data_out.append(data_in[index: index + num_days])

    return data_out


# TODO: Move to seperate file
def normalize_data(data_in):
    data_init = np.array(data_in)
    normalized = np.zeros_like(data_init)
    normalized[:, 1:, :] = data_init[:, 1:, :] / data_init[:, 0:1, :] - 1

    unnormalized = data_init[2400:int(normalized.shape[0] + 1), 0:1, 20]

    return normalized, unnormalized


# TODO: Move to seperate file
# TODO: Refactor data splitting
def load_data(filename, num_days):
    # Load raw data and remove zero values
    raw_data = pd.read_csv(filename, dtype=float).values
    raw_data = remove_zeros(raw_data)

    # Convert the raw data to a list and then to a 3D array
    data = raw_data.tolist()
    data_3D = to_array3D(data, num_days)

    # Normalize data and get normalized and unnormalized values
    normalized, unnormalized = normalize_data(data_3D)

    # Split data on the pivot and grab training data
    pivot = round(0.9 * normalized.shape[0])
    training_data = normalized[:int(pivot), :]

    # Shuffle data
    np.random.shuffle(training_data)

    # Separate training data
    x_train = training_data[:, :-1]
    y_train = training_data[:, -1]
    y_train = y_train[:, 20]

    # Separate testing data
    x_test = normalized[int(pivot):, :-1]
    y_test = normalized[int(pivot):, 49, :]
    y_test = y_test[:, 20]

    # Get previous day's data
    prev_y = normalized[int(pivot):, 48, :]
    prev_y = prev_y[:, 20]

    # Get the window size
    w_size = num_days - 1

    return x_train, y_train, x_test, y_test, prev_y, unnormalized, w_size


x_train, y_train, x_test, y_test, prev_y, unnormalized, w_size = load_data(
    "./data/bitcoin_historical.csv", 50)
