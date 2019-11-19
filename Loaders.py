import numpy as np
import pandas as pd

import Manipulators
from Array3D import to_array3D


def load_data(filename, num_days):
    # TODO: Refactor data splitting
    # TODO: Add docs

    # Load raw data and remove zero values
    raw_data = pd.read_csv(filename, dtype=float).values
    raw_data = Manipulators.remove_zeros(raw_data)

    # Convert the raw data to a list and then to a 3D array
    data = raw_data.tolist()
    data_3D = to_array3D(data, num_days)

    # Normalize data and get normalized and unnormalized values
    normalized, unnormalized = Manipulators.normalize_data(data_3D)

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

    # Get the forecast window
    forecast = num_days - 1

    return x_train, y_train, x_test, y_test, prev_y, unnormalized, forecast
