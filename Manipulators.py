import numpy as np

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

def normalize_data(data_in):
    """ Takes an input 3D array and normalizes the values.

    Parameters
    ----------
    data_in: list, required
            The input data that is a 3D array.
    
    Returns
    -------
    normalized: numpy array
            The normalized data.
    unnormalized: numpy array
            The unnormalized values, used for later testing.
    """
    data_init = np.array(data_in)
    normalized = np.zeros_like(data_init)
    normalized[:, 1:, :] = data_init[:, 1:, :] / data_init[:, 0:1, :] - 1

    unnormalized = data_init[2400:int(normalized.shape[0] + 1), 0:1, 20]

    return normalized, unnormalized