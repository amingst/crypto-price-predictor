def to_array3D(data_in, num_days):
    """ Takes an input array and the number of days and converts to a 3D array.

    Parameters
    ----------
    data_in: list, required
            The input data in a 2D array.
    num_days: int, required
            The number of days to take from the input data.
    
    Returns
    -------
    data_out: list
            An output list that is 3D.

    """
    data_out = []

    for index in range(len(data_in) - num_days):
        data_out.append(data_in[index: index + num_days])

    return data_out