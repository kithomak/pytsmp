import numpy as np

from pytsmp import utils


def naive_sliding_dot_product(Q, T):
    """
    Calculate the sliding dot product in a naive way. Used for testing.

    :param Q: The query series.
    :type Q: numpy array
    :param T: A time series to query on its subsequences.
    :type T: numpy array
    :return: The dot product between Q and all subsequences in T.
    :rtype: numpy array, of shape (len(T)-len(Q)+1,)
    :raises: ValueError: If len(T) < len(Q).
    """
    result = np.empty((len(T) - len(Q) + 1,))
    for i in range(len(T)-len(Q)+1):
        result[i] = np.dot(T[i:i+len(Q)], Q)
    return result

def naive_distance_profile(Q, T):
    """
    Calculate the distance profile in a naive way. Used for testing.

    :param Q: The query series.
    :type Q: numpy array
    :param T: A time series to query on its subsequences.
    :type T: numpy array
    :return: The dot product between Q and all subsequences in T.
    :rtype: numpy array, of shape (len(T)-len(Q)+1,)
    :raises: ValueError: If len(T) < len(Q).
    """
    m = len(Q)
    result = np.empty((len(T) - len(Q) + 1,))
    normalized_Q = utils.z_normalize(Q)
    for i in range(len(T)-len(Q)+1):
        result[i] = np.linalg.norm(normalized_Q - utils.z_normalize(T[i:i+m]))
    return result


