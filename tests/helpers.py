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
    """
    m = len(Q)
    result = np.empty((len(T) - len(Q) + 1,))
    normalized_Q = utils.z_normalize(Q)
    for i in range(len(T)-len(Q)+1):
        result[i] = np.linalg.norm(normalized_Q - utils.z_normalize(T[i:i+m]))
    return result

def naive_matrix_profile(ts1, ts2=None, window_size=None, exclusion_zone=1/2):
    """
    Calculate the matrix profile in a naive way when ts1 != ts2. Used for testing.

    :param ts1: The query series.
    :type ts1: numpy array
    :param ts2: A time series to query on its subsequences. If None, then ts1 will be used.
    :type ts2: numpy array
    :param int window_size: The window size.
    :param float exclusion_zone: Exclusion zone, the length of exclusion zone is this number times window_size.
                                 Must be non-negative. This parameter will be ignored if ts2 is not None.
    :return: The matrix profile and index profile of ts1 and ts2.
    :rtype: numpy arrays, both of shape (len(T)-len(Q)+1,)
    """
    is_same_ts = False
    if ts2 is None:
        is_same_ts = True
        ts2 = np.copy(ts1)
        exclusion_zone = round(window_size * exclusion_zone + 1e-5)
    l1 = len(ts1) - window_size + 1
    l2 = len(ts2) - window_size + 1
    mat_profile = np.full((l1,), np.inf)
    ind_profile = np.full((l1,), np.nan)
    for i in range(l2):
        D = naive_distance_profile(ts2[i:i+window_size], ts1)
        if is_same_ts:
            lower_ez_bound = max(0, i - exclusion_zone)
            upper_ez_bound = min(len(ts2), i + exclusion_zone) + 1
            D[lower_ez_bound: upper_ez_bound] = np.inf
        ind_profile[mat_profile > D] = i
        mat_profile = np.minimum(D, mat_profile)
    return mat_profile, ind_profile


