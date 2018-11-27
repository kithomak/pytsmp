import numpy as np

# from pytsmp import misc


def rolling_average(t, window_size):
    """
    Compute rolling average of a given time series t, with given window size. Raise ValueError
    if the window_size is larger than the length of t.

    Note: This algorithm is not numerically stable for calculating the rolling statistics
    of large numbers.

    :param t: Time series to calculate rolling average.
    :type t: numpy array
    :param window_size: Window size
    :type window_size: int
    :return: The rolling average
    :rtype: numpy array, of shape (len(t) - window_size + 1,)
    :raises: ValueError: If len(T) < window_size.
    """
    if len(t) < window_size:
        raise ValueError("Window size should be smaller than the length of time series.")
    cumsum = np.cumsum(np.insert(t, 0, 0) / window_size)
    return cumsum[window_size:] - cumsum[:-window_size]


def rolling_avg_sd(t, window_size):
    """
    Compute rolling average and standard derivation of a given time series t, with given window size.
    Raise ValueError if the window_size is larger than the length of t.

    Note: This algorithm is not numerically stable for calculating the rolling statistics
    of large numbers.

    :param t: Time series to calculate rolling average.
    :type t: numpy array
    :param window_size: window size
    :type window_size: int
    :return: The rolling average and the rolling sd
    :rtype: 2 numpy arrays, of shape (len(t) - window_size + 1,)
    :raises: ValueError: If len(t) < window_size.
    """
    if len(t) < window_size:
        raise ValueError("Window size should be smaller than the length of time series.")
    cumsum = np.cumsum(np.insert(t, 0, 0))
    cumsum_squared = np.cumsum(np.insert(t, 0, 0) ** 2)
    cummean = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    cummean_squared = (cumsum_squared[window_size:] - cumsum_squared[:-window_size]) / window_size
    return cummean, np.sqrt(np.maximum(cummean_squared - (cummean ** 2), 0))


def z_normalize(t):
    """
    Calculate the z-normalization of the time series.

    :param t: Time series to calculate the z-normalization.
    :type t: numpy array
    :return: The z-normalized time-series.
    :rtype: numpy array
    :raises: ValueError: If given a constant sequence.
    """
    std = np.std(t)
    if std == 0:
        raise ValueError("Cannot normalize a constant series.")
    else:
        return (t - np.mean(t)) / std


def sliding_dot_product(Q, T):
    """
    Sliding dot product calculation using FFT. See Table 1 in the Matrix Profile I paper for details.

    The given query Q must be shorter than the time series T, otherwise ValueError will be raised.
    The returning array has length len(T)-len(Q)+1.

    :param Q: The query series.
    :type Q: numpy array
    :param T: A time series to query on its subsequences.
    :type T: numpy array
    :return: The dot product between Q and all subsequences in T.
    :rtype: numpy array, of shape (len(T)-len(Q)+1,)
    :raises: ValueError: If len(T) < len(Q).
    """
    n = len(T)
    m = len(Q)
    if n < m:
        raise ValueError("T should be a series at least as long as Q")
    T_a = np.pad(T, (0, n % 2), 'constant')
    Q_r = Q[::-1]
    Q_ra = np.pad(Q_r, (0, n + (n % 2) - m), 'constant')
    Q_raf = np.fft.rfft(Q_ra)
    T_af = np.fft.rfft(T_a)
    QT = np.fft.irfft(Q_raf * T_af)
    return QT[m-1:n]


def calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T):
    """
    Subroutine for calculating the distance profile of a time series T with respect to a query Q,
    if the sliding dot product between the time series and the query,
    the moving average, moving sd of the time series and the mean and sd of the query are known.
    Note that T and Q are not required in the input.

    :param QT: The sliding dot product of T and Q.
    :type QT: numpy array
    :param int m: Length of Q.
    :param int mean_Q: Mean of Q.
    :param int sigma_Q: Standard derivation of Q.
    :param mean_T: The rolling mean (with window size m) of T
    :type mean_T: numpy array
    :param sigma_T: The rolling sd (with window size m) of T
    :type sigma_T: numpy array
    :return: The distance profile of T with respect to Q.
    :rtype: numpy array, of shape (len(T)-len(Q)+1,)
    """
    # Take max with 0 before the sqaure root to eliminate complex numbers resulted in floating point error.
    D = np.sqrt(np.maximum(2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)), 0))
    return D


def mass(Q, T):
    """
    Mueen's algorithm for similarity search (MASS) algorithm. See Table 2 in the Matrix
    Profile I paper for details.

    The given query Q must be shorter than the time series T, otherwise ValueError will be raised.
    The returning array has length len(T)-len(Q)+1.

    :param Q: The query series.
    :type Q: numpy array
    :param T: A time series to query on its subsequences.
    :type T: numpy array
    :return: The distance profile D of Q and all subsequences in T.
    :rtype: numpy array, of shape (len(T)-len(Q)+1,)
    :raises: ValueError: If len(T) < len(Q).
    """
    n = len(T)
    m = len(Q)
    if n < m:
        raise ValueError("T should be a series at least as long as Q")
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    return D


