from abc import ABC, abstractmethod
import numpy as np
from tqdm.autonotebook import tqdm

from pytsmp import utils


class MatrixProfile(ABC):
    """
    The base class for matrix profile computation. This is an abstract class, you cannot instantiate from this class.
    """
    def __init__(self, ts1, ts2=None, window_size=None, exclusion_zone=1/2, verbose=True, s_size=1):
        """
        Base constructor.

        :param ts1: Time series for calculating the matrix profile.
        :type ts1: numpy array
        :param ts2: A second time series to compute matrix profile with respect to ts1. If None, ts1 will be used.
        :type ts2: numpy array
        :param int window_size: Subsequence length, must be a positive integer less than the length of both ts1 and ts2
        :param float exclusion_zone: Exclusion zone, the length of exclusion zone is this number times window_size,
                                     centered at the point of interest.
                                     Must be non-negative. This parameter will be ignored if ts2 is not None.
        :param bool verbose: To be written
        :param float s_size: Ratio of random calculations performed for anytime algorithms. Must be between 0 and 1,
                             1 means calculate all, and 0 means none. This parameter will be ignored if the algorithm
                             is not anytime.
        :raises: ValueError: If the input is invalid.
        """
        self.ts1 = np.copy(ts1).astype("float64")
        self.ts2 = np.copy(ts2).astype("float64") if ts2 is not None else np.copy(ts1)
        if type(window_size) == int and 0 < window_size <= min(len(self.ts1), len(self.ts2)):
            self.window_size = window_size
        else:
            raise ValueError("Incorrect window size specified.")
        if exclusion_zone >= 0:
            self.ez = exclusion_zone
            self.exclusion_zone = round(window_size * exclusion_zone + 1e-5)
        else:
            raise ValueError("Exclusion zone must be non-negative.")
        self.verbose = bool(verbose)
        if 0 < s_size <= 1:
            self.s_size = s_size
        else:
            raise ValueError("s_size must be between 0 and 1.")

        self._same_ts = ts2 is None
        self._matrix_profile = np.full((len(self.ts1) - self.window_size + 1,), np.inf)
        self._index_profile = np.full((len(self.ts1) - self.window_size + 1,), np.nan)

        self._compute_matrix_profile()

    @property
    @abstractmethod
    def is_anytime(self):
        """
        A property stating whether the algorithm for computing the matrix profile
        in this class is an anytime algorithm.

        :return: whether the algorithm in this class is an anytime algorithm.
        :rtype: bool
        """
        return False

    @property
    @abstractmethod
    def _iterator(self):
        """
        The iterator to use in the computation of matrix profile. Defined separately to accomodate
        resume computation of anytime algorithms.

        :return: Iterator used in the computation of matrix profile.
        :rtype: iterator
        """
        return NotImplementedError

    def get_matrix_profile(self):
        """
        Get the matrix profile.

        :return: The matrix profile.
        :rtype: numpy array, of shape (len(ts1)-windows_size+1,)
        """
        return np.copy(self._matrix_profile)

    def get_index_profile(self):
        """
        Get the index profile.

        :return: The index profile.
        :rtype: numpy array, of shape (len(ts1)-windows_size+1,)
        """
        return np.copy(self._index_profile)

    def get_profiles(self):
        """
        Get the matrix profile and the index profile.

        :return: The matrix profile and the index profile.
        :rtype: 2 numpy arrays, both of shape (len(ts1)-window_size+1,)
        """
        return self.get_matrix_profile(), self.get_index_profile()

    @abstractmethod
    def _compute_matrix_profile(self):
        """
        Compute the matrix profile using the method indicated by the class.

        :return: None.
        """
        raise NotImplementedError

    def _elementwise_min(self, D, idx):
        """
        Subroutine for calculating elementwise min and min_index for matrix profile updates.

        :param D: Distance profile for update.
        :type D: numpy array
        :param int idx: Index (of ts2) corresponding to the distance profile.
        """
        if self._same_ts:
            lower_ez_bound = max(0, idx - self.exclusion_zone)
            upper_ez_bound = min(len(self.ts2), idx + self.exclusion_zone) + 1
            D[lower_ez_bound: upper_ez_bound] = np.inf
        self._index_profile[self._matrix_profile > D] = idx
        self._matrix_profile = np.minimum(self._matrix_profile, D)

    def update_ts1(self, pt):
        """
        Update the time-series ts1 with a new data point at the end of the series. If doing self-join (ts1 == ts2),
        both series will be updated.

        :param float pt: The value of the new data point, to be attached to the end of ts1.
        """
        self.ts1 = np.append(self.ts1, pt)
        if self._same_ts:
            self.ts2 = np.copy(self.ts1)
        s = self.ts1[-self.window_size:]
        idx = len(self.ts1) - self.window_size
        D = utils.mass(s, self.ts2)
        if self._same_ts:
            lower_ez_bound = max(0, idx - self.exclusion_zone)
            upper_ez_bound = min(len(self.ts2), idx + self.exclusion_zone) + 1
            D[lower_ez_bound:upper_ez_bound] = np.inf
            self._index_profile[self._matrix_profile > D[:-1]] = idx
            self._matrix_profile = np.minimum(self._matrix_profile, D[:-1])
        min_idx = np.argmin(D)
        self._index_profile = np.append(self._index_profile, min_idx)
        self._matrix_profile = np.append(self._matrix_profile, D[min_idx])

    def update_ts2(self, pt):
        """
        Update the time-series ts2 with a new data point at the end of the series. If doing self-join (ts1 == ts2),
        both series will be updated.

        :param float pt: The value of the new data point, to be attached to the end of ts2.
        """
        if self._same_ts:
            self.update_ts1(pt)
        else:
            self.ts2 = np.append(self.ts2, pt)
            s = self.ts2[-self.window_size:]
            idx = len(self.ts2) - self.window_size
            D = utils.mass(s, self.ts1)
            self._elementwise_min(D, idx)


class STAMP(MatrixProfile):
    """
    Class for the calculation of matrix profile using STAMP algorithm. See [MP1]_ for more details.

    .. [MP1] C.C.M. Yeh, Y. Zhu, L. Ulanova, N. Begum, Y. Ding, H.A. Dau, D. Silva, A. Mueen and E. Keogh.
       "Matrix profile I: All pairs similarity joins for time series: A unifying view that includes
       motifs, discords and shapelets". IEEE ICDM 2016.

    :param ts1: Time series for calculating the matrix profile.
    :type ts1: numpy array
    :param ts2: A second time series to compute matrix profile with respect to ts1. If None, ts1 will be used.
    :type ts2: numpy array
    :param int window_size: Subsequence length, must be a positive integer less than the length of both ts1 and ts2.
    :param float exclusion_zone: Exclusion zone, the length of exclusion zone is this number times window_size,
                                 centered at the point of interest.
                                 Must be non-negative. This parameter will be ignored if ts2 is not None.
    :param bool verbose: To be written
    :param float s_size: Ratio of random calculations performed for anytime algorithms. Must be between 0 and 1,
                         1 means calculate all, and 0 means none.
    :raises: ValueError: If the input is invalid.
    """
    def __init__(self, ts1, ts2=None, window_size=None, exclusion_zone=1/2, verbose=True, s_size=1):
        super().__init__(ts1, ts2, window_size, exclusion_zone, verbose, s_size)

    @property
    def is_anytime(self):
        return True

    @property
    def _iterator(self):
        idxes = np.random.permutation(range(len(self.ts2) - self.window_size + 1))
        idxes = idxes[:round(self.s_size * len(idxes) + 1e-5)]
        if self.verbose:
            _iterator = tqdm(idxes)
        else:
            _iterator = idxes
        return _iterator

    def _compute_matrix_profile(self):
        """
        Compute the matrix profile using STAMP.
        """
        try:
            for n_iter, idx in enumerate(self._iterator):
                D = utils.mass(self.ts2[idx: idx+self.window_size], self.ts1)
                self._elementwise_min(D, idx)
        except KeyboardInterrupt:
            if self.verbose:
                tqdm.write("Calculation interrupted at iteration {}. Approximate result returned.".format(n_iter))


class STOMP(MatrixProfile):
    """
    Class for the calculation of matrix profile using STOMP algorithm. This is faster than STAMP (actually the
    fastest known algorithm), but is not an anytime algorithm. See [MP2]_ for more details.

    .. [MP2] Y. Zhu, Z. Zimmerman, N.S. Senobari, C.C.M. Yeh, G. Funning, A. Mueen, P. Berisk and E. Keogh.
       "Matrix Profile II: Exploiting a Novel Algorithm and GPUs to Break the One Hundred Million
       Barrier for Time Series Motifs and Joins". IEEE ICDM 2016.

    :param ts1: Time series for calculating the matrix profile.
    :type ts1: numpy array
    :param ts2: A second time series to compute matrix profile with respect to ts1. If None, ts1 will be used.
    :type ts2: numpy array
    :param int window_size: Subsequence length, must be a positive integer less than the length of both ts1 and ts2.
    :param float exclusion_zone: Exclusion zone, the length of exclusion zone is this number times window_size,
                                 centered at the point of interest.
                                 Must be non-negative. This parameter will be ignored if ts2 is not None.
    :param bool verbose: To be written
    :param float s_size: This parameter will be ignored by STOMP since it is not an anytime algorithm.
    :raises: ValueError: If the input is invalid.
    """
    def __init__(self, ts1, ts2=None, window_size=None, exclusion_zone=1/2, verbose=True, s_size=1):
        super().__init__(ts1, ts2, window_size, exclusion_zone, verbose, s_size)

    @property
    def is_anytime(self):
        return False

    @property
    def _iterator(self):
        idxes = range(1, len(self.ts2) - self.window_size + 1)
        if self.verbose:
            _iterator = tqdm(idxes)
        else:
            _iterator = idxes
        return _iterator

    def _compute_matrix_profile(self):
        """
        Compute the matrix profile using STOMP.
        """
        mu_T, sigma_T = utils.rolling_avg_sd(self.ts1, self.window_size)
        QT = utils.sliding_dot_product(self.ts2[:self.window_size], self.ts1)
        if self._same_ts:
            mu_Q, sigma_Q = mu_T, sigma_T
            TQ = np.copy(QT)
        else:
            mu_Q, sigma_Q = utils.rolling_avg_sd(self.ts2, self.window_size)
            TQ = utils.sliding_dot_product(self.ts1[:self.window_size], self.ts2)
        D = utils.calculate_distance_profile(QT, self.window_size, mu_Q[0], sigma_Q[0], mu_T, sigma_T)
        if self._same_ts:
            lower_ez_bound = 0
            upper_ez_bound = min(len(self.ts2), self.exclusion_zone) + 1
            D[lower_ez_bound:upper_ez_bound] = np.inf
        self._matrix_profile = np.copy(D)
        self._index_profile = np.zeros((len(self.ts1) - self.window_size + 1,))
        for idx in self._iterator:
            QT[1:] = QT[:len(self.ts1)-self.window_size] - self.ts1[:len(self.ts1)-self.window_size] * self.ts2[idx-1] \
                     + self.ts1[self.window_size:] * self.ts2[idx + self.window_size - 1]
            QT[0] = TQ[idx]
            D = utils.calculate_distance_profile(QT, self.window_size, mu_Q[idx], sigma_Q[idx], mu_T, sigma_T)
            self._elementwise_min(D, idx)


class SCRIMP(MatrixProfile):
    """
    Class for the calculation of matrix profile using SCRIMP algorithm. This is faster than STAMP (slightly slower
    than STOMP), and is also an anytime algorithm. See [MP3]_ for more details.

    .. [MP3] Y. Zhu, C.C.M. Yeh, Z. Zimmerman, K. Kamgar and E. Keogh.
       "Matrix Proﬁle XI: SCRIMP++: Time Series Motif Discovery at Interactive Speed". IEEE ICDM 2018.

    :param ts1: Time series for calculating the matrix profile.
    :type ts1: numpy array
    :param ts2: A second time series to compute matrix profile with respect to ts1. If None, ts1 will be used.
    :type ts2: numpy array
    :param int window_size: Subsequence length, must be a positive integer less than the length of both ts1 and ts2.
    :param float exclusion_zone: Exclusion zone, the length of exclusion zone is this number times window_size,
                                 centered at the point of interest.
                                 Must be non-negative. This parameter will be ignored if ts2 is not None.
    :param bool verbose: To be written
    :param float s_size: This parameter will be ignored by STOMP since it is not an anytime algorithm.
    :raises: ValueError: If the input is invalid.
    """
    def __init__(self, ts1, ts2=None, window_size=None, exclusion_zone=1/2, verbose=True, s_size=1):
        super().__init__(ts1, ts2, window_size, exclusion_zone, verbose, s_size)

    @property
    def is_anytime(self):
        return True

    @property
    def _iterator(self):
        if self._same_ts:
            idxes = np.random.permutation(range(self.exclusion_zone + 1,
                                                len(self.ts2) - self.window_size + 1))
        else:
            idxes = np.random.permutation(range(-len(self.ts1) + self.window_size,
                                                len(self.ts2) - self.window_size + 1))
            idxes = range(-len(self.ts1) + self.window_size,
                                                len(self.ts2) - self.window_size + 1)
        idxes = idxes[:round(self.s_size * len(idxes) + 1e-5)]
        if self.verbose:
            _iterator = tqdm(idxes)
        else:
            _iterator = idxes
        return _iterator

    def _compute_matrix_profile(self):
        """
        Compute the matrix profile using SCRIMP.
        """
        n1 = len(self.ts1)
        n2 = len(self.ts2)
        mu_T, sigma_T = utils.rolling_avg_sd(self.ts1, self.window_size)
        if self._same_ts:
            mu_Q, sigma_Q = mu_T, sigma_T
        else:
            mu_Q, sigma_Q = utils.rolling_avg_sd(self.ts2, self.window_size)
        for k in self._iterator:
            if k >= 0:
                # compute diagonals starting from a slot in first column
                q = self.ts2[k:k+n1] * self.ts1[:n2-k]
                q = utils.rolling_sum(q, self.window_size)
                D = utils.calculate_distance_profile(q, self.window_size, mu_Q[k:k+len(q)], sigma_Q[k:k+len(q)],
                                                     mu_T[:len(q)], sigma_T[:len(q)])
                self._index_profile[:len(q)] = np.where(D < self._matrix_profile[:len(q)],
                                                np.arange(k, k + len(q)), self._index_profile[:len(q)])
                self._matrix_profile[:len(q)] = np.minimum(D, self._matrix_profile[:len(q)])
                if self._same_ts:
                    self._index_profile[k:k+len(q)] = np.where(D < self._matrix_profile[k:k+len(q)],
                                                np.arange(len(q)), self._index_profile[k:k+len(q)])
                    self._matrix_profile[k:k+len(q)] = np.minimum(D, self._matrix_profile[k:k+len(q)])
            else:
                # compute diagonals starting from a slot in first row
                k = -k
                q = self.ts2[:n1-k] * self.ts1[k:k+n2]
                q = utils.rolling_sum(q, self.window_size)
                D = utils.calculate_distance_profile(q, self.window_size, mu_Q[:len(q)], sigma_Q[:len(q)],
                                                     mu_T[k:k+len(q)], sigma_T[k:k+len(q)])
                self._index_profile[k:k+len(q)] = np.where(D < self._matrix_profile[k:k+len(q)],
                                                        np.arange(len(q)), self._index_profile[k:k+len(q)])
                self._matrix_profile[k:k+len(q)] = np.minimum(D, self._matrix_profile[k:k+len(q)])


