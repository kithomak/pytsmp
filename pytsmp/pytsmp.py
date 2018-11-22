import numpy as np
from tqdm import tqdm

from pytsmp import utils


class MatrixProfile:
    """
    The base class for matrix profile computation. **Do not** initialize from this class directly.
    """
    _is_anytime = False

    def __init__(self, ts1, ts2=None, window_size=None, exclusion_zone=1/2, verbose=True, s_size=1):
        """
        Base initialize.

        :param ts1: Time series for calculating the matrix profile.
        :type ts1: numpy array
        :param ts2: A second time series to compute matrix profile with respect to ts1. If None, ts1 will be used.
        :type ts2: numpy array
        :param int window_size: Subsequence length, must be a positive integer less than the length of both ts1 and ts2
        :param float exclusion_zone: Exclusion zone, the length of exclusion zone is this number times window_size.
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
            self.s_length = round(s_size * (len(self.ts2) - self.window_size + 1) + 1e-5)
        else:
            raise ValueError("s_size must be between 0 and 1.")

        self._same_ts = ts2 is None
        self._matrix_profile = np.full((len(self.ts1) - self.window_size + 1,), np.inf)
        self._index_profile = np.full((len(self.ts1) - self.window_size + 1,), np.nan)
        if self._is_anytime:
            idxes = np.random.permutation(range(len(self.ts2) - self.window_size + 1))[:self.s_length]
        else:
            idxes = np.arange(len(self.ts2) - self.window_size + 1)
        if self.verbose:
            self._iterator = tqdm(idxes)
        else:
            self._iterator = idxes

        self._compute_matrix_profile()

    def get_profiles(self):
        """
        Get the matrix profile and the index profile.

        :return: The matrix profile and the index profile.
        :rtype: numpy arrays, both of shape (len(ts1)-window_size+1,)
        """
        return np.copy(self._matrix_profile), np.copy(self._index_profile)

    def _compute_matrix_profile(self):
        """
        Compute the matrix profile using the method indicated by the class.
        Raise NotImplementedError in the base class.

        :return: None.
        """
        raise NotImplementedError("Please initialize from one of the STAMP, STOMP or SCRIMP class. " +
                                  "Do not initialize from the MatrixProfile class directly.")

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
            # self._elementwise_min(D[:-1], idx)
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

    .. [MP1] Yeh CCM, Zhu Y, Ulanova L, Begum N, Ding Y, Dau HA, et al. "Matrix profile I: All
       pairs similarity joins for time series: A unifying view that includes motifs, discords and
       shapelets". *Proc - IEEE Int Conf Data Mining, ICDM. 2017;1317–22*.
       (http://www.cs.ucr.edu/~eamonn/MatrixProfile.html)

    :param ts1: Time series for calculating the matrix profile.
    :type ts1: numpy array
    :param ts2: A second time series to compute matrix profile with respect to ts1. If None, ts1 will be used.
    :type ts2: numpy array
    :param int window_size: Subsequence length, must be a positive integer less than the length of both ts1 and ts2.
    :param float exclusion_zone: Exclusion zone, the length of exclusion zone is this number times window_size.
                                 Must be non-negative. This parameter will be ignored if ts2 is not None.
    :param bool verbose: To be written
    :param float s_size: Ratio of random calculations performed for anytime algorithms. Must be between 0 and 1,
                         1 means calculate all, and 0 means none. This parameter will be ignored if the algorithm
                         is not anytime.
    :raises: ValueError: If the input is invalid.
    """
    _is_anytime = True

    def __init__(self, ts1, ts2=None, window_size=None, exclusion_zone=1/2, verbose=True, s_size=1):
        super().__init__(ts1, ts2, window_size, exclusion_zone, verbose, s_size)

    def _compute_matrix_profile(self):
        """
        Compute the matrix profile using STAMP.
        """
        for idx in self._iterator:
            D = utils.mass(self.ts2[idx: idx+self.window_size], self.ts1)
            self._elementwise_min(D, idx)


