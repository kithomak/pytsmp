import pytest
import numpy as np

from pytsmp import utils
from tests import helpers


class TestRollingSum:
    def test_rolling_sum_window_too_large(self):
        with pytest.raises(ValueError):
            t = np.random.rand(500)
            ans = utils.rolling_sum(t, 1000)

    def test_rolling_sum_sanity1(self):
        t = np.random.rand(1000)
        ra = utils.rolling_sum(t, 1)
        assert np.allclose(t, ra), "rolling_sum_sanity1: rolling sum of window size 1 should equal the series itself"

    def test_rolling_sum_sanity2(self):
        t = np.random.rand(1000)
        ra = utils.rolling_sum(t, 1000)
        assert np.allclose([np.sum(t)], ra), "rolling_sum_sanity2: rolling sum of full window size should equal the series sum"

    def test_rolling_sum_random_data(self):
        t = np.random.rand(1000)
        m = np.random.randint(10, 1000)
        ra = utils.rolling_sum(t, m)
        ans = np.array([np.sum(t[i:i+m]) for i in range(1000 - m + 1)])
        assert len(ra) == 1000 - m + 1, "rolling_sum_random_data: rolling sum should have correct length"
        assert np.allclose(ra, ans), "rolling_sum_random_data: rolling sum should be computed correctly"


class TestRollingAvgSd:
    def test_rolling_avg_sd_window_too_large(self):
        with pytest.raises(ValueError):
            t = np.random.rand(500)
            ans = utils.rolling_avg_sd(t, 1000)

    def test_rolling_avg_sd_sanity1(self):
        t = np.random.rand(1000)
        ra, rsd = utils.rolling_avg_sd(t, 1)
        assert np.allclose(t, ra), "rolling_avg_sd_sanity1: rolling sum of window size 1 should equal the series itself"
        assert np.max(rsd) < 1e-5, \
            "rolling_avg_sd_sanity1: rolling sd of window size 1 should be all zero"

    def test_rolling_avg_sd_sanity2(self):
        t = np.random.rand(1000)
        ra, rsd = utils.rolling_avg_sd(t, 1000)
        assert np.allclose([np.mean(t)], ra), "rolling_avg_sd_sanity2: rolling sum of full window size should equal the series mean"
        assert np.allclose([np.std(t)], rsd), \
            "rolling_avg_sd_sanity2: rolling sd of full window size should equal the series sd"

    def test_rolling_avg_sd_data1(self):
        t = np.loadtxt("./data/random_walk_data.csv")
        ra, rsd = utils.rolling_avg_sd(t, 1000)
        ra_ans = np.loadtxt("./data/random_walk_data_rolling_mean.csv")
        rsd_ans = np.loadtxt("./data/random_walk_data_rolling_std.csv")
        assert np.allclose(ra, ra_ans), "rolling_avg_sd_random_data: rolling sum should be computed correctly"
        assert np.allclose(rsd, rsd_ans), "rolling_avg_sd_random_data: rolling sd should be computed correctly"

    def test_rolling_avg_sd_random_data(self):
        t = np.random.rand(1000)
        m = np.random.randint(10, 1000)
        ra, rsd = utils.rolling_avg_sd(t, m)
        naive_ra = np.array([np.mean(t[i:i+m]) for i in range(1000 - m + 1)])
        naive_rsd = np.array([np.std(t[i:i+m]) for i in range(1000 - m + 1)])
        assert np.allclose(ra, naive_ra), "rolling_avg_sd_random_data: rolling sum should be computed correctly"
        assert np.allclose(rsd, naive_rsd), "rolling_avg_sd_random_data: rolling sd should be computed correctly"


class TestZNormalize:
    def test_z_normalize_constant_seq(self):
        with pytest.raises(ValueError):
            t = np.ones(100)
            ans = utils.z_normalize(t)

    def test_z_normalize_sanity(self):
        large = np.random.randint(10, 100)
        small = np.random.randint(large)
        normalized = utils.z_normalize([small, large])
        assert np.allclose([-1, 1], normalized), "z_normalize_sanity: z-normalization should be computed correctly"


class TestSlidingDotProduct:
    def test_sliding_dot_product_query_too_long(self):
        with pytest.raises(ValueError):
            q = np.random.rand(1000)
            t = np.random.rand(200)
            sdp = utils.sliding_dot_product(q, t)

    def test_sliding_dot_product_sanity1(self):
        q = np.zeros(200)
        t = np.random.rand(1000)
        sdp = utils.sliding_dot_product(q, t)
        assert len(sdp) == len(t) - len(q) + 1, "sliding_dot_product_sanity1: result should have correct length"
        assert np.array_equal(sdp, np.zeros(len(t) - len(q) + 1)), \
            "sliding_dot_product_sanity1: dot product of zero vector should be zero"

    def test_sliding_dot_product_sanity2(self):
        q = np.array([1])
        t = np.random.rand(1000)
        sdp = utils.sliding_dot_product(q, t)
        assert len(sdp) == len(t) - len(q) + 1, "sliding_dot_product_sanity2: result should have correct length"
        assert np.allclose(sdp, t), "sliding_dot_product_sanity2: dot product of a vector with [1] should contain itself"

    def test_sliding_dot_product_data1(self):
        t = np.loadtxt("./data/random_walk_data.csv")
        q = t[:1000]
        sdp = utils.sliding_dot_product(q, t)
        ans = np.loadtxt("./data/random_walk_data_sdp.csv")
        assert len(sdp) == len(t) - len(q) + 1, "sliding_dot_product_data1: result should have correct length"
        assert np.allclose(sdp, ans), "sliding_dot_product_data1: sliding dot product should be computed correctly"

    def test_sliding_dot_product_random_data(self):
        n = np.random.randint(100, 1000)
        m = np.random.randint(10, n)
        q = np.random.rand(m)
        t = np.random.rand(n)
        sdp = utils.sliding_dot_product(q, t)
        ans = helpers.naive_sliding_dot_product(q, t)
        assert len(sdp) == n - m + 1, "sliding_dot_product_random_data: sliding dot product should have correct length"
        assert np.allclose(sdp, ans), "sliding_dot_product_random_data: sliding dot product should be computed correctly"


class TestCalculateDistanceProfile:
    def test_calculate_distance_profile_invalid_QT(self):
        with pytest.raises(ValueError):
            qt = np.random.rand(101)
            rolling_mean = np.random.rand(100)
            rolling_std = np.random.rand(100)
            dp = utils.calculate_distance_profile(qt, 10, rolling_mean[0], rolling_std[0], rolling_mean, rolling_std)

    def test_calculate_distance_profile_invalid_rolling_mean(self):
        with pytest.raises(ValueError):
            qt = np.random.rand(100)
            rolling_mean = np.random.rand(101)
            rolling_std = np.random.rand(100)
            dp = utils.calculate_distance_profile(qt, 10, rolling_mean[0], rolling_std[0], rolling_mean, rolling_std)

    def test_calculate_distance_profile_invalid_rolling_std(self):
        with pytest.raises(ValueError):
            qt = np.random.rand(100)
            rolling_mean = np.random.rand(100)
            rolling_std = np.random.rand(101)
            dp = utils.calculate_distance_profile(qt, 10, rolling_mean[0], rolling_std[0], rolling_mean, rolling_std)

    def test_calculate_distance_profile_constant_query(self):
        n = 100
        m = np.random.randint(10, n // 2)
        t = np.random.rand(n)
        q = np.full(m, np.random.rand())
        qt = utils.sliding_dot_product(q, t)
        rolling_mean, rolling_std = utils.rolling_avg_sd(t, m)
        dp = utils.calculate_distance_profile(qt, m, q[0], 0, rolling_mean, rolling_std)
        assert np.allclose(dp, np.full(n - m + 1, np.sqrt(m))), "calculate_distance_profile_constant_query: " \
                                        "distance of nonconstant sequence to constant query is sqrt(m) by definition."

    def test_calculate_distance_profile_constant_sequence(self):
        n = 100
        m = np.random.randint(10, n // 2)
        t = np.full(n, np.random.rand())
        q = np.random.rand(m)
        qt = utils.sliding_dot_product(q, t)
        rolling_mean, rolling_std = utils.rolling_avg_sd(t, m)
        dp = utils.calculate_distance_profile(qt, m, np.mean(q), np.std(q), rolling_mean, rolling_std)
        assert np.allclose(dp, np.full(n - m + 1, np.sqrt(m))), "calculate_distance_profile_constant_sequence: " \
                                        "distance of nonconstant query to constant sequence is sqrt(m) by definition."

    def test_calculate_distance_profile_constant_sequence_and_query(self):
        n = 100
        m = np.random.randint(10, n // 2)
        t = np.full(n, np.random.rand())
        q = np.full(m, np.random.rand())
        qt = utils.sliding_dot_product(q, t)
        rolling_mean, rolling_std = utils.rolling_avg_sd(t, m)
        dp = utils.calculate_distance_profile(qt, m, np.mean(q), np.std(q), rolling_mean, rolling_std)
        assert np.allclose(dp, np.full(n - m + 1, 0)), "calculate_distance_profile_constant_sequence_and_query: " \
                                        "distance of constant query to constant sequence is ero by definition."

    def test_calculate_distance_profile_data1(self):
        qt = np.loadtxt("./data/random_walk_data_sdp.csv")
        rolling_mean = np.loadtxt("./data/random_walk_data_rolling_mean.csv")
        rolling_std = np.loadtxt("./data/random_walk_data_rolling_std.csv")
        m = 1000
        dp = utils.calculate_distance_profile(qt, m, rolling_mean[0], rolling_std[0], rolling_mean, rolling_std)
        ans = np.loadtxt("./data/random_walk_data_distance_profile.csv")
        assert len(dp) == len(qt), "mass_data1: result should have correct length"
        assert np.allclose(dp, ans), "calculate_distance_profile_data1: distance profile should be computer correctly"


class TestMASS:
    def test_mass_query_too_long(self):
        with pytest.raises(ValueError):
            q = np.random.rand(1000)
            t = np.random.rand(200)
            dp = utils.mass(q, t)

    def test_mass_sanity(self):
        t = np.random.rand(1000)
        m = np.random.randint(len(t) - 10)
        k = np.random.randint(10, len(t) - m)
        q = t[k:k+m]
        dp = utils.mass(q, t)
        assert dp[k] < 1e-5, "test_mass_sanity: distance of a series to itself should be zero"

    def test_mass_data1(self):
        t = np.loadtxt("./data/random_walk_data.csv")
        q = t[:1000]
        dp = utils.mass(q, t)
        ans = np.loadtxt("./data/random_walk_data_distance_profile.csv")
        assert len(dp) == len(t) - len(q) + 1, "mass_data1: result should have correct length"
        assert np.allclose(dp, ans, atol=1e-5), "mass_data1: distance profile should be computed correctly"

    def test_mass_random_data(self):
        n = np.random.randint(100, 1000)
        m = np.random.randint(10, n)
        q = np.random.rand(m)
        t = np.random.rand(n)
        dp = utils.mass(q, t)
        ans = helpers.naive_distance_profile(q, t)
        assert np.allclose(dp, ans), "mass_random_data: distance profile should be computed correctly"


