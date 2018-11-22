import pytest
import numpy as np

from pytsmp import utils
from tests import helpers


def test_rolling_average_window_too_large():
    with pytest.raises(ValueError):
        t = np.random.rand(500)
        ans = utils.rolling_average(t, 1000)

def test_rolling_average_sanity1():
    t = np.random.rand(1000)
    ra = utils.rolling_average(t, 1)
    assert np.allclose(t, ra), "rolling_average_sanity1: rolling average of window size 1 should equal the series itself"

def test_rolling_average_sanity2():
    t = np.random.rand(1000)
    ra = utils.rolling_average(t, 1000)
    assert np.allclose([np.mean(t)], ra), "rolling_average_sanity2: rolling average of full window size should equal the series mean"

def test_rolling_average_random_data():
    t = np.random.rand(1000)
    m = np.random.randint(10, 1000)
    ra = utils.rolling_average(t, m)
    ans = np.array([np.mean(t[i:i+m]) for i in range(1000 - m + 1)])
    assert np.allclose(ra, ans), "rolling_average_random_data: rolling average should be computed correctly"



def test_rolling_avg_sd_window_too_large():
    with pytest.raises(ValueError):
        t = np.random.rand(500)
        ans = utils.rolling_avg_sd(t, 1000)

def test_rolling_avg_sd_sanity1():
    t = np.random.rand(1000)
    ra, rsd = utils.rolling_avg_sd(t, 1)
    assert np.allclose(t, ra), "rolling_avg_sd_sanity1: rolling average of window size 1 should equal the series itself"
    assert np.max(rsd) < 1e-5, \
        "rolling_avg_sd_sanity1: rolling sd of window size 1 should be all zero"

def test_rolling_avg_sd_sanity2():
    t = np.random.rand(1000)
    ra, rsd = utils.rolling_avg_sd(t, 1000)
    assert np.allclose([np.mean(t)], ra), "rolling_avg_sd_sanity2: rolling average of full window size should equal the series mean"
    assert np.allclose([np.std(t)], rsd), \
        "rolling_avg_sd_sanity2: rolling sd of full window size should equal the series sd"

def test_rolling_avg_sd_random_data():
    t = np.random.rand(1000)
    m = np.random.randint(10, 1000)
    ra, rsd = utils.rolling_avg_sd(t, m)
    naive_ra = np.array([np.mean(t[i:i+m]) for i in range(1000 - m + 1)])
    naive_rsd = np.array([np.std(t[i:i+m]) for i in range(1000 - m + 1)])
    assert np.allclose(ra, naive_ra), "rolling_avg_sd_random_data: rolling average should be computed correctly"
    assert np.allclose(rsd, naive_rsd), "rolling_avg_sd_random_data: rolling sd should be computed correctly"



def test_z_normalize_constant_seq():
    with pytest.raises(ValueError):
        t = np.ones(100)
        ans = utils.z_normalize(t)

def test_z_normalize_sanity():
    large = np.random.randint(10, 100)
    small = np.random.randint(large)
    normalized = utils.z_normalize([small, large])
    assert np.allclose([-1, 1], normalized), "z_normalize_sanity: z-normalization should be computed correctly"



def test_sliding_dot_product_query_too_long():
    with pytest.raises(ValueError):
        q = np.random.rand(1000)
        t = np.random.rand(200)
        sdp = utils.sliding_dot_product(q, t)

def test_sliding_dot_product_sanity1():
    q = np.zeros(200)
    t = np.random.rand(1000)
    sdp = utils.sliding_dot_product(q, t)
    assert len(sdp) == len(t) - len(q) + 1, "sliding_dot_product_sanity1: result should have correct length"
    assert np.array_equal(sdp, np.zeros(len(t) - len(q) + 1)), \
        "sliding_dot_product_sanity1: dot product of zero vector should be zero"

def test_sliding_dot_product_sanity2():
    q = np.array([1])
    t = np.random.rand(1000)
    sdp = utils.sliding_dot_product(q, t)
    assert len(sdp) == len(t) - len(q) + 1, "sliding_dot_product_sanity2: result should have correct length"
    assert np.allclose(sdp, t), "sliding_dot_product_sanity2: dot product of a vector with [1] should contain itself"

def test_sliding_dot_product_data1():
    t = np.loadtxt("./tests/data/random_walk_data.csv")
    q = t[:1000]
    sdp = utils.sliding_dot_product(q, t)
    ans = np.loadtxt("./tests/data/random_walk_data_sdp.csv")
    assert len(sdp) == len(t) - len(q) + 1, "sliding_dot_product_data1: result should have correct length"
    assert np.allclose(sdp, ans), "sliding_dot_product_data1: sliding dot product should be computed correctly"

def test_sliding_dot_product_random_data():
    n = np.random.randint(100, 1000)
    m = np.random.randint(10, n)
    q = np.random.rand(m)
    t = np.random.rand(n)
    sdp = utils.sliding_dot_product(q, t)
    ans = helpers.naive_sliding_dot_product(q, t)
    assert len(sdp) == n - m + 1, "sliding_dot_product_random_data: sliding dot product should have correct length"
    assert np.allclose(sdp, ans), "sliding_dot_product_random_data: sliding dot product should be computed correctly"



def test_mass_query_too_long():
    with pytest.raises(ValueError):
        q = np.random.rand(1000)
        t = np.random.rand(200)
        dp = utils.mass(q, t)

def test_mass_sanity():
    t = np.random.rand(1000)
    m = np.random.randint(len(t) - 10)
    k = np.random.randint(10, len(t) - m)
    q = t[k:k+m]
    dp = utils.mass(q, t)
    assert dp[k] < 1e-5, "test_mass_sanity: distance of a series to itself should be zero"

def test_mass_data1():
    t = np.loadtxt("./tests/data/random_walk_data.csv")
    q = t[:1000]
    dp = utils.mass(q, t)
    ans = np.loadtxt("./tests/data/random_walk_data_distance_profile.csv")
    assert len(dp) == len(t) - len(q) + 1, "mass_data1: result should have correct length"
    assert np.allclose(dp, ans, atol=1e-5), "mass_data1: distance profile should be computed correctly"

def test_mass_random_data():
    n = np.random.randint(100, 1000)
    m = np.random.randint(10, n)
    q = np.random.rand(m)
    t = np.random.rand(n)
    dp = utils.mass(q, t)
    ans = helpers.naive_distance_profile(q, t)
    assert np.allclose(dp, ans), "mass_random_data: distance profile should be computed correctly"


