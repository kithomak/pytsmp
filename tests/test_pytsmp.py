import pytest
import numpy as np

from pytsmp import pytsmp


def test_MatrixProfile_init():
    with pytest.raises(NotImplementedError):
        t = np.random.rand(1000)
        mp = pytsmp.MatrixProfile(t, window_size=100, verbose=False)

def test_STAMP_init_incorrect_window_size1():
    with pytest.raises(ValueError) as excinfo:
        t = np.random.rand(1000)
        mp = pytsmp.STAMP(t, window_size=0, verbose=False)
        assert str(excinfo.value) == "Incorrect window size specified."

def test_STAMP_init_incorrect_window_size2():
    with pytest.raises(ValueError) as excinfo:
        t = np.random.rand(1000)
        mp = pytsmp.STAMP(t, window_size=2.3, verbose=False)
        assert str(excinfo.value) == "Incorrect window size specified."

def test_STAMP_init_incorrect_window_size3():
    with pytest.raises(ValueError) as excinfo:
        t1 = np.random.rand(1000)
        t2 = np.random.rand(500)
        mp = pytsmp.STAMP(t1, t2, window_size=501, verbose=False)
        assert str(excinfo.value) == "Incorrect window size specified."

def test_STAMP_init_incorrect_exclusion_zone():
    with pytest.raises(ValueError) as excinfo:
        t = np.random.rand(1000)
        mp = pytsmp.STAMP(t, window_size=10, exclusion_zone=-1, verbose=False)
        assert str(excinfo.value) == "Exclusion zone must be non-negative."

def test_STAMP_init_incorrect_s_size1():
    with pytest.raises(ValueError) as excinfo:
        t = np.random.rand(1000)
        mp = pytsmp.STAMP(t, window_size=10, s_size=0, verbose=False)
        assert str(excinfo.value) == "s_size must be between 0 and 1."

def test_STAMP_init_incorrect_s_size2():
    with pytest.raises(ValueError) as excinfo:
        t = np.random.rand(1000)
        mp = pytsmp.STAMP(t, window_size=10, s_size=1.2, verbose=False)
        assert str(excinfo.value) == "s_size must be between 0 and 1."

def test_STAMP_init_check_mutation():
    t1 = np.random.rand(100)
    t2 = np.random.rand(100)
    w = 10
    mp = pytsmp.STAMP(t1, t2, window_size=w, exclusion_zone=0, verbose=False)
    t1[0] = -10
    t2[0] = -10
    assert t1[0] != mp.ts1[0], "STAMP_init_check_mutation: Matrix profile init should leave original array intact."
    assert t2[0] != mp.ts2[0], "STAMP_init_check_mutation: Matrix profile init should leave original array intact."

def test_STAMP_get_profile_check_length():
    n = np.random.randint(100, 1000)
    m = np.random.randint(100, 1000)
    t1 = np.random.rand(n)
    t2 = np.random.rand(m)
    w = np.random.randint(10, min(n, m))
    mp = pytsmp.STAMP(t1, t2, window_size=w, verbose=False)
    mpro, ipro = mp.get_profiles()
    assert len(mpro) == n - w + 1, "STAMP_get_profile_check_length: Matrix profile should have correct length"
    assert len(ipro) == n - w + 1, "STAMP_get_profile_check_length: Index profile should have correct length"

def test_STAMP_get_profile_check_mutation():
    t = np.random.rand(1000)
    w = 10
    mp = pytsmp.STAMP(t, window_size=w, verbose=False)
    mpro, ipro = mp.get_profiles()
    mpro[0] = -1
    ipro[0] = -1
    mpro2, ipro2 = mp.get_profiles()
    assert mpro[0] != mpro2[0], "STAMP_get_profile_check_mutation: " \
                                "Get profile should return a copy of the matrix profile, not the internal one."
    assert ipro[0] != ipro2[0], "STAMP_get_profile_check_mutation: " \
                                "Get profile should return a copy of the index profile, not the internal one."

def test_STAMP_compute_matrix_profile_sanity():
    t = np.random.rand(1000)
    w = 10
    mp = pytsmp.STAMP(t, t, window_size=w, verbose=False)
    mpro, ipro = mp.get_profiles()
    assert np.allclose(mpro, np.zeros(len(t) - w + 1), atol=1e-5), "STAMP_compute_matrix_profile_sanity: " \
                                                    "Should compute the matrix profile correctly in the trivial case."
    assert np.array_equal(ipro, np.arange(len(t) - w + 1)), "STAMP_compute_matrix_profile_sanity: " \
                                                    "Should compute the index profile correctly in the trivial case."

def test_STAMP_compute_matrix_profile_data1():
    t = np.loadtxt("./tests/data/random_walk_data.csv")
    mpro_ans = np.loadtxt("./tests/data/random_walk_data_mpro.csv")
    ipro_ans = np.loadtxt("./tests/data/random_walk_data_ipro.csv")
    w = 50
    mp = pytsmp.STAMP(t, window_size=w, verbose=False)
    mpro, ipro = mp.get_profiles()
    assert np.allclose(mpro, mpro_ans), "STAMP_compute_matrix_profile_data1: " \
                                        "Should compute the matrix profile correctly. " \
                                        "Max error is {}".format(np.max(np.abs(mpro - mpro_ans)))
    # assert np.allclose(ipro, ipro_ans), "STAMP_compute_matrix_profile_data1: " \
    #                                                           "Should compute the index profile correctly."

def test_STAMP_update_ts1_random_data():
    n = np.random.randint(200, 1000)
    m = np.random.randint(200, 1000)
    t1 = np.random.rand(n)
    t2 = np.random.rand(m)
    w = np.random.randint(10, min(n, m) // 4)
    mp = pytsmp.STAMP(t1[:-1], t2, window_size=w, verbose=False)
    mp.update_ts1(t1[-1])
    mpro, ipro = mp.get_profiles()
    mp2 = pytsmp.STAMP(t1, t2, window_size=w, verbose=False)
    mpro2, ipro2 = mp2.get_profiles()
    assert np.allclose(mpro, mpro2), "STAMP_update_ts1_random_data: " \
                                     "update_ts1 should update the matrix profile properly on random data. " \
                                     "Max error is {}".format(np.max(np.abs(mpro - mpro2)))
    assert np.allclose(ipro, ipro2), "STAMP_update_ts1_random_data: " \
                                     "update_ts1 should update the index profile properly on random data."

def test_STAMP_update_ts2_random_data():
    n = np.random.randint(200, 1000)
    m = np.random.randint(200, 1000)
    t1 = np.random.rand(n)
    t2 = np.random.rand(m)
    w = np.random.randint(10, min(n, m) // 4)
    mp = pytsmp.STAMP(t1, t2[:-1], window_size=w, verbose=False)
    mp.update_ts2(t2[-1])
    mpro, ipro = mp.get_profiles()
    mp2 = pytsmp.STAMP(t1, t2, window_size=w, verbose=False)
    mpro2, ipro2 = mp2.get_profiles()
    assert np.allclose(mpro, mpro2), "STAMP_update_ts2_random_data: " \
                                     "update_ts2 should update the matrix profile properly on random data. " \
                                     "Max error is {}".format(np.max(np.abs(mpro - mpro2)))
    assert np.allclose(ipro, ipro2), "STAMP_update_ts2_random_data: " \
                                     "update_ts2 should update the index profile properly on random data."

def test_STAMP_update_ts1_same_data():
    n = np.random.randint(200, 1000)
    t = np.random.rand(n)
    w = np.random.randint(10, n // 4)
    mp = pytsmp.STAMP(t[:-1], window_size=w, verbose=False)
    mp.update_ts1(t[-1])
    mpro, ipro = mp.get_profiles()
    mp2 = pytsmp.STAMP(t, window_size=w, verbose=False)
    mpro2, ipro2 = mp2.get_profiles()
    assert np.allclose(mpro, mpro2), "STAMP_update_ts1_same_data: " \
                                     "update_ts1 should update the matrix profile properly when ts1 == ts2. " \
                                     "Max error is {}".format(np.max(np.abs(mpro - mpro2)))
    assert np.allclose(ipro, ipro2), "STAMP_update_ts1_same_data: " \
                                     "update_ts1 should update the index profile properly when ts1 == ts2."

def test_STAMP_update_ts2_same_data():
    n = np.random.randint(200, 1000)
    t = np.random.rand(n)
    w = np.random.randint(10, n // 4)
    mp = pytsmp.STAMP(t[:-1], window_size=w, verbose=False)
    mp.update_ts2(t[-1])
    mpro, ipro = mp.get_profiles()
    mp2 = pytsmp.STAMP(t, window_size=w, verbose=False)
    mpro2, ipro2 = mp2.get_profiles()
    assert np.allclose(mpro, mpro2), "STAMP_update_ts2_same_data: " \
                                     "update_ts2 should update the matrix profile properly when ts1 == ts2. " \
                                     "Max error is {}".format(np.max(np.abs(mpro - mpro2)))
    assert np.allclose(ipro, ipro2), "STAMP_update_ts2_same_data: " \
                                     "update_ts2 should update the index profile properly when ts1 == ts2."


