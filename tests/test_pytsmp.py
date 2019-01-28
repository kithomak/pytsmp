import pytest
import numpy as np

from pytsmp import pytsmp
from tests import helpers


class TestMatrixProfile:
    def test_MatrixProfile_init(self):
        with pytest.raises(TypeError):
            t = np.random.rand(1000)
            mp = pytsmp.MatrixProfile(t, window_size=100, verbose=False)

class TestSTAMP:
    def test_STAMP_init_incorrect_window_size1(self):
        with pytest.raises(ValueError) as excinfo:
            t = np.random.rand(1000)
            mp = pytsmp.STAMP(t, window_size=0, verbose=False)
            assert str(excinfo.value) == "Incorrect window size specified."

    def test_STAMP_init_incorrect_window_size2(self):
        with pytest.raises(ValueError) as excinfo:
            t = np.random.rand(1000)
            mp = pytsmp.STAMP(t, window_size=2.3, verbose=False)
            assert str(excinfo.value) == "Incorrect window size specified."

    def test_STAMP_init_incorrect_window_size3(self):
        with pytest.raises(ValueError) as excinfo:
            t1 = np.random.rand(1000)
            t2 = np.random.rand(500)
            mp = pytsmp.STAMP(t1, t2, window_size=501, verbose=False)
            assert str(excinfo.value) == "Incorrect window size specified."

    def test_STAMP_init_incorrect_exclusion_zone(self):
        with pytest.raises(ValueError) as excinfo:
            t = np.random.rand(1000)
            mp = pytsmp.STAMP(t, window_size=10, exclusion_zone=-1, verbose=False)
            assert str(excinfo.value) == "Exclusion zone must be non-negative."

    def test_STAMP_init_incorrect_s_size1(self):
        with pytest.raises(ValueError) as excinfo:
            t = np.random.rand(1000)
            mp = pytsmp.STAMP(t, window_size=10, s_size=0, verbose=False)
            assert str(excinfo.value) == "s_size must be between 0 and 1."

    def test_STAMP_init_incorrect_s_size2(self):
        with pytest.raises(ValueError) as excinfo:
            t = np.random.rand(1000)
            mp = pytsmp.STAMP(t, window_size=10, s_size=1.2, verbose=False)
            assert str(excinfo.value) == "s_size must be between 0 and 1."

    def test_STAMP_is_anytime(self):
        t = np.random.rand(1000)
        mp = pytsmp.STAMP(t, window_size=10, s_size=1, verbose=True)  # for coverage purpose
        is_anytime = mp.is_anytime
        assert is_anytime == True, "STAMP_is_anytime: STAMP should be an anytime algorithm."

    def test_STAMP_init_check_mutation(self):
        t1 = np.random.rand(100)
        t2 = np.random.rand(100)
        w = 10
        mp = pytsmp.STAMP(t1, t2, window_size=w, exclusion_zone=0, verbose=False)
        t1[0] = -10
        t2[0] = -10
        assert t1[0] != mp.ts1[0], "STAMP_init_check_mutation: Matrix profile init should leave original array intact."
        assert t2[0] != mp.ts2[0], "STAMP_init_check_mutation: Matrix profile init should leave original array intact."

    def test_STAMP_get_profiles_check_length(self):
        n = np.random.randint(100, 1000)
        m = np.random.randint(100, 1000)
        t1 = np.random.rand(n)
        t2 = np.random.rand(m)
        w = np.random.randint(10, min(n, m))
        mp = pytsmp.STAMP(t1, t2, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        assert len(mpro) == n - w + 1, "STAMP_get_profile_check_length: Matrix profile should have correct length"
        assert len(ipro) == n - w + 1, "STAMP_get_profile_check_length: Index profile should have correct length"

    def test_STAMP_get_profiles_check_mutation(self):
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

    def test_STAMP_compute_matrix_profile_sanity(self):
        t = np.random.rand(1000)
        w = 10
        mp = pytsmp.STAMP(t, t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, np.zeros(len(t) - w + 1), atol=1e-5), "STAMP_compute_matrix_profile_sanity: " \
                                                        "Should compute the matrix profile correctly in the trivial case."
        assert np.array_equal(ipro, np.arange(len(t) - w + 1)), "STAMP_compute_matrix_profile_sanity: " \
                                                        "Should compute the index profile correctly in the trivial case."

    def test_STAMP_compute_matrix_profile_same_random_data(self):
        n = np.random.randint(100, 200)  # anything larger will be too time-consuming
        t = np.random.rand(n)
        w = np.random.randint(10, n // 4)
        mp = pytsmp.STAMP(t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        mp_naive, ip_naive = helpers.naive_matrix_profile(t, window_size=w)
        assert np.allclose(mpro, mp_naive), "STAMP_compute_matrix_profile_same_random_data: " \
                                            "Should compute the matrix profile correctly."
        assert np.allclose(ipro, ip_naive), "STAMP_compute_matrix_profile_same_random_data: " \
                                            "Should compute the index profile correctly."

    def test_STAMP_compute_matrix_profile_random_data(self):
        n = np.random.randint(100, 200)
        m = np.random.randint(100, 200)
        t1 = np.random.rand(n)
        t2 = np.random.rand(m)
        w = np.random.randint(10, min(n, m) // 4)
        mp = pytsmp.STAMP(t1, t2, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        mp_naive, ip_naive = helpers.naive_matrix_profile(t1, t2, window_size=w)
        assert np.allclose(mpro, mp_naive), "STAMP_compute_matrix_profile_random_data: " \
                                            "Should compute the matrix profile correctly."
        assert np.allclose(ipro, ip_naive), "STAMP_compute_matrix_profile_random_data: " \
                                            "Should compute the index profile correctly."

    def test_STAMP_compute_matrix_profile_data1(self):
        t = np.loadtxt("./data/random_walk_data.csv")
        mpro_ans = np.loadtxt("./data/random_walk_data_mpro.csv")
        ipro_ans = np.loadtxt("./data/random_walk_data_ipro.csv")
        w = 50
        mp = pytsmp.STAMP(t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, mpro_ans), "STAMP_compute_matrix_profile_data1: " \
                                            "Should compute the matrix profile correctly. " \
                                            "Max error is {}".format(np.max(np.abs(mpro - mpro_ans)))
        # assert np.allclose(ipro, ipro_ans), "STAMP_compute_matrix_profile_data1: " \
        #                                                           "Should compute the index profile correctly."

    def test_STAMP_compute_matrix_profile_data2(self):
        t = np.loadtxt("./data/candy_production.csv")
        mpro_ans = np.loadtxt("./data/candy_production_mpro.csv")
        ipro_ans = np.loadtxt("./data/candy_production_ipro.csv")
        w = 80
        mp = pytsmp.STAMP(t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, mpro_ans), "STAMP_compute_matrix_profile_data2: " \
                                            "Should compute the matrix profile correctly. " \
                                            "Max error is {}".format(np.max(np.abs(mpro - mpro_ans)))
        assert np.allclose(ipro, ipro_ans), "STAMP_compute_matrix_profile_data1: " \
                                            "Should compute the index profile correctly."

    def test_STAMP_compute_matrix_profile_data3(self):
        t = np.loadtxt("./data/bitcoin_price.csv")
        mpro_ans = np.loadtxt("./data/bitcoin_price_mpro.csv")
        ipro_ans = np.loadtxt("./data/bitcoin_price_ipro.csv")
        w = 100
        mp = pytsmp.STAMP(t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, mpro_ans), "STAMP_compute_matrix_profile_data3: " \
                                            "Should compute the matrix profile correctly. " \
                                            "Max error is {}".format(np.max(np.abs(mpro - mpro_ans)))
        assert np.allclose(ipro, ipro_ans), "STAMP_compute_matrix_profile_data3: " \
                                            "Should compute the index profile correctly."

    def test_STAMP_update_ts1_random_data(self):
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

    def test_STAMP_update_ts1_multiple_random_data(self):
        n = np.random.randint(200, 1000)
        m = np.random.randint(200, 1000)
        t1 = np.random.rand(n)
        t2 = np.random.rand(m)
        w = np.random.randint(10, min(n, m) // 4)
        times = np.random.randint(5, 50)
        mp = pytsmp.STAMP(t1[:-times], t2, window_size=w, verbose=False)
        for i in range(-times, 0, 1):
            mp.update_ts1(t1[i])
        mpro, ipro = mp.get_profiles()
        mp2 = pytsmp.STAMP(t1, t2, window_size=w, verbose=False)
        mpro2, ipro2 = mp2.get_profiles()
        assert np.allclose(mpro, mpro2), "STAMP_update_ts1_multiple_random_data: " \
                                         "update_ts1 should update the matrix profile multiple times properly on random data. " \
                                         "Max error is {}".format(np.max(np.abs(mpro - mpro2)))
        assert np.allclose(ipro, ipro2), "STAMP_update_ts1_random_data: " \
                                         "update_ts1 should update the index profile multiple times properly on random data."

    def test_STAMP_update_ts2_random_data(self):
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

    def test_STAMP_update_ts2_multiple_random_data(self):
        n = np.random.randint(200, 1000)
        m = np.random.randint(200, 1000)
        t1 = np.random.rand(n)
        t2 = np.random.rand(m)
        w = np.random.randint(10, min(n, m) // 4)
        times = np.random.randint(5, 50)
        mp = pytsmp.STAMP(t1, t2[:-times], window_size=w, verbose=False)
        for i in range(-times, 0, 1):
            mp.update_ts2(t2[i])
        mpro, ipro = mp.get_profiles()
        mp2 = pytsmp.STAMP(t1, t2, window_size=w, verbose=False)
        mpro2, ipro2 = mp2.get_profiles()
        assert np.allclose(mpro, mpro2), "STAMP_update_ts2_multiple_random_data: " \
                                         "update_ts2 should update the matrix profile multiple times properly on random data. " \
                                         "Max error is {}".format(np.max(np.abs(mpro - mpro2)))
        assert np.allclose(ipro, ipro2), "STAMP_update_ts2_random_data: " \
                                         "update_ts2 should update the index profile multiple times properly on random data."

    def test_STAMP_update_interleave_random_data(self):
        n = np.random.randint(200, 1000)
        m = np.random.randint(200, 1000)
        t1 = np.random.rand(n)
        t2 = np.random.rand(m)
        w = np.random.randint(10, min(n, m) // 4)
        times = np.random.randint(5, 25)
        mp = pytsmp.STAMP(t1[:-times], t2[:-times], window_size=w, verbose=False)
        for i in range(-times, 0, 1):
            mp.update_ts1(t1[i])
            mp.update_ts2(t2[i])
        mpro, ipro = mp.get_profiles()
        mp2 = pytsmp.STAMP(t1, t2, window_size=w, verbose=False)
        mpro2, ipro2 = mp2.get_profiles()
        assert np.allclose(mpro, mpro2), "STAMP_update_interleave_random_data: " \
                                         "update_ts1 and update_ts2 should update the matrix profile multiple times " \
                                         "properly on random data. " \
                                         "Max error is {}".format(np.max(np.abs(mpro - mpro2)))
        assert np.allclose(ipro, ipro2), "STAMP_update_interleave_random_data: " \
                                         "update_ts1 and update_ts2 should update the index profile multiple times " \
                                         "properly on random data."

    def test_STAMP_update_ts1_same_data(self):
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

    def test_STAMP_update_ts1_multiple_same_data(self):
        n = np.random.randint(200, 1000)
        t = np.random.rand(n)
        w = np.random.randint(10, n // 4)
        times = np.random.randint(5, 50)
        mp = pytsmp.STAMP(t[:-times], window_size=w, verbose=False)
        for i in range(-times, 0, 1):
            mp.update_ts1(t[i])
        mpro, ipro = mp.get_profiles()
        mp2 = pytsmp.STAMP(t, window_size=w, verbose=False)
        mpro2, ipro2 = mp2.get_profiles()
        assert np.allclose(mpro, mpro2), "STAMP_update_ts1_multiple_same_data: " \
                                         "update_ts1 should update the matrix profile multiple times properly when ts1 == ts2. " \
                                         "Max error is {}".format(np.max(np.abs(mpro - mpro2)))
        assert np.allclose(ipro, ipro2), "STAMP_update_ts1_multiple_same_data: " \
                                         "update_ts1 should update the index profile multiple times properly when ts1 == ts2."

    def test_STAMP_update_ts2_same_data(self):
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

    def test_STAMP_update_ts2_multiple_same_data(self):
        n = np.random.randint(200, 1000)
        t = np.random.rand(n)
        w = np.random.randint(10, n // 4)
        times = np.random.randint(5, 50)
        mp = pytsmp.STAMP(t[:-times], window_size=w, verbose=False)
        for i in range(-times, 0, 1):
            mp.update_ts2(t[i])
        mpro, ipro = mp.get_profiles()
        mp2 = pytsmp.STAMP(t, window_size=w, verbose=False)
        mpro2, ipro2 = mp2.get_profiles()
        assert np.allclose(mpro, mpro2), "STAMP_update_ts2_multiple_same_data: " \
                                         "update_ts2 should update the matrix profile multiple times properly when ts1 == ts2. " \
                                         "Max error is {}".format(np.max(np.abs(mpro - mpro2)))
        assert np.allclose(ipro, ipro2), "STAMP_update_ts2_multiple_same_data: " \
                                         "update_ts2 should update the index profile multiple times properly when ts1 == ts2."

    def test_STAMP_update_interleave_same_data(self):
        n = np.random.randint(200, 1000)
        t = np.random.rand(n)
        w = np.random.randint(10, n // 4)
        times = np.random.randint(5, 25)
        mp = pytsmp.STAMP(t[:-times], window_size=w, verbose=False)
        for i in range(-times, 0, 1):
            if i % 2 == 0:
                mp.update_ts1(t[i])
            else:
                mp.update_ts2(t[i])
        mpro, ipro = mp.get_profiles()
        mp2 = pytsmp.STAMP(t, window_size=w, verbose=False)
        mpro2, ipro2 = mp2.get_profiles()
        assert np.allclose(mpro, mpro2), "STAMP_update_interleave_same_data: " \
                                         "update_ts1 and update_ts2 should update the matrix profile multiple times " \
                                         "properly when ts1 == ts2. " \
                                         "Max error is {}".format(np.max(np.abs(mpro - mpro2)))
        assert np.allclose(ipro, ipro2), "STAMP_update_interleave_same_data: " \
                                         "update_ts1 and update_ts2 should update the index profile multiple times " \
                                         "properly when ts1 == ts2."


class TestSTOMP:
    def test_STOMP_is_anytime(self):
        t = np.random.rand(1000)
        mp = pytsmp.STOMP(t, window_size=10, s_size=1, verbose=True)
        is_anytime = mp.is_anytime
        assert is_anytime == False, "STOMP_is_anytime: STOMP should not be an anytime algorithm."

    def test_STOMP_get_profiles_check_length(self):
        n = np.random.randint(100, 1000)
        m = np.random.randint(100, 1000)
        t1 = np.random.rand(n)
        t2 = np.random.rand(m)
        w = np.random.randint(10, min(n, m))
        mp = pytsmp.STOMP(t1, t2, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        assert len(mpro) == n - w + 1, "STOMP_get_profile_check_length: Matrix profile should have correct length"
        assert len(ipro) == n - w + 1, "STOMP_get_profile_check_length: Index profile should have correct length"

    def test_STOMP_get_profiles_check_mutation(self):
        t = np.random.rand(1000)
        w = 10
        mp = pytsmp.STOMP(t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        mpro[0] = -1
        ipro[0] = -1
        mpro2, ipro2 = mp.get_profiles()
        assert mpro[0] != mpro2[0], "STOMP_get_profile_check_mutation: " \
                                    "Get profile should return a copy of the matrix profile, not the internal one."
        assert ipro[0] != ipro2[0], "STOMP_get_profile_check_mutation: " \
                                    "Get profile should return a copy of the index profile, not the internal one."

    def test_STOMP_compute_matrix_profile_sanity(self):
        t = np.random.rand(1000)
        w = 10
        mp = pytsmp.STOMP(t, t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, np.zeros(len(t) - w + 1), atol=1e-5), "STOMP_compute_matrix_profile_sanity: " \
                                                        "Should compute the matrix profile correctly in the trivial case."
        assert np.array_equal(ipro, np.arange(len(t) - w + 1)), "STOMP_compute_matrix_profile_sanity: " \
                                                        "Should compute the index profile correctly in the trivial case."

    def test_STOMP_compute_matrix_profile_same_random_data(self):
        n = np.random.randint(100, 200)  # anything larger will be too time-consuming
        t = np.random.rand(n)
        w = np.random.randint(10, n // 4)
        mp = pytsmp.STOMP(t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        mp_naive, ip_naive = helpers.naive_matrix_profile(t, window_size=w)
        assert np.allclose(mpro, mp_naive), "STOMP_compute_matrix_profile_same_random_data: " \
                                            "Should compute the matrix profile correctly."
        assert np.allclose(ipro, ip_naive), "STOMP_compute_matrix_profile_same_random_data: " \
                                            "Should compute the index profile correctly."

    def test_STOMP_compute_matrix_profile_random_data(self):
        n = np.random.randint(100, 200)
        m = np.random.randint(100, 200)
        t1 = np.random.rand(n)
        t2 = np.random.rand(m)
        w = np.random.randint(10, min(n, m) // 4)
        mp = pytsmp.STOMP(t1, t2, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        mp_naive, ip_naive = helpers.naive_matrix_profile(t1, t2, window_size=w)
        assert np.allclose(mpro, mp_naive), "STOMP_compute_matrix_profile_random_data: " \
                                            "Should compute the matrix profile correctly."
        assert np.allclose(ipro, ip_naive), "STOMP_compute_matrix_profile_random_data: " \
                                            "Should compute the index profile correctly."

    def test_STOMP_compute_matrix_profile_data1(self):
        t = np.loadtxt("./data/random_walk_data.csv")
        mpro_ans = np.loadtxt("./data/random_walk_data_mpro.csv")
        ipro_ans = np.loadtxt("./data/random_walk_data_ipro.csv")
        w = 50
        mp = pytsmp.STOMP(t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, mpro_ans), "STOMP_compute_matrix_profile_data1: " \
                                            "Should compute the matrix profile correctly. " \
                                            "Max error is {}".format(np.max(np.abs(mpro - mpro_ans)))
        # assert np.allclose(ipro, ipro_ans), "STOMP_compute_matrix_profile_data1: " \
        #                                     "Should compute the index profile correctly."

    def test_STOMP_compute_matrix_profile_data2(self):
        t = np.loadtxt("./data/candy_production.csv")
        mpro_ans = np.loadtxt("./data/candy_production_mpro.csv")
        ipro_ans = np.loadtxt("./data/candy_production_ipro.csv")
        w = 80
        mp = pytsmp.STOMP(t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, mpro_ans), "STOMP_compute_matrix_profile_data2: " \
                                            "Should compute the matrix profile correctly. " \
                                            "Max error is {}".format(np.max(np.abs(mpro - mpro_ans)))
        assert np.allclose(ipro, ipro_ans), "STOMP_compute_matrix_profile_data1: " \
                                            "Should compute the index profile correctly."

    def test_STOMP_compute_matrix_profile_data3(self):
        t = np.loadtxt("./data/bitcoin_price.csv")
        mpro_ans = np.loadtxt("./data/bitcoin_price_mpro.csv")
        ipro_ans = np.loadtxt("./data/bitcoin_price_ipro.csv")
        w = 100
        mp = pytsmp.STOMP(t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, mpro_ans), "STOMP_compute_matrix_profile_data3: " \
                                            "Should compute the matrix profile correctly. " \
                                            "Max error is {}".format(np.max(np.abs(mpro - mpro_ans)))
        assert np.allclose(ipro, ipro_ans), "STOMP_compute_matrix_profile_data3: " \
                                            "Should compute the index profile correctly."


class TestSCRIMP:
    def test_SCRIMP_is_anytime(self):
        t = np.random.rand(1000)
        mp = pytsmp.SCRIMP(t, window_size=10, s_size=1, verbose=True, pre_scrimp=0)
        is_anytime = mp.is_anytime
        assert is_anytime == True, "SCRIMP_is_anytime: SCRIMP should be an anytime algorithm."

    def test_SCRIMP_get_profiles_check_length(self):
        n = np.random.randint(100, 1000)
        m = np.random.randint(100, 1000)
        t1 = np.random.rand(n)
        t2 = np.random.rand(m)
        w = np.random.randint(10, min(n, m))
        mp = pytsmp.SCRIMP(t1, t2, window_size=w, verbose=False, pre_scrimp=0)
        mpro, ipro = mp.get_profiles()
        assert len(mpro) == n - w + 1, "SCRIMP_get_profile_check_length: Matrix profile should have correct length"
        assert len(ipro) == n - w + 1, "SCRIMP_get_profile_check_length: Index profile should have correct length"

    def test_SCRIMP_get_profiles_check_mutation(self):
        t = np.random.rand(1000)
        w = 10
        mp = pytsmp.SCRIMP(t, window_size=w, verbose=False, pre_scrimp=0)
        mpro, ipro = mp.get_profiles()
        mpro[0] = -1
        ipro[0] = -1
        mpro2, ipro2 = mp.get_profiles()
        assert mpro[0] != mpro2[0], "SCRIMP_get_profile_check_mutation: " \
                                    "Get profile should return a copy of the matrix profile, not the internal one."
        assert ipro[0] != ipro2[0], "SCRIMP_get_profile_check_mutation: " \
                                    "Get profile should return a copy of the index profile, not the internal one."

    def test_SCRIMP_compute_matrix_profile_sanity(self):
        t = np.random.rand(1000)
        w = 10
        mp = pytsmp.SCRIMP(t, t, window_size=w, verbose=False, pre_scrimp=0)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, np.zeros(len(t) - w + 1), atol=1e-5), "SCRIMP_compute_matrix_profile_sanity: " \
                                                        "Should compute the matrix profile correctly in the trivial case."
        assert np.array_equal(ipro, np.arange(len(t) - w + 1)), "SCRIMP_compute_matrix_profile_sanity: " \
                                                        "Should compute the index profile correctly in the trivial case."

    def test_SCRIMP_compute_matrix_profile_same_random_data(self):
        n = np.random.randint(100, 200)  # anything larger will be too time-consuming
        t = np.random.rand(n)
        w = np.random.randint(10, n // 4)
        mp = pytsmp.SCRIMP(t, window_size=w, verbose=False, pre_scrimp=0)
        mpro, ipro = mp.get_profiles()
        mp_naive, ip_naive = helpers.naive_matrix_profile(t, window_size=w)
        assert np.allclose(mpro, mp_naive), "SCRIMP_compute_matrix_profile_same_random_data: " \
                                            "Should compute the matrix profile correctly."
        assert np.allclose(ipro, ip_naive), "SCRIMP_compute_matrix_profile_same_random_data: " \
                                            "Should compute the index profile correctly."

    def test_SCRIMP_compute_matrix_profile_random_data(self):
        n = np.random.randint(100, 200)
        m = np.random.randint(100, 200)
        t1 = np.random.rand(n)
        t2 = np.random.rand(m)
        w = np.random.randint(10, min(n, m) // 4)
        mp = pytsmp.SCRIMP(t1, t2, window_size=w, verbose=False, pre_scrimp=0)
        mpro, ipro = mp.get_profiles()
        mp_naive, ip_naive = helpers.naive_matrix_profile(t1, t2, window_size=w)
        assert np.allclose(mpro, mp_naive), "SCRIMP_compute_matrix_profile_random_data: " \
                                            "Should compute the matrix profile correctly."
        assert np.allclose(ipro, ip_naive), "SCRIMP_compute_matrix_profile_random_data: " \
                                            "Should compute the index profile correctly."

    def test_SCRIMP_compute_matrix_profile_data1(self):
        t = np.loadtxt("./data/random_walk_data.csv")
        mpro_ans = np.loadtxt("./data/random_walk_data_mpro.csv")
        ipro_ans = np.loadtxt("./data/random_walk_data_ipro.csv")
        w = 50
        mp = pytsmp.SCRIMP(t, window_size=w, verbose=False, pre_scrimp=0)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, mpro_ans), "SCRIMP_compute_matrix_profile_data1: " \
                                            "Should compute the matrix profile correctly. " \
                                            "Max error is {}".format(np.max(np.abs(mpro - mpro_ans)))
        # assert np.allclose(ipro, ipro_ans), "SCRIMP_compute_matrix_profile_data1: " \
        #                                     "Should compute the index profile correctly."

    def test_SCRIMP_compute_matrix_profile_data2(self):
        t = np.loadtxt("./data/candy_production.csv")
        mpro_ans = np.loadtxt("./data/candy_production_mpro.csv")
        ipro_ans = np.loadtxt("./data/candy_production_ipro.csv")
        w = 80
        mp = pytsmp.SCRIMP(t, window_size=w, verbose=False, pre_scrimp=0)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, mpro_ans), "SCRIMP_compute_matrix_profile_data2: " \
                                            "Should compute the matrix profile correctly. " \
                                            "Max error is {}".format(np.max(np.abs(mpro - mpro_ans)))
        assert np.allclose(ipro, ipro_ans), "SCRIMP_compute_matrix_profile_data1: " \
                                            "Should compute the index profile correctly."

    def test_SCRIMP_compute_matrix_profile_data3(self):
        t = np.loadtxt("./data/bitcoin_price.csv")
        mpro_ans = np.loadtxt("./data/bitcoin_price_mpro.csv")
        ipro_ans = np.loadtxt("./data/bitcoin_price_ipro.csv")
        w = 100
        mp = pytsmp.SCRIMP(t, window_size=w, verbose=False, pre_scrimp=0)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, mpro_ans), "SCRIMP_compute_matrix_profile_data3: " \
                                            "Should compute the matrix profile correctly. " \
                                            "Max error is {}".format(np.max(np.abs(mpro - mpro_ans)))
        assert np.allclose(ipro, ipro_ans), "SCRIMP_compute_matrix_profile_data3: " \
                                            "Should compute the index profile correctly."


class TestPreSCRIMP:
    def test_PreSCRIMP_is_anytime(self):
        t = np.random.rand(1000)
        mp = pytsmp.PreSCRIMP(t, window_size=10, s_size=1, verbose=True)
        is_anytime = mp.is_anytime
        assert is_anytime == True, "PreSCRIMP_is_anytime: PreSCRIMP should be an anytime algorithm."

    def test_PreSCRIMP_init_incorrect_pre_scrimp1(self):
        with pytest.raises(ValueError) as excinfo:
            t = np.random.rand(1000)
            mp = pytsmp.PreSCRIMP(t, window_size=10, verbose=False, sample_rate=0)
            assert str(excinfo.value) == "sample_rate must be positive."

    def test_PreSCRIMP_init_incorrect_pre_scrimp2(self):
        with pytest.raises(ValueError) as excinfo:
            t = np.random.rand(1000)
            mp = pytsmp.PreSCRIMP(t, window_size=10, verbose=False, sample_rate=-2)
            assert str(excinfo.value) == "sample_rate must be positive."

    def test_PreSCRIMP_get_profiles_check_length(self):
        n = np.random.randint(100, 1000)
        m = np.random.randint(100, 1000)
        t1 = np.random.rand(n)
        t2 = np.random.rand(m)
        w = np.random.randint(10, min(n, m))
        mp = pytsmp.PreSCRIMP(t1, t2, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        assert len(mpro) == n - w + 1, "PreSCRIMP_get_profile_check_length: Matrix profile should have correct length"
        assert len(ipro) == n - w + 1, "PreSCRIMP_get_profile_check_length: Index profile should have correct length"

    def test_PreSCRIMP_get_profiles_check_mutation(self):
        t = np.random.rand(1000)
        w = 10
        mp = pytsmp.PreSCRIMP(t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        mpro[0] = -1
        ipro[0] = -1
        mpro2, ipro2 = mp.get_profiles()
        assert mpro[0] != mpro2[0], "PreSCRIMP_get_profile_check_mutation: " \
                                    "Get profile should return a copy of the matrix profile, not the internal one."
        assert ipro[0] != ipro2[0], "PreSCRIMP_get_profile_check_mutation: " \
                                    "Get profile should return a copy of the index profile, not the internal one."

    def test_PreSCRIMP_compute_matrix_profile_sanity1(self):
        t = np.random.rand(1000)
        w = 10
        mp = pytsmp.PreSCRIMP(t, t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, np.zeros(len(t) - w + 1), atol=1e-5), "PreSCRIMP_compute_matrix_profile_sanity1: " \
                                                        "Should compute the matrix profile correctly in the trivial case."
        assert np.array_equal(ipro, np.arange(len(t) - w + 1)), "PreSCRIMP_compute_matrix_profile_sanity1: " \
                                                        "Should compute the index profile correctly in the trivial case."

    def test_PreSCRIMP_compute_matrix_profile_sanity2(self):
        t = np.random.rand(1000)
        w = 50
        mpp = pytsmp.PreSCRIMP(t, t, window_size=w, verbose=False)
        mprop, iprop = mpp.get_profiles()
        mp = pytsmp.SCRIMP(t, t, window_size=w, verbose=False, pre_scrimp=0)
        mpro, ipro = mp.get_profiles()
        assert (mprop > mpro - 1e-5).all(), "PreSCRIMP_compute_matrix_profile_sanity2: PreSCRIMP should be an " \
                                     "upper approximation for the actual matrix profile."

    @pytest.mark.skip(reason="Randomized tests on approximate algorithms do not seem a correct thing to do.")
    def test_PreSCRIMP_compute_matrix_profile_same_random_data(self):
        n = np.random.randint(100, 200)  # anything larger will be too time-consuming
        t = np.random.rand(n)
        w = np.random.randint(10, n // 4)
        mp = pytsmp.PreSCRIMP(t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        mp_naive, ip_naive = helpers.naive_matrix_profile(t, window_size=w)
        assert np.allclose(mpro, mp_naive), "PreSCRIMP_compute_matrix_profile_same_random_data: " \
                                            "Should compute the matrix profile correctly."
        assert np.allclose(ipro, ip_naive), "PreSCRIMP_compute_matrix_profile_same_random_data: " \
                                            "Should compute the index profile correctly."

    @pytest.mark.skip(reason="Randomized tests on approximate algorithms do not seem a correct thing to do.")
    def test_PreSCRIMP_compute_matrix_profile_random_data(self):
        n = np.random.randint(100, 200)
        m = np.random.randint(100, 200)
        t1 = np.random.rand(n)
        t2 = np.random.rand(m)
        w = np.random.randint(10, min(n, m) // 4)
        mp = pytsmp.PreSCRIMP(t1, t2, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        mp_naive, ip_naive = helpers.naive_matrix_profile(t1, t2, window_size=w)
        assert np.allclose(mpro, mp_naive), "PreSCRIMP_compute_matrix_profile_random_data: " \
                                            "Should compute the matrix profile correctly."
        assert np.allclose(ipro, ip_naive), "PreSCRIMP_compute_matrix_profile_random_data: " \
                                            "Should compute the index profile correctly."

    @pytest.mark.skip(reason="To be tested later.")
    def test_PreSCRIMP_compute_matrix_profile_data1(self):
        t = np.loadtxt("./data/random_walk_data.csv")
        mpro_ans = np.loadtxt("./data/random_walk_data_mpro.csv")
        ipro_ans = np.loadtxt("./data/random_walk_data_ipro.csv")
        w = 50
        mp = pytsmp.PreSCRIMP(t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, mpro_ans), "PreSCRIMP_compute_matrix_profile_data1: " \
                                            "Should compute the matrix profile correctly. " \
                                            "Max error is {}".format(np.max(np.abs(mpro - mpro_ans)))
        # assert np.allclose(ipro, ipro_ans), "PreSCRIMP_compute_matrix_profile_data1: " \
        #                                     "Should compute the index profile correctly."

    @pytest.mark.skip(reason="To be tested later.")
    def test_PreSCRIMP_compute_matrix_profile_data2(self):
        t = np.loadtxt("./data/candy_production.csv")
        mpro_ans = np.loadtxt("./data/candy_production_mpro.csv")
        ipro_ans = np.loadtxt("./data/candy_production_ipro.csv")
        w = 80
        mp = pytsmp.PreSCRIMP(t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, mpro_ans), "PreSCRIMP_compute_matrix_profile_data2: " \
                                            "Should compute the matrix profile correctly. " \
                                            "Max error is {}".format(np.max(np.abs(mpro - mpro_ans)))
        assert np.allclose(ipro, ipro_ans), "PreSCRIMP_compute_matrix_profile_data1: " \
                                            "Should compute the index profile correctly."

    @pytest.mark.skip(reason="To be tested later.")
    def test_PreSCRIMP_compute_matrix_profile_data3(self):
        t = np.loadtxt("./data/bitcoin_price.csv")
        mpro_ans = np.loadtxt("./data/bitcoin_price_mpro.csv")
        ipro_ans = np.loadtxt("./data/bitcoin_price_ipro.csv")
        w = 100
        mp = pytsmp.PreSCRIMP(t, window_size=w, verbose=False)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, mpro_ans), "PreSCRIMP_compute_matrix_profile_data3: " \
                                            "Should compute the matrix profile correctly. " \
                                            "Max error is {}".format(np.max(np.abs(mpro - mpro_ans)))
        assert np.allclose(ipro, ipro_ans), "PreSCRIMP_compute_matrix_profile_data3: " \
                                            "Should compute the index profile correctly."

class TestSCRIMP_PreSCRIMP:
    def test_SCRIMP_init_incorrect_pre_scrimp(self):
        with pytest.raises(ValueError) as excinfo:
            t = np.random.rand(1000)
            mp = pytsmp.SCRIMP(t, window_size=10, verbose=False, pre_scrimp=-1)
            assert str(excinfo.value) == "pre_scrimp parameter must be non-negative."

    def test_SCRIMP_init_pre_scrimp_zero(self):
        t = np.random.rand(1000)
        mp = pytsmp.SCRIMP(t, window_size=10, s_size=1, verbose=False, pre_scrimp=0)
        assert getattr(mp, "_pre_scrimp_class", None) is None, "SCRIMP_init_pre_scrimp_zero: " \
                                                               "PreSCRIMP should not run if pre_scrimp = 0."

    def test_SCRIMP_init_pre_scrimp_nonzero(self):
        t = np.random.rand(1000)
        mp = pytsmp.SCRIMP(t, window_size=10, s_size=1, verbose=False, pre_scrimp=1/2)
        assert getattr(mp, "_pre_scrimp_class", None) is not None, "SCRIMP_init_pre_scrimp_nonzero: " \
                                                                   "PreSCRIMP should run if pre_scrimp > 0."

    def test_SCRIMP_PreSCRIMP_get_profiles_check_length(self):
        n = np.random.randint(100, 1000)
        m = np.random.randint(100, 1000)
        t1 = np.random.rand(n)
        t2 = np.random.rand(m)
        w = np.random.randint(10, min(n, m))
        mp = pytsmp.SCRIMP(t1, t2, window_size=w, verbose=False, pre_scrimp=1/4)
        mpro, ipro = mp.get_profiles()
        assert len(mpro) == n - w + 1, "SCRIMP_get_profile_check_length: Matrix profile should have correct length"
        assert len(ipro) == n - w + 1, "SCRIMP_get_profile_check_length: Index profile should have correct length"

    def test_SCRIMP_PreSCRIMP_get_profiles_check_mutation(self):
        t = np.random.rand(1000)
        w = 10
        mp = pytsmp.SCRIMP(t, window_size=w, verbose=False, pre_scrimp=1/4)
        mpro, ipro = mp.get_profiles()
        mpro[0] = -1
        ipro[0] = -1
        mpro2, ipro2 = mp.get_profiles()
        assert mpro[0] != mpro2[0], "SCRIMP_get_profile_check_mutation: " \
                                    "Get profile should return a copy of the matrix profile, not the internal one."
        assert ipro[0] != ipro2[0], "SCRIMP_get_profile_check_mutation: " \
                                    "Get profile should return a copy of the index profile, not the internal one."

    def test_SCRIMP_PreSCRIMP_compute_matrix_profile_sanity(self):
        t = np.random.rand(1000)
        w = 10
        mp = pytsmp.SCRIMP(t, t, window_size=w, verbose=False, pre_scrimp=1/4)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, np.zeros(len(t) - w + 1), atol=1e-5), "SCRIMP_compute_matrix_profile_sanity: " \
                                                        "Should compute the matrix profile correctly in the trivial case."
        assert np.array_equal(ipro, np.arange(len(t) - w + 1)), "SCRIMP_compute_matrix_profile_sanity: " \
                                                        "Should compute the index profile correctly in the trivial case."

    def test_SCRIMP_PreSCRIMP_compute_matrix_profile_same_random_data(self):
        n = np.random.randint(100, 200)  # anything larger will be too time-consuming
        t = np.random.rand(n)
        w = np.random.randint(10, n // 4)
        mp = pytsmp.SCRIMP(t, window_size=w, verbose=False, pre_scrimp=1/4)
        mpro, ipro = mp.get_profiles()
        mp_naive, ip_naive = helpers.naive_matrix_profile(t, window_size=w)
        assert np.allclose(mpro, mp_naive), "SCRIMP_compute_matrix_profile_same_random_data: " \
                                            "Should compute the matrix profile correctly."
        assert np.allclose(ipro, ip_naive), "SCRIMP_compute_matrix_profile_same_random_data: " \
                                            "Should compute the index profile correctly."

    def test_SCRIMP_PreSCRIMP_compute_matrix_profile_random_data(self):
        n = np.random.randint(100, 200)
        m = np.random.randint(100, 200)
        t1 = np.random.rand(n)
        t2 = np.random.rand(m)
        w = np.random.randint(10, min(n, m) // 4)
        mp = pytsmp.SCRIMP(t1, t2, window_size=w, verbose=False, pre_scrimp=1/4)
        mpro, ipro = mp.get_profiles()
        mp_naive, ip_naive = helpers.naive_matrix_profile(t1, t2, window_size=w)
        assert np.allclose(mpro, mp_naive), "SCRIMP_compute_matrix_profile_random_data: " \
                                            "Should compute the matrix profile correctly."
        assert np.allclose(ipro, ip_naive), "SCRIMP_compute_matrix_profile_random_data: " \
                                            "Should compute the index profile correctly."

    def test_SCRIMP_PreSCRIMP_compute_matrix_profile_data1(self):
        t = np.loadtxt("./data/random_walk_data.csv")
        mpro_ans = np.loadtxt("./data/random_walk_data_mpro.csv")
        ipro_ans = np.loadtxt("./data/random_walk_data_ipro.csv")
        w = 50
        mp = pytsmp.SCRIMP(t, window_size=w, verbose=False, pre_scrimp=1/4)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, mpro_ans), "SCRIMP_compute_matrix_profile_data1: " \
                                            "Should compute the matrix profile correctly. " \
                                            "Max error is {}".format(np.max(np.abs(mpro - mpro_ans)))
        # assert np.allclose(ipro, ipro_ans), "SCRIMP_compute_matrix_profile_data1: " \
        #                                     "Should compute the index profile correctly."

    def test_SCRIMP_PreSCRIMP_compute_matrix_profile_data2(self):
        t = np.loadtxt("./data/candy_production.csv")
        mpro_ans = np.loadtxt("./data/candy_production_mpro.csv")
        ipro_ans = np.loadtxt("./data/candy_production_ipro.csv")
        w = 80
        mp = pytsmp.SCRIMP(t, window_size=w, verbose=False, pre_scrimp=1/4)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, mpro_ans), "SCRIMP_compute_matrix_profile_data2: " \
                                            "Should compute the matrix profile correctly. " \
                                            "Max error is {}".format(np.max(np.abs(mpro - mpro_ans)))
        assert np.allclose(ipro, ipro_ans), "SCRIMP_compute_matrix_profile_data1: " \
                                            "Should compute the index profile correctly."

    def test_SCRIMP_PreSCRIMP_compute_matrix_profile_data3(self):
        t = np.loadtxt("./data/bitcoin_price.csv")
        mpro_ans = np.loadtxt("./data/bitcoin_price_mpro.csv")
        ipro_ans = np.loadtxt("./data/bitcoin_price_ipro.csv")
        w = 100
        mp = pytsmp.SCRIMP(t, window_size=w, verbose=False, pre_scrimp=1/4)
        mpro, ipro = mp.get_profiles()
        assert np.allclose(mpro, mpro_ans), "SCRIMP_compute_matrix_profile_data3: " \
                                            "Should compute the matrix profile correctly. " \
                                            "Max error is {}".format(np.max(np.abs(mpro - mpro_ans)))
        assert np.allclose(ipro, ipro_ans), "SCRIMP_compute_matrix_profile_data3: " \
                                            "Should compute the index profile correctly."


