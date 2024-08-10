import os
import sys


import numpy as np
import pytest
from scipy import stats

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../sequential_batch_testing"
)

from initialization import System
from poisson_binomial import (
    build_probability_table,
    compute_probability_of_not_stopping,
    compute_probability_of_not_stopping_interval,
    poisson_binomial_pdf,
)


class TestPoissonBinomial:
    def test_poisson_binomial(self):
        """Ensures poisson binomial behaves like binomial in equal case"""
        n = 10
        for k in range(n):
            assert poisson_binomial_pdf(k, [0.62 for _ in range(10)]) == pytest.approx(
                stats.binom.pmf(k, p=0.62, n=10)
            )
            assert poisson_binomial_pdf(k, [0.41 for _ in range(10)]) == pytest.approx(
                stats.binom.pmf(k, p=0.41, n=10)
            )

    def test_probability_table(self):
        p = 0.62
        first_row = [(1 - p) ** i for i in range(11)]
        table = build_probability_table([p for _ in range(10)])
        assert np.allclose(table[0], first_row)

        p = 0.41
        first_row = [(1 - p) ** i for i in range(11)]
        table = build_probability_table([p for _ in range(10)])
        assert np.allclose(table[0], first_row)

        p_list = np.random.rand(10)
        table = build_probability_table(p_list)

        for n in range(10):
            for k in range(n + 1):
                assert table[k, n] == pytest.approx(poisson_binomial_pdf(k, p_list[:n]))

    def test_probability_of_not_stopping(self):
        n = 50
        p_list = np.random.rand(n)
        probabilities_of_not_stopping_series = compute_probability_of_not_stopping(
            n, p_list
        )
        probabilities_of_not_stopping_parallel = compute_probability_of_not_stopping(
            1, p_list
        )
        for k in range(n + 1):
            assert probabilities_of_not_stopping_series[k] == pytest.approx(
                np.prod([p for p in p_list[:k]])
            )
            assert probabilities_of_not_stopping_parallel[k] == pytest.approx(
                np.prod([1 - p for p in p_list[:k]])
            )

    def test_compare_k_interval(self):

        n = 10

        for k in range(1, n + 1):
            system = System(n, "n", 1, k=k)
            p_list = [x.probability for x in system.components]
            assert compute_probability_of_not_stopping(k, p_list) == pytest.approx(
                compute_probability_of_not_stopping_interval(system, p_list)
            ), f"{k=}, {p_list=}"
