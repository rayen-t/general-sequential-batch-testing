import os
import sys

import numpy as np
import pytest
from sequential_testing import *

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../sequential_batch_testing"
)

import batching
from initialization import System
from poisson_binomial import poisson_binomial_pdf


class TestComputeQt:
    def test_trivial(self):
        pt = [1]
        assert (compute_qt(pt) == np.array([1])).all()

        pt = [0.5, 0.5]
        assert (compute_qt(pt) == [1, 0.5]).all()


class TestOptiamlBatching:
    def test_trivial(self):
        B = 5
        assert optimal_batching([1], B) == (6, [1])
        assert optimal_batching([1], 0) == (1, [1])
        assert optimal_batching([0.5, 0.5], B) == (7, [2])
        assert optimal_batching([0.5, 0.5], 0) == (1.5, [1, 2])
        assert optimal_batching([1, 0], B) == (6, [1, 2])


class TestGetCost:
    def test_get_cost_trivial(self):
        for t in range(5):
            assert get_cost([5], t, 5, 3.14) == 5 + 3.14
            assert get_cost([1, 2, 3, 4, 5], t, 5, 3.14) == pytest.approx(
                (t + 1) * (4.14)
            )


def get_p_t(costs, succ_probabilities):
    cum_cost = np.cumsum(costs)
    fail_probability = 1 - succ_probabilities
    cum_prob = np.cumprod(succ_probabilities)
    p_t = np.zeros(cum_cost[-1])
    for i, cost in enumerate(cum_cost):
        if i == 0:
            p_t[cost - 1] = fail_probability[i]
        elif i == len(cum_cost) - 1:
            p_t[cost - 1] = cum_prob[i - 1]
        else:
            p_t[cost - 1] = cum_prob[i - 1] * fail_probability[i]
    assert sum(p_t) == pytest.approx(1)
    return p_t


class TestCompareOld:
    def test_compare_random(self):
        for n in range(4, 100):
            costs = np.random.randint(1, 10, n)
            succ_probabilities = np.random.rand(n)
            p_t = get_p_t(costs, succ_probabilities)
            batch_cost = np.random.uniform(0, 20)
            system = System(
                n,
                batch_cost,
                is_series=True,
                costs=costs,
                probabilities=succ_probabilities,
            )
            _, optimal_batch_cost1 = batching.optimal_batching(system, list(range(n)))
            optimal_batch_cost2, batched_solution = optimal_batching(p_t, batch_cost)
            optimal_batch_cost3 = sum(
                [
                    p_t[i] * get_cost(batched_solution, i, len(p_t), batch_cost)
                    for i in range(len(p_t))
                ]
            )
            assert optimal_batch_cost2 == pytest.approx(optimal_batch_cost1), "DP wrong"
            assert optimal_batch_cost2 == pytest.approx(
                optimal_batch_cost3
            ), "check wrong"


if __name__ == "__main__":
    for n in range(4, 100):
        costs = np.random.randint(1, 10, n)
        succ_probabilities = np.random.rand(n)
        p_t = get_p_t(costs, succ_probabilities)
        batch_cost = np.random.uniform(0, 20)
        system = System(
            n,
            batch_cost,
            is_series=True,
            costs=costs,
            probabilities=succ_probabilities,
        )
        _, optimal_batch_cost1 = batching.optimal_batching(system, list(range(n)))
        optimal_batch_cost2, batched_solution = optimal_batching(p_t, batch_cost)
        optimal_batch_cost3 = sum(
            [
                p_t[i] * get_cost(batched_solution, i, len(p_t), batch_cost)
                for i in range(len(p_t))
            ]
        )
        print(optimal_batch_cost1)
        print(batched_solution)
