import itertools
import os
import sys

import pytest

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../sequential_batch_testing"
)

import batching
import numpy as np
import policy_evaluations
import unbatched_policy_computations
from helper import generate_cases, generate_cases_fixed_batch
from initialization import System


class TestOptimalBatching:
    @pytest.mark.parametrize("params", generate_cases())
    def test_optimal_batching_terminates(self, params):
        n = params["n"]
        system = System(**params)
        unbatched_policy = unbatched_policy_computations.compute_unbatched_policy(
            system
        )

        batched_policy, best_cost = batching.optimal_batching(system, unbatched_policy)
        batched_policy_items = list(itertools.chain(*batched_policy))
        assert isinstance(best_cost, float), "Batched policy cost should be a float."
        assert len(batched_policy_items) == n, "All items should be tested."
        assert set(batched_policy_items) == set(range(n)), "All items should be tested."
        assert best_cost == pytest.approx(
            policy_evaluations.evaluate_k_of_n_systems(system, batched_policy)
        ), "Batch cost computed by the algorithm should be the same as manually recalculating."
        cost_single_batch = policy_evaluations.evaluate_k_of_n_systems(
            system, [unbatched_policy]
        )
        cost_singleton = policy_evaluations.evaluate_k_of_n_systems(
            system, [[i] for i in unbatched_policy]
        )

        assert best_cost <= cost_single_batch or best_cost == pytest.approx(
            cost_single_batch
        ), "Optimal batching should be better than testing each item individually."
        assert best_cost <= cost_singleton or best_cost == pytest.approx(
            cost_singleton
        ), "Optimal batching should be better than testing everything at once."

    @pytest.mark.parametrize("params", generate_cases_fixed_batch())
    def test_optimal_for_0_batch_cost(self, params):
        n = params["n"]
        system = System(**params)
        unbatched_policy = unbatched_policy_computations.compute_unbatched_policy(
            system
        )

        batched_policy, best_cost = batching.optimal_batching(system, unbatched_policy)
        batched_policy_items = list(itertools.chain(*batched_policy))
        assert isinstance(best_cost, float), "Batched policy cost should be a float."
        assert len(batched_policy) == n or best_cost == pytest.approx(
            policy_evaluations.evaluate_k_of_n_systems(
                system, [[x] for x in unbatched_policy]
            )
        ), "All items should be tested one by one."
        assert set(batched_policy_items) == set(range(n)), "All items should be tested."
        assert best_cost == pytest.approx(
            policy_evaluations.evaluate_k_of_n_systems(system, batched_policy)
        ), "Batch cost computed by the algorithm should be the same as manually recalculating."
        cost_single_batch = policy_evaluations.evaluate_k_of_n_systems(
            system, [unbatched_policy]
        )
        cost_singleton = policy_evaluations.evaluate_k_of_n_systems(
            system, [[i] for i in unbatched_policy]
        )

        assert best_cost <= cost_single_batch or best_cost == pytest.approx(
            cost_single_batch
        ), "Optimal batching should be better than testing each item individually."
        assert best_cost == pytest.approx(
            cost_singleton
        ), "Optimal batching should be better than testing everything at once."


class TestRandomizedBatching:
    @pytest.mark.parametrize("params", generate_cases())
    def test_randomized_batching_series(self, params):
        if params["batch_cost"] == 0:
            return
        n = params["n"]
        system = System(**params)

        unbatched_policy = unbatched_policy_computations.compute_unbatched_policy(
            system
        )
        batched_policy = batching.randomized_batching(system, unbatched_policy)
        batched_policy_items = list(itertools.chain(*batched_policy))
        assert len(batched_policy_items) == n, "Batched policy does not have 10 batches"
        assert set(itertools.chain(*batched_policy)) == set(range(n))
        cost = policy_evaluations.evaluate_k_of_n_systems(system, batched_policy)
        optimal_policy, optimal_cost = batching.optimal_batching(
            system, unbatched_policy
        )
        batched_policy = [tuple(p) for p in batched_policy]
        assert optimal_cost <= cost or (
            optimal_cost == pytest.approx(cost) and optimal_policy == batched_policy
        ), "Randomized batching should be worse than optimal batching."


class TestDerandomizedBatching:
    @pytest.mark.parametrize("params", generate_cases())
    def test_find_critical_value(self, params):
        if params["batch_cost"] == 0:
            return
        system = System(**params)
        gamma = system.batch_cost * system.r
        unbatched_policy = unbatched_policy_computations.compute_unbatched_policy(
            system
        )
        unbatched_order = [system.components[i] for i in unbatched_policy]
        cum_cost = np.cumsum([x.cost for x in unbatched_order])
        critical_values = batching.find_critical_values(cum_cost, gamma)
        crit_value_batches = set()
        lin_value_batches = set()
        for val in critical_values:
            crit_value_batch = batching.assign_batches(
                cum_cost, gamma, val, unbatched_policy
            )
            crit_value_batch = tuple(tuple(p) for p in crit_value_batch)
            crit_value_batches.add(crit_value_batch)
        for val2 in np.linspace(0, 1, 500):
            lin_value_batch = batching.assign_batches(
                cum_cost, gamma, val2, unbatched_policy
            )
            lin_value_batch = tuple(tuple(p) for p in lin_value_batch)
            lin_value_batches.add(lin_value_batch)
        assert crit_value_batches.issuperset(lin_value_batches)
        assert len(crit_value_batches) >= len(lin_value_batches)

    @pytest.mark.parametrize("params", generate_cases())
    def test_derandomized_batching(self, params):
        if params["batch_cost"] == 0:
            return
        system = System(**params)
        unbatched_policy = unbatched_policy_computations.compute_unbatched_policy(
            system
        )
        batched_policy = batching.derandomized_batching(system, unbatched_policy)
        batched_policy_items = list(itertools.chain(*batched_policy))
        assert (
            len(batched_policy_items) == params["n"]
        ), "Batched policy should cover all items."
        derandomized_cost = policy_evaluations.evaluate_k_of_n_systems(
            system, batched_policy
        )
        randomized_policy = batching.randomized_batching(system, unbatched_policy)
        randomized_cost = policy_evaluations.evaluate_k_of_n_systems(
            system, randomized_policy
        )
        assert derandomized_cost <= randomized_cost or (
            derandomized_cost == pytest.approx(randomized_cost)
            and batched_policy == randomized_policy
        ), "Derandomized batching should be better than randomized batching."
        optimal_policy, optimal_cost = batching.optimal_batching(
            system, unbatched_policy
        )
        assert derandomized_cost >= optimal_cost or (
            derandomized_cost == pytest.approx(optimal_cost)
            and batched_policy == tuple(optimal_policy)
        ), "Derandomized batching should be worse than optimal batching."
