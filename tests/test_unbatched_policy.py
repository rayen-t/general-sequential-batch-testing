import os
import sys

import pytest
from helper import generate_cases, generate_cases_fixed_batch, generate_cases_unit_cost

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../sequential_batch_testing"
)


from initialization import System
from policy_evaluations import (
    evaluate_k_of_n_systems,
    evaluate_series_system,
    evaluate_unbatched_policy,
)
from unbatched_policy_computations import (
    compile_ALG_2,
    compute_optimal_unbatched_policy,
    compute_unbatched_k_of_n_policy,
    compute_unbatched_k_of_n_unit_cost_policy,
    compute_unbatched_parallel_policy,
    compute_unbatched_policy,
    compute_unbatched_series_policy,
    compute_unbatched_ssc_policy,
    sort_increasing_cost,
)


class TestSeriesPolicy:
    def test_greedy_order(self):
        n = 5
        system1 = System(n, "n", 1, k=n)
        greedy_policy = compute_unbatched_series_policy(system1)
        assert (
            1 - system1.components[greedy_policy[0]].probability
        ) / system1.components[greedy_policy[0]].cost >= (
            1 - system1.components[greedy_policy[-1]].probability
        ) / system1.components[
            greedy_policy[-1]
        ].cost

        greedy_policy_ratio = [
            (1 - system1.components[i].probability) / system1.components[i].cost
            for i in greedy_policy
        ]
        assert greedy_policy_ratio == sorted(greedy_policy_ratio, reverse=True)

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    @pytest.mark.parametrize("problem_type", [1, 2])
    def test_greedy_is_optimal(self, n, problem_type):
        system1 = System(n, 0, problem_type, k=n)
        greedy_policy = compute_unbatched_series_policy(system1)
        greedy_singleton_policy = [[i] for i in greedy_policy]
        greedy_cost1 = evaluate_series_system(system1, greedy_singleton_policy)
        greedy_cost2 = evaluate_k_of_n_systems(system1, greedy_singleton_policy)
        assert greedy_cost1 == pytest.approx(greedy_cost2)
        optimal_policy = compute_optimal_unbatched_policy(system1)
        optimal_singleton_policy = [[i] for i in optimal_policy]
        optimal_cost = evaluate_k_of_n_systems(system1, optimal_singleton_policy)
        assert greedy_cost1 == pytest.approx(optimal_cost)


class TestParallelPolicy:
    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    @pytest.mark.parametrize("problem_type", [1, 2])
    def test_greedy_is_optimal(sel, n, problem_type):
        system1 = System(n, 0, problem_type, k=1)
        greedy_policy = compute_unbatched_parallel_policy(system1)
        greedy_singleton_policy = [[i] for i in greedy_policy]
        greedy_cost = evaluate_k_of_n_systems(system1, greedy_singleton_policy)
        assert greedy_cost == pytest.approx(greedy_cost)
        optimal_policy = compute_optimal_unbatched_policy(system1)
        optimal_singleton_policy = [[i] for i in optimal_policy]
        optimal_cost = evaluate_k_of_n_systems(system1, optimal_singleton_policy)
        assert greedy_cost == pytest.approx(optimal_cost)


class TestKofNPolicy:
    def test_compile_ALG_2(self):
        n = 10
        k = 10
        r = 1
        batch_cost = "n"
        system = System(n, batch_cost, r, k=k)
        ALG = compile_ALG_2(system)
        ALG_succ = ALG[1]
        ratios = [
            system.components[i].cost / system.components[i].probability
            for i in ALG_succ
        ]

        assert ratios[0] <= ratios[-1]
        for i in range(0, len(ratios) - 1):
            assert ratios[i] <= ratios[i + 1]
        ALG_fail = ALG[0]
        ratios = [
            system.components[i].cost / (1 - system.components[i].probability)
            for i in ALG_fail
        ]
        assert ratios[0] <= ratios[-1]
        for i in range(0, len(ratios) - 1):
            assert ratios[i] <= ratios[i + 1]

    @pytest.mark.parametrize("params", generate_cases())
    def test_k_of_n_policy(self, params):
        system = System(**params)
        system.batch_cost = 0
        unbatched_policy = compute_unbatched_k_of_n_policy(system)
        policy_cost = evaluate_k_of_n_systems(system, [[i] for i in unbatched_policy])
        optimal_policy = compute_optimal_unbatched_policy(system)
        optimal_cost = evaluate_k_of_n_systems(system, [[i] for i in optimal_policy])
        assert optimal_cost <= policy_cost or optimal_cost == pytest.approx(
            policy_cost
        ), "Optimal policy should be better than policy"
        assert policy_cost <= 2 * optimal_cost, "Policy should a 2 approximation"


class TestUnbatchedPolicy:
    @pytest.mark.parametrize("params", generate_cases())
    def test_correct_algorithm_selected(self, params):
        system = System(**params)
        unbatched_policy = compute_unbatched_policy(system)
        if system.is_series:
            assert unbatched_policy == compute_unbatched_series_policy(system)
        elif system.k is not None and system.unit_cost:
            assert unbatched_policy == compute_unbatched_k_of_n_unit_cost_policy(system)
        elif system.k is not None:
            assert unbatched_policy == compute_unbatched_k_of_n_policy(system)
        else:
            assert unbatched_policy == compute_unbatched_ssc_policy(system)


class TestKofNUnitCostPolicy:
    @pytest.mark.parametrize("params", generate_cases_unit_cost())
    def test_policy_better_than_greedy(self, params):
        system = System(**params)
        unbatched_policy = compute_unbatched_k_of_n_unit_cost_policy(system)
        policy_cost = evaluate_k_of_n_systems(system, [[i] for i in unbatched_policy])
        greedy_policy1 = compute_unbatched_series_policy(system)
        greedy_cost1 = evaluate_k_of_n_systems(system, [[i] for i in greedy_policy1])
        greedy_policy2 = compute_unbatched_parallel_policy(system)
        greedy_cost2 = evaluate_k_of_n_systems(system, [[i] for i in greedy_policy2])
        optimal_policy = compute_optimal_unbatched_policy(system)
        optimal_cost = evaluate_k_of_n_systems(system, [[i] for i in optimal_policy])
        assert policy_cost <= greedy_cost1 or policy_cost == pytest.approx(
            greedy_cost1
        ), "Policy should be better than greedy series"
        assert policy_cost <= greedy_cost2 or policy_cost == pytest.approx(
            greedy_cost2
        ), "Policy should be better than greedy parallel"
        assert optimal_cost <= policy_cost or optimal_cost == pytest.approx(
            policy_cost
        ), "Optimal policy should be better than policy"
        assert policy_cost <= 1.5 * optimal_cost, "Policy should a 1.5 approximation"


class TestSSCPolicy:
    def test_decreasing_cost(self):
        n = 5
        system1 = System(n, "n", 1, k=n)
        inc = sort_increasing_cost(system1)
        assert system1.components[inc[0]].cost <= system1.components[inc[-1]].cost
        assert all(
            [system1.components[inc[i]].cost <= system1.components[inc[i + 1]].cost]
            for i in range(n - 1)
        )

    @pytest.mark.parametrize("n", [5, 6, 7])
    @pytest.mark.parametrize("problem_type", [1, 2])
    @pytest.mark.parametrize("batch_cost", ["n", "n/2", "n/4", 0, 1000])
    @pytest.mark.parametrize("unit_cost", [True, False])
    @pytest.mark.parametrize(
        "interval_breakpoints", [[1], [1, 2, 3, 4, 5], [2, 4], [1, 3, 5], [5], [1, 5]]
    )
    def test_correct_items(
        self, n, problem_type, batch_cost, unit_cost, interval_breakpoints
    ):
        system = System(
            n,
            batch_cost,
            problem_type,
            unit_cost=unit_cost,
            interval_breakpoints=interval_breakpoints,
        )
        unbatched_policy = compute_unbatched_policy(system)
        assert set(unbatched_policy) == set(
            range(n)
        ), "All items should be in the policy"

    @pytest.mark.parametrize("n", [5, 6, 7])
    @pytest.mark.parametrize("problem_type", [1, 2])
    @pytest.mark.parametrize("unit_cost", [True, False])
    @pytest.mark.parametrize(
        "interval_breakpoints", [[1], [1, 2, 3, 4, 5], [2, 4], [1, 3, 5], [5], [1, 5]]
    )
    def test_optimal_correct_items(
        self, n, problem_type, unit_cost, interval_breakpoints
    ):
        system = System(
            n,
            0,
            problem_type,
            unit_cost=unit_cost,
            interval_breakpoints=interval_breakpoints,
        )
        unbatched_policy = compute_optimal_unbatched_policy(system)
        assert set(unbatched_policy) == set(
            range(n)
        ), "All items should be in the policy"


class TestUnbatchedScoring:
    @pytest.mark.parametrize("params", generate_cases_fixed_batch())
    def test_unbatched_scoring(self, params):
        system = System(**params)
        unbatched_policy = compute_unbatched_policy(system)
        policy_cost = evaluate_unbatched_policy(system, unbatched_policy)
        policy_cost2 = evaluate_k_of_n_systems(system, [[i] for i in unbatched_policy])
        assert policy_cost == pytest.approx(policy_cost2)
        optimal_policy = compute_optimal_unbatched_policy(system)
        optimal_cost = evaluate_unbatched_policy(system, optimal_policy)
        optimal_cost2 = evaluate_k_of_n_systems(system, [[i] for i in optimal_policy])
        assert optimal_cost == pytest.approx(optimal_cost2)
