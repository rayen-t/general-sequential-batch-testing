import os
import sys


from helper import generate_cases
import pytest

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../sequential_batch_testing"
)


from initialization import System
from policy_evaluations import (
    evaluate_k_of_n_systems,
    evaluate_k_of_n_systems_slow,
    evaluate_series_system,
)
from unbatched_policy_computations import (
    compute_unbatched_policy,
    compute_unbatched_series_policy,
    compute_unbatched_parallel_policy,
    compute_unbatched_k_of_n_policy,
)


class TestPolicyEvaluations:
    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    @pytest.mark.parametrize("batch_cost", ["n", "n/2", "n/4", 0, 1000])
    @pytest.mark.parametrize("problem_type", [1, 2])
    def test_evaluation_for_series_system(self, n, batch_cost, problem_type):
        system = System(n, batch_cost, problem_type, k=n)
        policy_unbatched = compute_unbatched_series_policy(system)

        policy = [
            policy_unbatched[i : i + 2] for i in range(0, len(policy_unbatched), 2)
        ]
        seies_baseline = evaluate_series_system(system, policy)

        production = evaluate_k_of_n_systems(system, policy)
        if system.k is not None:
            slow_code = evaluate_k_of_n_systems_slow(system, policy)
            assert slow_code == pytest.approx(
                seies_baseline
            ), "Slow code should match baseline"
        assert production == pytest.approx(
            seies_baseline
        ), "Production code should match baseline"

        policy_full = [list(range(n))]
        expected = system.batch_cost + sum(
            [system.components[i].cost for i in range(n)]
        )
        assert evaluate_series_system(system, policy_full) == pytest.approx(expected)
        assert evaluate_k_of_n_systems(system, policy_full) == pytest.approx(expected)

    @pytest.mark.parametrize("params", generate_cases())
    def test_evaluation_for_kn(self, params):
        n = params["n"]
        system = System(**params)
        policy_unbatched = compute_unbatched_policy(system)

        policy = [
            policy_unbatched[i : i + 2] for i in range(0, len(policy_unbatched), 2)
        ]
        if system.k is not None:
            seies_baseline = evaluate_k_of_n_systems_slow(system, policy)
            production = evaluate_k_of_n_systems(system, policy)
            assert production == pytest.approx(seies_baseline)

        policy_full = [list(range(n))]
        expected = system.batch_cost + sum(
            [system.components[i].cost for i in range(n)]
        )
        assert evaluate_k_of_n_systems(system, policy_full) == pytest.approx(expected)
        if system.k is not None:
            assert evaluate_k_of_n_systems_slow(system, policy_full) == pytest.approx(
                expected
            )
