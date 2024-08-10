import itertools
from typing import Sequence

import numpy as np
from poisson_binomial import (
    compute_probability_of_not_stopping,
    compute_probability_of_not_stopping_interval,
    poisson_binomial_pdf,
)


def evaluate_series_system(
    system,
    unbatched_policy,
) -> float:
    """Evaluate the expected cost of the policy using dynamic programing.

    Args:
        system (System): The system to be evaluated.
        policy (tuple): The policy to be evaluated.

    Returns:
        float: The expected cost of the policy.
    """
    num_batches = len(unbatched_policy)

    def recurs(batch_index):
        # Base case: if we have tested all batches, return 0
        if batch_index >= num_batches:
            return 0
        else:
            # Inductive step: return the cost of the current batch plus the expected cost of the rest of the batches
            batch = unbatched_policy[batch_index]
            return (
                system.batch_cost
                + sum([system.components[i].cost for i in batch])
                + np.prod([system.components[i].probability for i in batch])
                * recurs(batch_index + 1)
            )

    return recurs(0)


def evaluate_k_of_n_systems_slow(system, policy):
    memo = {}

    def recurs(batch_index, number_passed, number_tested):
        if (
            batch_index >= len(policy)
            or number_passed >= system.k
            or number_passed + (system.n - number_tested) < system.k
        ):
            return 0

        elif (batch_index, number_passed, number_tested) in memo:
            return memo[(batch_index, number_passed, number_tested)]

        else:
            batch = policy[batch_index]
            batch_items = [system.components[i] for i in batch]
            testing_cost = sum([x.cost for x in batch_items])
            expected_cost = system.batch_cost + testing_cost

            for k in range(len(batch) + 1):
                increase = poisson_binomial_pdf(
                    k, [x.probability for x in batch_items]
                ) * recurs(
                    batch_index + 1, number_passed + k, number_tested + len(batch)
                )
                expected_cost += increase

            memo[(batch_index, number_passed, number_tested)] = expected_cost
            return expected_cost

    recurs(0, 0, 0)
    return memo[(0, 0, 0)]


def evaluate_unbatched_policy(system, unbatched_policy: Sequence[int]) -> float:
    """Given a system and an unbatched policy, evaluate the expected cost of the policy.

    Args:
        system (System): Parameters of the system
        unbatched_policy (tuple[int]): Unbatched policy given as a tuple of integers

    Returns:
        float: Expected cost incurred to determine the state of the system.
    """
    memo = {}

    def recurs(number_passed, number_tested):
        if number_tested >= len(unbatched_policy) or (
            system.determine_interval(number_passed, number_tested - number_passed)
            != -1
        ):
            return 0

        elif (number_passed, number_tested) in memo:
            return memo[(number_passed, number_tested)]

        else:
            test = unbatched_policy[number_tested]
            test_cost = system.components[test].cost
            test_probability = system.components[test].probability
            expected_cost = (
                test_cost
                + test_probability * recurs(number_passed + 1, number_tested + 1)
                + (1 - test_probability) * recurs(number_passed, number_tested + 1)
            )
            memo[(number_passed, number_tested)] = expected_cost
            return expected_cost

    recurs(0, 0)
    return memo[(0, 0)]


def evaluate_k_of_n_systems(system, policy):

    testing_sequence = list(itertools.chain(*policy))
    probability_sequence = [system.components[i].probability for i in testing_sequence]
    if system.k is not None:
        probabilities_of_not_stopping = compute_probability_of_not_stopping(
            system.k, probability_sequence
        )
    else:
        probabilities_of_not_stopping = compute_probability_of_not_stopping_interval(
            system, probability_sequence
        )

    cum_cost = 0
    number_of_components_seen = 0
    for batch in policy:
        batch_items = [system.components[i] for i in batch]
        testing_cost = sum([x.cost for x in batch_items])
        expected_cost = system.batch_cost + testing_cost

        cum_cost += (
            probabilities_of_not_stopping[number_of_components_seen] * expected_cost
        )
        number_of_components_seen += len(batch)

    return cum_cost
