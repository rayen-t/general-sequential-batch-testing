from itertools import permutations
from typing import Sequence

import numpy as np
from initialization import System
from policy_evaluations import evaluate_k_of_n_systems, evaluate_unbatched_policy


def compute_unbatched_policy(system: System) -> Sequence:
    if system.is_series:
        return compute_unbatched_series_policy(system)
    elif system.k is not None and not system.unit_cost:
        return compute_unbatched_k_of_n_policy(system)
    elif system.k is not None and system.unit_cost:
        return compute_unbatched_k_of_n_unit_cost_policy(system)
    elif system.k is None:
        return compute_unbatched_ssc_policy(system)
    else:
        raise ValueError("Invalid system configuration")


def compute_unbatched_series_policy(system: System) -> tuple:
    """Compute the greedy policy for the series system,
    which is known to be optimal

    Args:
        system (System): The parameters of the system

    Returns:
        tuple: Optimal policy for the system, given by the greedy algorithm
    """

    ratios = [x.cost / (1 - x.probability) for x in system.components]
    return tuple(np.argsort(ratios))


def compute_unbatched_parallel_policy(system: System) -> tuple:
    """Compute the greedy policy for the parallel system,
    which is known to be optimal

    Args:
        system (System): The parameters of the system

    Returns:
        tuple: Optimal policy for the system, given by the greedy algorithm
    """

    ratios = [x.cost / x.probability for x in system.components]
    return tuple(np.argsort(ratios))


def compile_ALG_2(system: System) -> Sequence:
    """Compile the ALG_2 algorithm for the k-of-n system

    Args:
        system (System): The parameters of the system

    Returns:
        dict: The ALG_2 algorithm
    """

    ALG = [(), ()]
    ALG[0] = compute_unbatched_series_policy(system)
    ALG[1] = compute_unbatched_parallel_policy(system)
    return ALG


def compute_unbatched_k_of_n_policy(system: System) -> Sequence:
    ALG = compile_ALG_2(system)
    unbatched_policy = round_robin(system, ALG, (1, 1))
    return unbatched_policy


def compute_unbatched_k_of_n_unit_cost_policy(system: System) -> Sequence:
    alg_1 = compute_unbatched_series_policy(system)
    alg_2 = compute_unbatched_parallel_policy(system)

    batch_cost_store = system.batch_cost
    system.batch_cost = 0

    score_1 = evaluate_k_of_n_systems(system, [[x] for x in alg_1])
    score_2 = evaluate_k_of_n_systems(system, [[x] for x in alg_2])
    system.batch_cost = batch_cost_store

    if score_1 < score_2:
        return alg_1
    else:
        return alg_2


def sort_increasing_cost(system: System) -> Sequence:
    return tuple(np.argsort([x.cost for x in system.components]))


def compute_unbatched_ssc_policy(system: System) -> Sequence:

    ALG = [(), (), ()]
    ALG[0] = compute_unbatched_series_policy(system)
    ALG[1] = compute_unbatched_parallel_policy(system)
    ALG[2] = sort_increasing_cost(system)

    return round_robin(system, ALG, (1, 1, np.sqrt(2)))


def compute_optimal_unbatched_policy(system: System) -> Sequence:

    best_cost = np.inf
    best_policy = None

    for perm in permutations(range(system.n)):

        cost = evaluate_unbatched_policy(system, perm)

        if cost < best_cost:
            best_cost = cost
            best_policy = perm

    assert best_policy is not None
    return best_policy


def round_robin(system: System, ALG: Sequence, alpha: Sequence) -> Sequence:
    """Compute the greedy policy for the k-of-n system,
    using the round-robin algorithm

    Args:
        system (System): The parameters of the system

    Returns:
        tuple: Optimal policy for the system, given by the greedy algorithm
    """
    k = len(ALG)

    if k != len(alpha):
        raise ValueError("ALG and alpha must have the same length")

    C = np.zeros(k, dtype=int)
    unbatched_policy = []
    processed = set()
    pointers = np.zeros(k, dtype=int)

    while len(unbatched_policy) < system.n:
        h_star = None
        min_value = np.inf

        for h in range(k):
            # Skip over components that have already been processed
            while pointers[h] < system.n and ALG[h][pointers[h]] in processed:
                pointers[h] += 1

            # If we have not reached the end of the list
            if pointers[h] < system.n:
                next_item = ALG[h][pointers[h]]
                delta_h = system.components[next_item].cost
                choice_value = (C[h] + delta_h) / alpha[h]
                if choice_value < min_value:
                    h_star = h
                    min_value = choice_value

        if h_star is None:
            raise ValueError("No component to test")

        next_test = ALG[h_star][pointers[h_star]]
        unbatched_policy.append(next_test)
        processed.add(ALG[h_star][pointers[h_star]])
        C[h_star] += system.components[next_test].cost

    assert len(unbatched_policy) == system.n and set(unbatched_policy) == set(
        range(system.n)
    )

    return tuple(unbatched_policy)


if __name__ == "__main__":
    system = System(10, 10, 1, interval_breakpoints=[3, 7])
    print(compute_unbatched_ssc_policy(system))
