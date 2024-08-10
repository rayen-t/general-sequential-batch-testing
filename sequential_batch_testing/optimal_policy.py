from itertools import permutations

from batching import optimal_batching

import initialization


def compute_optimal_policy(system: initialization.System) -> tuple:
    """Compute the optimal policy for the system

    Args:
        system (System): The parameters of the system

    Returns:
        tuple: The optimal policy and the expected cost
    """
    component_list = list(range(system.n))
    best_cost = float("inf")
    best_policy = None
    for unbatched_policy in permutations(component_list):
        batched_policy, cost = optimal_batching(system, unbatched_policy)
        if cost < best_cost:
            best_cost = cost
            best_policy = batched_policy
    return best_policy, best_cost
