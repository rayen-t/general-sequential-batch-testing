import numpy as np
import policy_evaluations
from poisson_binomial import (
    compute_probability_of_not_stopping,
    compute_probability_of_not_stopping_interval,
)


def unwind_optimal_batch(parent_pointers, unbatched_policy):
    current = 0
    batched_policy = []

    while current < len(unbatched_policy):
        parent = parent_pointers[current]
        batched_policy.append(unbatched_policy[current:parent])
        current = parent
    return batched_policy


def optimal_batching(system, unbatched_policy):
    memo = {}
    parent_pointers = {}
    unbatched_order = [system.components[i] for i in unbatched_policy]
    if system.k is not None:
        probability_of_not_stopping = compute_probability_of_not_stopping(
            system.k, [x.probability for x in unbatched_order]
        )
    else:
        probability_of_not_stopping = compute_probability_of_not_stopping_interval(
            system, [x.probability for x in unbatched_order]
        )

    def recurs(item_index):
        if item_index >= len(unbatched_policy):
            return 0
        elif item_index in memo:
            return memo[item_index]
        else:
            min_cost = np.inf
            min_idx = None
            for end in range(item_index + 1, len(unbatched_policy) + 1):
                batch = unbatched_policy[item_index:end]
                total_batch_cost = (
                    sum([system.components[i].cost for i in batch]) + system.batch_cost
                )
                cost = probability_of_not_stopping[
                    item_index
                ] * total_batch_cost + recurs(end)
                if cost < min_cost:
                    min_cost = cost
                    min_idx = end
            memo[item_index] = min_cost
            parent_pointers[item_index] = min_idx
            return min_cost

    recurs(0)
    batched_policy = unwind_optimal_batch(parent_pointers, unbatched_policy)
    return batched_policy, memo[0]


def assign_batches(cum_cost, gamma, epsilon, unbatched_policy):
    batch_numbers = [-1] * len(unbatched_policy)
    for idx, time in enumerate(cum_cost):
        try:
            batch_numbers[idx] = 1 + int(np.floor(time / gamma - epsilon))
        except:
            raise ValueError(f"{time=} {gamma=} {epsilon=}")
    batched_policy = [[] for _ in range(max(batch_numbers) + 1)]

    for idx, batch in enumerate(batch_numbers):
        batched_policy[batch].append(unbatched_policy[idx])
    batched_policy = tuple(tuple(x) for x in batched_policy if x)
    return batched_policy


def randomized_batching(system, unbatched_policy):
    offset = np.random.rand()
    unbatched_order = [system.components[i] for i in unbatched_policy]
    cum_cost = np.cumsum([x.cost for x in unbatched_order])
    batched_policy = assign_batches(cum_cost, system.gamma, offset, unbatched_policy)
    return batched_policy


def find_critical_values(cum_cost, gamma):
    corner_values = sorted([(time % gamma) / gamma for time in cum_cost])

    corner_values = [0] + corner_values + [1]
    critical_values = [
        (corner_values[i] + corner_values[i + 1]) / 2
        for i in range(len(corner_values) - 1)
    ]

    assert len(critical_values) == len(corner_values) - 1

    return critical_values


def derandomized_batching(system, unbatched_policy):
    unbatched_order = [system.components[i] for i in unbatched_policy]
    cum_cost = np.cumsum([x.cost for x in unbatched_order])
    critical_values = find_critical_values(cum_cost, system.gamma)

    best_cost = np.inf
    best_policy = None

    for epsilon in critical_values:
        batch = assign_batches(cum_cost, system.gamma, epsilon, unbatched_policy)
        cost = policy_evaluations.evaluate_k_of_n_systems(system, batch)
        if cost < best_cost:
            best_cost = cost
            best_policy = batch
    print(f"{best_cost=}")
    return best_policy
