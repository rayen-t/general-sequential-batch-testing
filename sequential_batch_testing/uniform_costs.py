import numpy as np


def recover_parent_pointer(parent_pointers, n):
    current = 0
    batched_policy = []

    while current < n:
        batch_size = parent_pointers[current]
        if batch_size == "D":
            batched_policy.append("D")
            return batched_policy
        batched_policy.append(batch_size)
        current += batch_size
    return batched_policy


def batch_uniform_costs_and_probabilities(n, p, c, batch_cost, dummy_item_cost):
    memo = {}
    parent_pointers = {}

    def recurs(i, n, p, c, batch_cost, dummy_item_cost):
        if i == n:
            return dummy_item_cost + batch_cost
        if i not in memo:
            min_cost = np.inf
            best_action = None
            for batch_size in range(1, n - i + 1):
                cost = (
                    batch_cost
                    + c * batch_size
                    + (p**batch_size)
                    * recurs(i + batch_size, n, p, c, batch_cost, dummy_item_cost)
                )
                if cost < min_cost:
                    min_cost = cost
                    best_action = batch_size
            if dummy_item_cost + batch_cost < min_cost:
                min_cost = dummy_item_cost + batch_cost
                best_action = "D"
            memo[i] = min_cost
            parent_pointers[i] = best_action
        return memo[i]

    recurs(0, n, p, c, batch_cost, dummy_item_cost)
    print(memo)
    batching = recover_parent_pointer(parent_pointers, n)
    return memo[0], batching


if __name__ == "__main__":
    n = 4
    p = 0.86
    c = 3
    batch_cost = 2
    D = 33
    print(alg := batch_uniform_costs_and_probabilities(n, p, c, batch_cost, D))
