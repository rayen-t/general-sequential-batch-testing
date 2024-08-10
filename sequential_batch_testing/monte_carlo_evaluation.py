from typing import Sequence

import initialization
import numpy as np
import unbatched_policy_computations


def stopping_rule(system, number_passed, number_failed):
    tests_remaining = system.n - number_passed - number_failed
    lower_bound = number_passed
    upper_bound = number_passed + tests_remaining

    stop_flag = lower_bound >= system.k or upper_bound < system.k
    assert stop_flag == (system.determine_interval(number_passed, number_failed) != -1)
    return stop_flag


def monte_carlo_costs(system, policy: Sequence, replications: int = 10000):
    results = np.empty(replications, dtype=int)
    for rep_no in range(replications):
        rands = np.random.rand(system.n)
        instance = np.array(
            [
                1 if x < system.components[i].probability else 0
                for i, x in enumerate(rands)
            ]
        )
        number_passed = 0
        number_failed = 0
        cost_incurred = 0
        for batch in policy:
            cost_incurred += system.batch_cost
            cost_incurred += sum([system.components[i].cost for i in batch])
            batch_passed = sum([instance[i] for i in batch])
            number_passed += batch_passed
            number_failed += len(batch) - batch_passed

            if stopping_rule(system, number_passed, number_failed):
                break
        results[rep_no] = cost_incurred
    return results.mean()


if __name__ == "__main__":
    n = 100
    k = 100
    r = 1
    batch_cost = "n"
    system = initialization.System(n, batch_cost, r, k=k)
    print(system.components)
    greedy_order = unbatched_policy_computations.compute_unbatched_series_policy(system)
    greedy_order = [[i] for i in greedy_order]
    E_greedy = monte_carlo_costs(system, greedy_order).mean()
    E_random = monte_carlo_costs(system, [list(range(n))]).mean()
    print(E_greedy, E_random)
