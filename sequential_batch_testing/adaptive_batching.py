import bisect
import time
from itertools import chain, combinations
from typing import Iterable, Sequence

import numpy as np
from initialization import System
from optimal_policy import compute_optimal_policy


def powerset(iterable):
    """
    powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    Note that the empty set is removed.
    We never want empty batches.
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs, n) for n in range(1, len(xs) + 1))


def reverse_powerset(iterable):
    """powerset([1,2,3]) --> (1, 2, 3) (1, 2) (1, 3) (2, 3) (1,) (2,) (3,) (,)
    For tabular DP.

    """
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs), -1, -1))


def generate_possible_batches(
    tested_indices: Iterable[int], n: int
) -> Iterable[frozenset]:
    """Generate all possible batches excluding what was already tested."""
    untested_indices = tuple(x for x in range(n) if x not in tested_indices)
    for location in powerset(untested_indices):
        yield frozenset(location)


def generate_stopping_lookup(n: int, intervals) -> np.ndarray:
    """Gives lookup table based where
    lookup[number_tested, number_passed] = True if we should stop testing
    """
    stopping_lookup = np.empty((n + 1, n + 1), dtype=bool)
    stopping_lookup[0, :] = False
    for number_tested in range(1, n + 1):
        for number_passed in range(0, n + 1):
            tests_remaining = n - number_tested
            lower_bound = number_passed
            upper_bound = number_passed + tests_remaining

            interval_index = bisect.bisect(intervals, lower_bound)
            if interval_index >= len(intervals):
                stopping_lookup[number_tested, number_passed] = True
            else:
                interval_boundary = intervals[interval_index]
                stopping_lookup[number_tested, number_passed] = (
                    upper_bound < interval_boundary
                )
    return stopping_lookup


def reached_stopping_condition(stopping_lookup, number_tested, number_passed):
    return stopping_lookup[number_tested, number_passed]


def generate_probability_lookup_table(
    probabilities,
) -> dict:
    n = len(probabilities)
    table = {}
    for s in range(n):
        s_set = frozenset([s])
        table[s_set, 0] = 1 - probabilities[s]
        table[s_set, 1] = probabilities[s]
    for s in powerset(range(n)):
        if len(s) == 1:
            continue
        a, *b = s
        s = frozenset(s)
        b = frozenset(b)
        table[s, 0] = (1 - probabilities[a]) * table[b, 0]
        table[s, len(s)] = probabilities[a] * table[b, len(s) - 1]
        for i in range(1, len(s)):
            table[s, i] = (
                probabilities[a] * table[b, i - 1]
                + (1 - probabilities[a]) * table[b, i]
            )
    return table


def recurs_dp(costs, probabilities, beta, intervals):
    memo = {}
    n = len(costs)
    probability_lookup_table = generate_probability_lookup_table(probabilities)
    stopping_lookup = generate_stopping_lookup(n, intervals)

    def recurs(tested_indices, number_passed):
        number_tested = len(tested_indices)
        if number_tested == len(costs):
            return 0
        elif reached_stopping_condition(stopping_lookup, number_tested, number_passed):
            return 0
        elif (tested_indices, number_passed) not in memo:
            min_cost = float("inf")

            for batch in generate_possible_batches(tested_indices, n):
                batch = batch
                newly_tested = tested_indices | batch
                testing_cost = sum(costs[i] for i in batch)
                value = beta + testing_cost
                for batch_passed in range(len(batch) + 1):
                    transition_probability = probability_lookup_table[
                        batch, batch_passed
                    ]
                    value += transition_probability * recurs(
                        newly_tested,
                        number_passed + batch_passed,
                    )
                if value < min_cost:
                    min_cost = value
            memo[tested_indices, number_passed] = min_cost
        return memo[tested_indices, number_passed]

    recurs(frozenset(), 0)
    return memo[frozenset(), 0]


def table_dp(
    costs: Sequence[float] | np.ndarray,
    probabilities,
    beta: float,
    intervals,
):
    """Compute the optimal adaptive policy using dynamic programming.

    Args:
        costs (_type_): _description_
        probabilities (_type_): _description_
        beta (_type_): _description_
        intervals (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = len(costs)
    probability_lookup_table = generate_probability_lookup_table(probabilities)
    stopping_lookup = generate_stopping_lookup(n, intervals)

    memo = {}
    for tested_indices in reverse_powerset(range(n)):
        tested_indices = frozenset(tested_indices)
        number_tested = len(tested_indices)
        for number_passed in range(number_tested + 1):
            if number_tested == n:
                memo[tested_indices, number_passed] = 0
            elif reached_stopping_condition(
                stopping_lookup, number_tested, number_passed
            ):
                memo[tested_indices, number_passed] = 0
            else:
                min_cost = float("inf")

                for batch in generate_possible_batches(tested_indices, n):
                    batch = batch
                    newly_tested = tested_indices | batch
                    testing_cost = sum(costs[i] for i in batch)
                    value = beta + testing_cost
                    for batch_passed in range(len(batch) + 1):
                        transition_probability = probability_lookup_table[
                            batch, batch_passed
                        ]
                        value += (
                            transition_probability
                            * memo[newly_tested, number_passed + batch_passed]
                        )
                    if value < min_cost:
                        min_cost = value
                memo[tested_indices, number_passed] = min_cost
    return memo[frozenset(), 0]


def test_all():
    for n in range(2, 10):
        for k in range(1, n + 1):
            for _ in range(20):
                print(f"testing n={n}, intervals={k}")
                beta = np.random.uniform(1, 10)
                costs = np.random.uniform(1, 10, n)
                probabilities = np.random.uniform(0, 1, n)
                assert len(costs) == len(probabilities)
                start1 = time.time()
                a = table_dp(costs, probabilities, beta, [k])
                end1 = time.time()
                sys = System(
                    n,
                    beta,
                    k=k,
                    costs=costs,
                    probabilities=probabilities,
                )
                start2 = time.time()
                _, b = compute_optimal_policy(sys)
                end2 = time.time()
                print(f"A: {end1 - start1}, NA: {end2 - start2}")
                print(a, b)
                if k == 1 or k == n:
                    assert np.isclose(a, b)
                else:
                    assert a <= b or np.isclose(a, b)
    for n in range(2, 10):
        for number_of_intervals in range(1, n + 1):
            for _ in range(20):
                intervals = np.random.choice(
                    range(1, n + 1), number_of_intervals, replace=False
                )
                intervals.sort()
                beta = np.random.uniform(1, 10)
                costs = np.random.uniform(1, 10, n)
                probabilities = np.random.uniform(0, 1, n)
                assert len(costs) == len(probabilities)
                start1 = time.time()
                a = table_dp(costs, probabilities, beta, intervals)
                end1 = time.time()
                sys = System(
                    n,
                    beta,
                    interval_breakpoints=intervals,
                    costs=costs,
                    probabilities=probabilities,
                )
                start2 = time.time()
                _, b = compute_optimal_policy(sys)
                end2 = time.time()
                print(f"A: {end1 - start1}, NA: {end2 - start2}")
                print(a, b)
                if len(intervals) == 1 and (intervals[0] == n or intervals[0] == 1):
                    assert np.isclose(a, b)
                else:
                    assert a <= b or np.isclose(a, b)


if __name__ == "__main__":
    n = 12
    k = 6
    print(f"testing n={n}, intervals={k}")
    beta = np.random.uniform(1, 10)
    costs = np.random.uniform(1, 10, n)
    probabilities = np.random.uniform(0, 1, n)
    assert len(costs) == len(probabilities)
    start1 = time.time()
    a = table_dp(costs, probabilities, beta, [k])
