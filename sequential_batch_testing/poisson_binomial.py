import itertools

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def poisson_binomial_pdf(k: int, p_list: Sequence[float]) -> float:
    """Implements the Poisson binomial distribution,
    which counts the number of successes like a binomial distribution,
    but with different probabilities for each trial.

    Args:
        k (int): Number of success
        p_list (list[float]): _description_

    Returns:
        Pr[X=k]: Probability of seeing k successes.
    """
    n = len(p_list)
    prob = 0
    for subset in itertools.combinations(range(n), k):
        product = np.prod(
            [p_list[i] if i in subset else 1 - p_list[i] for i in range(n)]
        )
        prob += product
    return float(prob)


def build_probability_table(p_list: Sequence[float]) -> NDArray:
    """Construct a lookup table, where (i, j) gives the probability that

    Args:
        p_list (Iterable[float]): _description_

    Returns:
        Iterable[float]: _description_
    """
    n = len(p_list)
    table = np.zeros((n + 1, n + 1))

    # Initialize base case.

    table[0, 0] = 1

    # Initialize row 0
    for j in range(1, n + 1):
        table[0, j] = (1 - p_list[j - 1]) * table[0, j - 1]
    # Fill in the table.
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            table[i, j] = (
                p_list[j - 1] * table[i - 1, j - 1]
                + (1 - p_list[j - 1]) * table[i, j - 1]
            )
    return table


def compute_probability_of_not_stopping(k: int, p_list: Sequence[float]) -> NDArray:
    """Compute the probability of not stopping before component i.

    Args:
        k (int): Number of success required for the system to pass
        p_list (Iterable[float]): Probability of each component passing

    Returns:
        Iterable[float]: Iterable where index $i$ is the probability of not stopping before component $i$.
    """
    n = len(p_list)
    table = build_probability_table(p_list)
    probability_of_not_stopping = np.zeros(n + 1)
    for i in range(n + 1):
        lowest_valid_index = max(i - n + k, 0)
        highest_valid_index = k
        probability_of_not_stopping[i] = table[
            lowest_valid_index:highest_valid_index, i
        ].sum()

    return probability_of_not_stopping


def compute_probability_of_not_stopping_interval(system, p_list) -> NDArray:
    """Compute the probability of not stopping before component i.

    Args:
        interval (Iterable[int]): The interval of components to consider
        p_list (Iterable[float]): Probability of each component passing

    Returns:
        Iterable[float]: Iterable where index $i$ is the probability of not stopping before component $i$.
    """

    table = build_probability_table(p_list)
    n = system.n
    probability_of_not_stopping = np.zeros(n + 1)
    for number_tested in range(n + 1):
        for number_passed in range(number_tested + 1):
            number_failed = number_tested - number_passed

            if system.determine_interval(number_passed, number_failed) == -1:
                probability_of_not_stopping[number_tested] += table[
                    number_passed, number_tested
                ]

    return probability_of_not_stopping
