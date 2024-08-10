"""Experiemnts are from the paper
A Polynomial-Time Approximation Scheme for
Sequential Batch Testing of Series Systems"""

import bisect
import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

module_logger = logging.getLogger(__name__)


@dataclass
class Component:
    probability: float
    cost: float


# Constant values for best approximation ratio
r_values = {
    "ssc": 10.742,
    "k-of-n": 3.236,
    "unit-cost-k-of-n": 2.303,
    "series": 1.414,
}


class System:
    def __init__(
        self,
        n: int,
        batch_cost: float | str,
        problem_type: int | None = None,
        k: int | None = None,
        interval_breakpoints: None | list[int] | NDArray = None,
        is_series: bool = False,
        unit_cost: bool = False,
        seed: int | None = None,
        probabilities: NDArray | None = None,
        costs: NDArray | None = None,
    ):
        """Initialize a system object

        Args:
            n (int): Number of components
            batch_cost (float | str): Cost of testing a batch
            problem_type (int | None, optional): Distribution of success probabilities. Defaults to None.
            k (int | None, optional): Threshold of number of components. Defaults to None.
            interval_breakpoints (None | list[int], optional): Breakpoints for SSC. Defaults to None.
            is_series (bool, optional): Use for series systems. Defaults to False.
            unit_cost (bool, optional): Unit cost k-of-n. Defaults to False.
            seed (int | None, optional): Random seed. Defaults to None.
            probabilities (NDArray | None, optional): Probabilities of components working. Defaults to None.
            costs (NDArray | None, optional): Cost of testing each component. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """

        # Input validation
        if not (k is None) ^ (interval_breakpoints is None) ^ is_series:
            raise ValueError("System must be one special type.")
        if k is not None and not 0 < k <= n:
            raise ValueError("k is not valid.")
        if interval_breakpoints is not None and not all(
            interval_breakpoints[i] <= interval_breakpoints[i + 1]
            for i in range(len(interval_breakpoints) - 1)
        ):
            raise ValueError("Interval breakpoints must be sorted.")
        if not ((problem_type is not None) ^ (probabilities is not None)):
            raise ValueError("Provide either problem type or custom probabilities.")
        if unit_cost and (costs is not None):
            raise ValueError("Cannot override unit cost with custom cost.")

        self.n = n
        if is_series:
            self.k = n
        else:
            self.k = k
        self.unit_cost = unit_cost
        self.is_series = is_series
        self.problem_type = problem_type

        # r gives the width of each interval, optimized for each problem type.
        if is_series:
            self.r = r_values["series"]
        elif not unit_cost and k is not None:
            self.r = r_values["k-of-n"]
        elif unit_cost and k is not None:
            self.r = r_values["unit-cost-k-of-n"]
        else:
            self.r = r_values["ssc"]

        if seed is not None:
            np.random.seed(seed)

        if probabilities is None and problem_type == 1:
            probabilities = np.random.uniform(0.5, 1, n)
        elif probabilities is None and problem_type == 2:
            probabilities = np.random.uniform(0.9, 1, n)
        elif probabilities is None and problem_type == 3:
            probabilities = np.random.uniform(0, 1, n)
        elif probabilities is not None and len(probabilities) != n:
            raise ValueError(
                "Probabilities must be the same length as the number of components."
            )

        assert probabilities is not None

        if batch_cost == "n":
            batch_cost = n
        elif batch_cost == "n/2":
            batch_cost = n / 2
        elif batch_cost == "n/4":
            batch_cost = n / 4
        else:
            batch_cost = batch_cost
        self.batch_cost = batch_cost

        # Testing cost is uniform [1, 10]
        if costs is None and not unit_cost:
            costs = np.random.uniform(1, 10, n)
        elif costs is None and unit_cost:
            costs = np.ones(n)
        elif costs is not None and len(costs) != n:
            raise ValueError(
                "Costs must be the same length as the number of components."
            )
        assert costs is not None

        components = list(zip(probabilities, costs))
        self.components = tuple(Component(*x) for x in components)
        if self.k is not None:
            self.interval_breakpoints = [self.k]
        else:
            self.interval_breakpoints = interval_breakpoints
        assert isinstance(self.batch_cost, (float, int))
        self.gamma = self.batch_cost * self.r

        module_logger.info(f"System created: {self}")

    def __repr__(self):
        return (
            f"System(n={self.n}, batch_cost={self.batch_cost}, "
            f"k={self.k}, unit_cost={self.unit_cost}, is_series={self.is_series}, "
            f"r={self.r}, gamma={self.gamma})"
        )

    def __str__(self):
        probabilities = ", ".join(f"{x.probability:.3f}" for x in self.components)
        costs = ", ".join(f"{x.cost:.3f}" for x in self.components)
        pattern = "\n\t".join(
            (
                "System:",
                f"n = {self.n}",
                f"batch_cost = {self.batch_cost}",
                f"k = {self.k}",
                f"unit_cost = {self.unit_cost}",
                f"is_series = {self.is_series}",
                f"r = {self.r}",
                f"gamma = {self.gamma}",
                f"probabilities = [{probabilities}]",
                f"costs         = [{costs}]",
                f"interval_breakpoints = {self.interval_breakpoints}",
            )
        )
        return pattern

    def determine_interval(
        self, number_passed: int, number_failed=None, overwritten_interval=None
    ) -> int:
        """
        Determine the interval in which the number of passed components lies.

        Args:
            number_passed (int): Number of components that passed the test.
            number_failed (int, optional): Number of components that failed the test. Defaults to None.
            overwritten_interval (int, optional): Overwritten interval. Defaults to None.

        Returns:
            int: The interval in which the number of passed components lies.
                Returns -1 if interval cannot be determined
        """
        if number_failed is not None and number_passed + number_failed > self.n:
            raise ValueError("Number of passed and failed components exceeds n.")
        if overwritten_interval is not None:
            interval = overwritten_interval
        else:
            interval = self.interval_breakpoints

        assert interval is not None

        interval_number = bisect.bisect(interval, number_passed)
        # Determine whether stopping condition is reached.
        if interval_number < len(interval) and number_failed is not None:
            number_remaining = self.n - number_passed - number_failed
            upper_bound = number_passed + number_remaining
            interval_boundary = interval[interval_number]
            if upper_bound >= interval_boundary:
                return -1
        return interval_number


if __name__ == "__main__":
    print(System(10, "n", problem_type=1, unit_cost=False, is_series=True))
