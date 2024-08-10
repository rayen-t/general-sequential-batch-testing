import os
import sys

import pytest

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../sequential_batch_testing"
)

from initialization import System


class TestBatchSimulator:
    def test_initialization(self):
        """Checks if the random variables are in reasonable ranges."""
        n = 1000
        system1 = System(n, "n", 1, k=n)
        assert system1.n == n
        assert system1.k == n
        assert system1.batch_cost == n
        assert all(
            [1 <= component.cost <= 10 for component in system1.components]
        ), "Cost is not in the legal range of [1, 10000]"
        assert all(
            [0.5 <= component.probability <= 1 for component in system1.components],
        ), "Probability is not in the legal range of [0.5, 1]"

        system2 = System(n, "n/2", 2, k=n)
        assert all(
            [0.9 <= component.probability <= 1 for component in system2.components],
        ), "Probability is not in the legal range of [0.9, 1]"
        assert system2.batch_cost == n / 2

    def test_invalid_input(self):
        n = 100
        System(n, "n", 1, interval_breakpoints=[1, 2])
        System(n, "n", 1, k=1)
        with pytest.raises(ValueError):
            System(n, "n", 1, k=0, interval_breakpoints=[0])
        with pytest.raises(ValueError):
            System(n, "n", 1, k=101)


class TestDetermineInterval:
    def test_series(self):
        n = 100
        system = System(n, "n", 1, k=n)
        assert system.determine_interval(0) == 0
        assert system.determine_interval(1) == 0
        assert system.determine_interval(99) == 0
        assert system.determine_interval(100) == 1
        assert system.determine_interval(101) == 1
        assert system.determine_interval(99, 1) == 0
        assert system.determine_interval(100, 0) == 1
        assert system.determine_interval(50, 0) == -1
        assert system.determine_interval(50, 1) == 0
        with pytest.raises(ValueError):
            system.determine_interval(50, 51)

    def test_parallel(self):
        n = 100
        system = System(n, "n", 1, k=1)
        assert system.determine_interval(0) == 0
        assert system.determine_interval(1) == 1
        assert system.determine_interval(99) == 1
        assert system.determine_interval(0, 99) == -1
        assert system.determine_interval(1, 99) == 1
        assert system.determine_interval(99, 1) == 1
        assert system.determine_interval(0, 100) == 0
        with pytest.raises(ValueError):
            system.determine_interval(50, 51)

    def test_k_of_n(self):
        n = 17
        system = System(n, "n", 1, k=11)
        assert system.determine_interval(0) == 0
        assert system.determine_interval(1) == 0
        assert system.determine_interval(10) == 0
        assert system.determine_interval(11) == 1
        assert system.determine_interval(17) == 1

        with pytest.raises(ValueError):
            system.determine_interval(18, 9)

        assert system.determine_interval(0, 6) == -1
        assert system.determine_interval(0, 7) == 0
        assert system.determine_interval(10, 0) == -1
        assert system.determine_interval(11, 0) == 1
        assert system.determine_interval(3, 7) == 0
        assert system.determine_interval(3, 6) == -1

    def test_single_case(self):
        n = 5
        system = System(n, "n", 1, interval_breakpoints=[1, 2, 3, 4, 5])
        assert system.determine_interval(0) == 0
        assert system.determine_interval(1) == 1
        assert system.determine_interval(2) == 2
        assert system.determine_interval(3) == 3
        assert system.determine_interval(4) == 4
        assert system.determine_interval(5) == 5

        for i in range(6):
            for j in range(6):
                if i + j > 5:
                    with pytest.raises(ValueError):
                        system.determine_interval(i, j)
                elif i + j == 5:
                    assert system.determine_interval(i, j) == i
                else:
                    assert system.determine_interval(i, j) == -1

    def test_normal_interval(self):
        n = 10
        system = System(n, "n", 1, interval_breakpoints=[2, 4, 6, 8, 10])

        assert system.determine_interval(0) == 0
        assert system.determine_interval(1) == 0
        assert system.determine_interval(2) == 1
        assert system.determine_interval(3) == 1
        assert system.determine_interval(4) == 2
        assert system.determine_interval(5) == 2
        assert system.determine_interval(6) == 3
        assert system.determine_interval(7) == 3
        assert system.determine_interval(8) == 4
        assert system.determine_interval(9) == 4
        assert system.determine_interval(10) == 5
        assert system.determine_interval(2, 7) == 1
        assert system.determine_interval(2, 8) == 1
        assert system.determine_interval(2, 6) == -1
        assert system.determine_interval(3, 7) == 1
        assert system.determine_interval(3, 6) == -1
        assert system.determine_interval(3, 5) == -1
