import random


def generate_cases():
    for n in range(3, 7):
        for batch_cost in ("n", "n/2", "n/4", 0, 1000):
            for problem_type in (1, 2):
                for unit_cost in (True, False):
                    for k in range(1, n + 1):
                        yield {
                            "n": n,
                            "batch_cost": batch_cost,
                            "problem_type": problem_type,
                            "k": k,
                            "unit_cost": unit_cost,
                        }
                for number_of_threshold in range(1, n - 1):
                    interval = random.sample(range(1, n), number_of_threshold)
                    interval.sort()
                    yield {
                        "n": n,
                        "batch_cost": batch_cost,
                        "problem_type": problem_type,
                        "interval_breakpoints": interval,
                    }
                yield {
                    "n": n,
                    "batch_cost": batch_cost,
                    "problem_type": problem_type,
                    "is_series": True,
                }


def generate_cases_fixed_batch():
    for n in range(3, 7):
        for k in range(1, n + 1):
            for problem_type in (1, 2):
                for unit_cost in (True, False):
                    yield {
                        "n": n,
                        "batch_cost": 0,
                        "problem_type": problem_type,
                        "k": k,
                        "unit_cost": unit_cost,
                    }


def generate_cases_unit_cost():
    for n in range(3, 7):
        for k in range(1, n + 1):
            for problem_type in (1, 2):
                yield {
                    "n": n,
                    "batch_cost": 0,
                    "problem_type": problem_type,
                    "k": k,
                    "unit_cost": True,
                }
