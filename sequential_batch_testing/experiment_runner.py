import json
import pathlib
import random
from copy import deepcopy
from time import time_ns

import numpy as np
from adaptive_batching import table_dp
from batching import derandomized_batching, randomized_batching
from initialization import System
from policy_evaluations import evaluate_k_of_n_systems, evaluate_unbatched_policy
from unbatched_policy_computations import compute_unbatched_policy


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def run_experiments():
    np.random.seed(0)
    time_began = time_ns()

    pathlib.Path(rf"./results/{time_began}").mkdir(parents=True, exist_ok=True)
    counter = 0
    for rep_no in range(10):
        for system_config in generate_systems():
            results = {}
            system = System(**system_config)
            results["data"] = get_data(system)
            results["rep_no"] = rep_no
            unbatched_begin = time_ns()
            unbatched_policy = compute_unbatched_policy(system)
            unbatched_end = time_ns()

            results["unbatched_time"] = unbatched_end - unbatched_begin

            results["unbatched_policy"] = unbatched_policy
            results["unbatched_cost"] = evaluate_unbatched_policy(
                system, unbatched_policy
            )

            randomized_begin = time_ns()
            results["randomized_batched_policy"] = randomized_batching(
                system, unbatched_policy
            )
            randomized_end = time_ns()
            results["randomized_time"] = randomized_end - randomized_begin

            results["randomized_batched_cost"] = evaluate_k_of_n_systems(
                system, results["randomized_batched_policy"]
            )

            derandomized_begin = time_ns()
            results["derandomized_batched_policy"] = derandomized_batching(
                system, unbatched_policy
            )
            derandomized_end = time_ns()
            results["derandomized_batch_time"] = derandomized_end - derandomized_begin

            results["derandomized_batched_cost"] = evaluate_k_of_n_systems(
                system,
                results["derandomized_batched_policy"],
            )
            results["number_of_batches"] = len(results["derandomized_batched_policy"])

            optimal_begin = time_ns()

            if system.is_series:
                adap_interval = [system.n]
            elif system.k is not None:
                adap_interval = [system.k]
            else:
                adap_interval = system.interval_breakpoints
            adap_costs = np.array([x.cost for x in system.components])
            adap_probs = np.array([x.probability for x in system.components])
            results["optimal_batched_cost"] = table_dp(
                adap_costs, adap_probs, system.batch_cost, adap_interval
            )

            optimal_end = time_ns()
            results["optimal_time"] = optimal_end - optimal_begin

            print(results)
            with open(rf"./results/{time_began}/results{counter}.json", "w") as f:
                json.dump(results, f, cls=NpEncoder)
            counter += 1


def generate_systems():
    for n in range(5, 16):
        for batch_cost in ["n", "n/2", "n/4"]:
            for problem_type in [1, 2]:
                yield {
                    "n": n,
                    "batch_cost": batch_cost,
                    "problem_type": problem_type,
                    "is_series": True,
                }
            for k in [
                int(np.floor(n / 4)),
                int(np.floor(n / 2)),
                int(np.ceil(3 * n / 4)),
            ]:
                yield {
                    "n": n,
                    "batch_cost": batch_cost,
                    "problem_type": 3,
                    "k": k,
                }
            for number_of_threshold in range(2, min(n, 5)):
                interval = random.sample(range(1, n), number_of_threshold)
                interval.sort()
                yield {
                    "n": n,
                    "batch_cost": batch_cost,
                    "problem_type": 3,
                    "interval_breakpoints": interval,
                }


def get_data(system):
    data = deepcopy(vars(system))
    data["costs"] = [x.cost for x in system.components]
    data["probabilities"] = [x.probability for x in system.components]
    data.pop("components")
    return data


if __name__ == "__main__":
    res = run_experiments()
    print(res)
