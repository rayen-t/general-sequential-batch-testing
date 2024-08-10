import numpy as np
from batching import derandomized_batching, randomized_batching
from initialization import System
from optimal_policy import compute_optimal_policy
from policy_evaluations import evaluate_k_of_n_systems
from unbatched_policy_computations import compute_unbatched_policy

costs = [
    9.0284409223401,
    1.5010981769487874,
    8.426153083694821,
    3.5818455476119304,
    5.169343028736045,
    1.5768912696826813,
    4.664186412477196,
    3.3755229080627513,
    9.07766289615652,
]
probabilities = [
    0.8314743176648678,
    0.6010259712649915,
    0.6627678961632093,
    0.9310761891727182,
    0.7825590113879715,
    0.9653262551796549,
    0.5090164383680122,
    0.9937903016767997,
    0.8596688206630175,
]
system = System(9, "n", k=1, unit_cost=False, costs=costs, probabilities=probabilities)
print(system)
unbatched_policy = compute_unbatched_policy(system)
print(unbatched_policy)
randomized_policy = randomized_batching(system, unbatched_policy)
randomized_cost = evaluate_k_of_n_systems(system, randomized_policy)

derandomized_policy = derandomized_batching(system, unbatched_policy)
print(randomized_policy)


def evaluate_parallel_system(system: System, policy) -> float:
    """Evaluate the cost of a policy for the parallel system

    Args:
        system (System): The parameters of the system
        policy (Sequence): The policy to evaluate

    Returns:
        float: The cost of the policy
    """

    def recurs(idx):
        if idx == len(policy):
            return 0
        set_items = [system.components[i] for i in policy[idx]]
        return (
            system.batch_cost
            + sum([x.cost for x in set_items])
            + np.prod([1 - x.probability for x in set_items]) * recurs(idx + 1)
        )

    return recurs(0)


unbatched_policy_order = [system.components[i] for i in unbatched_policy]
cum_cost = np.cumsum([x.cost for x in unbatched_policy_order])
print(cum_cost)
xi = np.random.rand()
print(system.gamma)
print("batching = ", [2 + np.floor(cost / system.gamma - xi) for cost in cum_cost])
for b in range(5):
    print(((b) * xi * system.gamma, (b + 1) * system.gamma * xi))
print(randomized_policy)
print(randomized_cost)
print(evaluate_parallel_system(system, derandomized_policy))
print(derandomized_cost := evaluate_k_of_n_systems(system, derandomized_policy))
print(optimal_policy := compute_optimal_policy(system))
