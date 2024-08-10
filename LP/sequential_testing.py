import bisect

import gurobipy as gp
import numpy as np
from gurobipy import GRB
import time
import pandas as pd

eps = 10e-5


# q_t = sum(pt_t, pt_t+1, ..., pt_T)
def compute_qt(pt):
    q_t = np.cumsum(pt[::-1])[::-1]
    return q_t


def backtrack_dp(memo, T):
    batch = []
    t = 0
    while t < T:
        _, t = memo[t]
        batch.append(t)
    return batch


def backtrack(parents, T):
    batch = []
    t = 0
    while t < T:
        t = parents[t]
        batch.append(t)
    return batch


# Min cost batching problem
def optimal_batching_dp(pt, B):
    T = len(pt)
    qt = compute_qt(pt)
    # print(qt)
    memo = {}

    def recurs(t):
        if t == T:
            return 0
        elif t not in memo:
            min_index = None
            min_cost = float("inf")
            for z in range(t + 1, T + 1):
                cost = recurs(z) + (B + z - t) * qt[t]
                if cost < min_cost:
                    min_cost = cost
                    min_index = z
            memo[t] = min_cost, min_index
        return memo[t][0]

    recurs(0)
    print(memo)
    solution = backtrack(memo, T)

    assert solution[-1] == T
    return memo[0][0], solution


def optimal_batching(pt, B):
    T = len(pt)
    qt = compute_qt(pt)
    memo = np.empty((T + 1))
    memo.fill(np.NaN)
    parents = np.empty((T + 1), int)
    parents.fill(-1)

    memo[T] = 0
    for t in range(T - 1, -1, -1):
        min_index = None
        min_cost = float("inf")
        for z in range(t + 1, T + 1):
            cost = memo[z] + (B + z - t) * qt[t]
            if cost < min_cost:
                min_cost = cost
                min_index = z
        memo[t] = min_cost
        parents[t] = min_index
    solution = backtrack(parents, T)
    assert solution[-1] == T
    return memo[0], solution


def get_cost(batched_solution, t, T, B):
    if t >= T:
        # return len(batched_solution) * B + T
        raise ValueError("t must be less than T")
    batch_index = bisect.bisect(batched_solution, t)
    testing_cost = batched_solution[batch_index]
    total_cost = testing_cost + (batch_index + 1) * B
    return total_cost


def column_generation(B, T):
    if T == 0:
        raise ValueError("T must be greater than 0")
    # master LP
    master = gp.Model("Apx ratio")
    master.setParam("OutputFlag", 0)  # turn off output reporting

    y = {}
    constraints = {}
    C = np.empty((T, 1))
    sigma_index = 0

    # Initialize variables
    mu = master.addVar(obj=1, vtype=GRB.CONTINUOUS, name="mu")
    y[sigma_index] = master.addVar(vtype=GRB.CONTINUOUS, name=f"y[{sigma_index}]")
    sigma_index += 1

    # Initialize constraints
    y_prob_dist_constr = master.addConstr(y[0] == 1)
    for t in range(T):
        C[t, 0] = B + T
        constraints[t] = master.addConstr(-C[t, 0] * y[0] + (B + t + 1) * mu >= 0)
    # print(C)
    master.update()

    while True:
        master.optimize()
        # Get optimal dual variables
        lambda_ = y_prob_dist_constr.Pi
        p_t = np.array([constraints[t].Pi for t in range(T)])

        dp_val, subproblem_solution = optimal_batching(p_t, B)
        col = np.array([[get_cost(subproblem_solution, t, T, B) for t in range(T)]])
        # print(col)
        if lambda_ <= dp_val + eps:
            break
        C = np.append(C, col.T, axis=1)
        grb_column = gp.Column()
        for t in range(T):
            grb_column.addTerms(-col[0, t], constraints[t])
        grb_column.addTerms(1, y_prob_dist_constr)
        y[sigma_index] = master.addVar(
            vtype=GRB.CONTINUOUS, name=f"y[{sigma_index}]", column=grb_column
        )

        master.update()
        sigma_index += 1

    return (
        master.objVal,
        C.shape,
        p_t,
    )


if __name__ == "__main__":
    T = 500
    res = {"B": [], "obj": [], "C_shape": [], "time": [], "p_t": []}
    for i in range(1, 51):
        B = i * 10
        print("testing B = ", B)
        start = time.time_ns()
        obj, c_shape, p_t = column_generation(B, T)
        end = time.time_ns()
        res["B"].append(B)
        res["obj"].append(obj)
        res["C_shape"].append(c_shape)
        res["time"].append((end - start) / 10**6)
        res["p_t"].append(p_t)
        print(f"{res=}")
        res_df = pd.DataFrame(res)
        res_df.to_csv("column_generation_results.csv")
    print(f"{max([values for values in res['obj']])}")
