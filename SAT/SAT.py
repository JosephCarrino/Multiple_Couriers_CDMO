from z3 import *
import numpy as np
from itertools import combinations
from ..utils.converter import get_file
from time import time


def at_least_one(bool_vars):
    return Or(bool_vars)


def at_most_one(bool_vars):
    return [Not(And(pair[0], pair[1])) for pair in combinations(bool_vars, 2)]


def exactly_one(bool_vars):
    return at_most_one(bool_vars) + [at_least_one(bool_vars)]


def at_most_k_seq(x, k, name):
    clauses = []
    n = len(x)
    s = [[Bool(f"s_{i}_{j}_{name}") for j in range(k)] for i in range(n - 1)]
    clauses.append(Or(Not(x[0]), s[0][0]))
    clauses += [Not(s[0][j]) for j in range(1, k)]
    clauses.append(Or(Not(x[n - 1]), Not(s[n - 2][k - 1])))
    for i in range(1, n - 1):
        clauses.append(Or(Not(x[i]), s[i][0]))
        clauses.append(Or(Not(s[i - 1][0]), s[i][0]))
        for j in range(1, k):
            clauses.append(Or(Not(s[i - 1][j]), s[i][j]))
            clauses.append(Or(Not(x[i]), Not(s[i - 1][j - 1]), s[i][j]))
        clauses.append(Or(Not(x[i]), Not(s[i - 1][k - 1])))
    return And(clauses)


MAXITER = 500


def multiple_couriers(
    m, n, D, l, s
):  # m is the number of couriers, n is the number of packages, D is the distance matrix, l is the load of each courier, s is the size of each package
    # i is the courier
    # j is the step
    # k is the one-hot encoding of the package
    c = [
        [[Bool(f"c_{i}_{j}_{k}") for k in range(n + 1)] for j in range(n + 2)]
        for i in range(m)
    ]

    # i is the courier
    # j is the package
    # k is the bitwise encoding of the package
    w = [
        [[Bool(f"w_{i}_{j}_{k}") for k in range(s[j])] for j in range(n)]
        for i in range(m)
    ]

    # i is the courier
    # start is a position of the route
    # end is the next position of the route
    d = [
        [
            [Bool(f"d_{i}_{start}_{end}") for end in range(n + 1)]
            for start in range(n + 1)
        ]
        for i in range(m)
    ]

    # Defining the solver instance
    solver = Solver()

    # Binding the two matrixes
    for i in range(m):
        for j in range(1, n + 1):
            for package in range(n):
                for k in range(s[package]):
                    solver.add(Implies(c[i][j][package], w[i][package][k]))
                    # solver.add(Implies(w[i][package][k], c[i][j][package]))

    # Binding the routing
    for i in range(m):
        for j in range(n):
            for start in range(n + 1):
                for end in range(n + 1):
                    solver.add(
                        Implies(And(c[i][j][start], c[i][j + 1][end]), d[i][start][end])
                    )

    # At each time, the courier carries exactly one package or he is at base
    for i in range(m):
        for j in range(n + 2):
            solver.add(exactly_one(c[i][j][:]))

    # Each package is carried only once
    for package in range(n):
        vars = []
        for i in range(m):
            for j in range(1, n + 1):
                vars.append(c[i][j][package])
        solver.add(exactly_one(vars))

    # The total weight carried by each courier must be less or equal than his
    # carriable weight
    for i in range(m):
        vars = []
        for j in range(n):
            for k in range(s[j]):
                vars.append(w[i][j][k])
        solver.add(at_most_k_seq(vars, l[i], f"weight_{i}"))

    # At start/end the courier must be at the base
    for i in range(m):
        solver.add(c[i][0][n])
        solver.add(c[i][n + 1][n])

    ## Breaking Symmetry ##

    # Lexicographic order for each courier
    for i in range(m):
        for j in range(1, n):
            for z in range(j + 1, n + 1):
                solver.add(Implies(c[i][j][n], c[i][z][n]))

    min_dist = 9999999
    for i in range(len(D)):
        for j in range(len(D[i])):
            if D[i][j] <= min_dist and D[i][j] != 0:
                min_dist = D[i][j]

    max_dist = 0
    for i in range(len(D)):
        max_dist += max(D[i])

    for i in range(m):
        dists = []
        for start in range(n + 1):
            for end in range(n + 1):
                for z in range(D[start][end]):
                    dists.append(d[i][start][end])

    iter = 1
    start_time = time()
    last_sol = None
    g = None
    while iter < MAXITER:
        k = int((min_dist + max_dist) / 2)
        # Getting the maximum distance
        solver.push()
        solver.add(at_most_k_seq(dists, k, f"Courier_dist_{i}"))

        sol = solver.check()
        # print(g)
        if sol != sat:
            min_dist = k
        else:
            max_dist = k
            last_sol = sol
            g = solver.model()
        print(k)
        print(sol)

        if abs(min_dist - max_dist) <= 1:
            if last_sol:
                if g != None:
                    paths = [[]for i in range(m)]
                    for i in range(m):
                        for j in range(n + 1):
                            for z in range(n + 1):
                                if is_true(g[c[i][j][z]]) and z != n:
                                    paths[i].append(z+1)
                return k, paths, f"{time() - start_time:.2f}", iter
            else:
                return 0, "Unsat", f"{time() - start_time:.2f}", iter
        iter += 1
        solver.pop()
    if g != None:
        paths = [[]for i in range(m)]
        for i in range(m):
            for j in range(n + 1):
                for k in range(n + 1):
                    if is_true(g[d[i][j][k]]):
                        paths[i].append(k)
    return k, last_sol, f"{time() - start_time:.2f}", iter


def minimizer_binary(instance, solver=multiple_couriers, maxiter=MAXITER):
    m = instance["m"]
    n = instance["n"]
    D = instance["D"]
    l = instance["l"]
    s = instance["s"]
    print(instance)
    return solver(m, n, D, l, s)


def solve_one(instances, idx, to_ret1=None, to_ret2=None, to_ret3=None, to_ret4=None):
    sol, mindist, time_passed, iter = minimizer_binary(instances[idx])
    if to_ret1 != None:
        to_ret1.put(sol)
        to_ret2.put(mindist)
        to_ret3.put(time_passed)
        to_ret4.put(iter)
    return sol, mindist, time_passed, iter


def main():
    instances = get_file()
    solve_one(instances, 0)


if __name__ == "__main__":
    main()
