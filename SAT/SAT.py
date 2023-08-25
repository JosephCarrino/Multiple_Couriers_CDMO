import itertools

from z3 import *
from itertools import combinations
from utils.converter import get_file
from time import time


def at_least_one(variables: list[BoolRef]):
    """
    Return constraint that at least one of the variables in variables is true
    :param variables: List of variables
    :return:
    """

    return Or(variables)


def at_most_one(variables: list[BoolRef]):
    """
    Return constraint that at most one of the variables in variables is true
    :param variables: List of variables
    :return:
    """

    return [Not(And(pair[0], pair[1])) for pair in combinations(variables, 2)]


def exactly_one(variables: list[BoolRef]):
    """
    Return constraint that exactly one of the variable in variables is true
    :param bool_vars: List of variables
    """

    return at_most_one(variables) + [at_least_one(variables)]


def at_most_k_correct(variables: list[BoolRef], k: int):
    """
    Return constraint that at most k of the variables in vars are true
    :param variables: List of variables
    :param k: Maximum number of variables that can be true
    :return:
    """

    return PbLe([(var, 1) for var in variables], k)


def less_than(a: BoolRef, b: BoolRef):
    """
    Return constraint that a < b
    """

    return And(a, Not(b))


def equal(a: BoolRef, b: BoolRef):
    """
    Return constraint that a == b
    """

    return Or(And(a, b), And(Not(a), Not(b)))


def lex_less_single(a: list[BoolRef], b: list[BoolRef]) -> bool:
    """
    Return constraint that a < b in lexicographic order:
    a := a_1, ..., a_n
    b := b_1, ..., b_n
    a_1 < b_1 or (a_1 == b_1 and lex_less_single(a_2...a_n, b_2...b_n))
    :param a: list of bools
    :param b: list of bools
    :return:
    """
    if not a or not b:
        return True

    return Or(less_than(a[0], b[0]), And(equal(a[0], b[0]), lex_less_single(a[1:], b[1:])))


def lex_less(a: list[list[BoolRef]], b: list[list[BoolRef]]) -> bool:
    """
    Return constraint that a < b in lexicographic order:
    :param a: List of lists of bools where each sublist is the encoding of a number
    :param b: List of lists of bools where each sublist is the encoding of a number
    :return:
    """

    if not a:
        return True
    if not b:
        return False

    return Or(lex_less_single(a[0], b[0]), And(a[0] == b[0], lex_less(a[1:], b[1:])))


MAXITER = 500


def multiple_couriers(
        m: int, n: int, D: list[list[int]], l: list[int], s: list[int]
):
    """
    SAT solver
    :param m: Number of couriers
    :param n: Number of packages
    :param D: Matrix of packages distances
    :param l: List of carriable weight by each courier
    :param s: List of weights of the packages
    :return: Minimized distance, found solution, computation time and iterations number
    """

    # So that the package representing the base doesn't count in the weight calculation
    s += [0]

    ## Ranges ##
    package_range = range(n + 1)
    time_range = range(n + 2)
    time_range_no_zero = range(1, time_range[-1] + 1)
    courier_range = range(m)

    ## Constant ##
    base_package = n
    last_time = time_range[-1]
    variable_coordinates = list(itertools.product(package_range, time_range, courier_range))

    ## Variables ##
    # # y[courier][time][package] = True if the courier carries the package at time
    y = [[[Bool(f"y_{courier}_{_time}_{package}")
           for package in package_range]
          for _time in time_range]
         for courier in courier_range]

    # # weights[courier][package] = True if the courier carries the package
    weights = [[Bool(f"weights_{courier}_{package}")
                for package in package_range]
               for courier in courier_range]

    # # distance[courier][start][end] = True if the courier goes from start to end at some time in the route
    distances = [[[Bool(f"distances_{courier}_{start}_{end}")
                   # for d in range(D[start][end])]
                   for end in package_range]
                  for start in package_range]
                 for courier in courier_range]

    solver = Solver()

    ## Constraints ##

    # Binding the weights
    for courier in courier_range:
        for _time in time_range:
            for package in package_range:
                solver.add(Implies(y[courier][_time][package], weights[courier][package]))

    # Binding the distances
    for courier in courier_range:
        for _time in time_range_no_zero:
            for p1 in package_range:
                for p2 in package_range:
                    if p1 == p2:
                        continue
                    condition = And(y[courier][_time - 1][p1], y[courier][_time][p2])

                    solver.add(
                        Implies(
                            condition,
                            distances[courier][p1][p2]
                        )
                    )

    # At each time, the courier carries exactly one package or he is at base
    for courier in courier_range:
        for _time in time_range:
            solver.add(exactly_one(y[courier][_time][:]))

    # Each package is carried only once
    for package in package_range:
        if package != base_package:
            solver.add(exactly_one([y[courier][_time][package] for courier in courier_range for _time in time_range]))

    # The total weight carried by each courier must be less or equal than his
    # carriable weight
    for courier in courier_range:
        solver.add(at_most_k_correct(
            [weights[courier][package] for package in package_range for _ in range(s[package])],
            l[courier],
        ))

    # At start/end the courier must be at the base
    for courier in courier_range:
        solver.add(y[courier][0][base_package])
        solver.add(y[courier][last_time][base_package])

    ## Optimization ##

    # Couriers must immediately start with a package after the base if they carry a package
    for courier in courier_range:
        for _time in time_range_no_zero:
            a = y[courier][_time][base_package]
            b = y[courier][1][base_package]

            solver.add(Implies(Not(a), Not(b)))

    # Couriers cannot go back to the base before taking other packages
    for courier in courier_range:
        for _time in time_range_no_zero:
            a = y[courier][_time][base_package]

            for _time2 in range(_time + 1, last_time):
                b = y[courier][_time2][base_package]
                solver.add(Implies(a, b))

    ## Breaking Symmetry ##
    # Lexicographic order for each courier
    for c1 in courier_range:
        for c2 in courier_range:
            if c1 < c2 and l[c1] == l[c2]:
                solver.add(lex_less(y[c1], y[c2]))

    ## Objective function ##

    # Getting minimum and maximum distance
    min_distance = math.inf
    for i in range(len(D)):
        for j in range(len(D[i])):
            if D[i][j] <= min_distance and D[i][j] != 0:
                min_distance = D[i][j]

    max_distance = 0
    for i in range(len(D)):
        max_distance += max(D[i])

    start_time = time()
    iterations = 1
    last_sol = None
    while iterations < MAXITER:
        k = int((min_distance + max_distance) / 2)

        # Getting the maximum distance
        solver.push()

        for courier in courier_range:
            courier_dist = [
                distances[courier][p1][p2] for p1 in package_range for p2 in package_range for _ in range(D[p1][p2])
            ]

            solver.add(at_most_k_correct(courier_dist, k))

        sol = solver.check()

        # print(g)
        if sol != sat:
            min_distance = k
        else:
            max_distance = k
            last_sol = solver.model()

        print(f"ITERATION: {iterations} - TIME: {time() - start_time} - STATUS: {sol} - DISTANCE: {k}")

        if abs(min_distance - max_distance) <= 1 or sol == sat:
            g = last_sol

            print(f"SAT SOLUTION: \n____________________\n")
            solution_matrix = [[0 for _ in range(last_time + 1)] for _ in range(len(courier_range))]

            for package, _time, courier in variable_coordinates:
                if g[y[courier][_time][package]]:
                    solution_matrix[courier][_time] = package + 1

            for i in range(len(solution_matrix)):
                print(f"Courier {i}: {solution_matrix[i]}")

            if abs(min_distance - max_distance) <= 1:
                return max_distance, solution_matrix, f"{time() - start_time:.2f}", iterations

        iterations += 1
        solver.pop()

    return max_distance, "Out of _time", f"{time() - start_time:.2f}", iterations


def minimizer_binary(instance, solver=multiple_couriers, maxiter=MAXITER):
    m = instance["m"]
    n = instance["n"]
    D = instance["D"]
    l = instance["l"]
    s = instance["s"]
    print(instance)
    return solver(m, n, D, l, s)


def solve_one(instances, idx, to_ret1=None, to_ret2=None, to_ret3=None, to_ret4=None):
    mindist, sol, time_passed, iterations = minimizer_binary(instances[idx])

    if to_ret1 is not None:
        to_ret1.put(mindist)
        to_ret2.put(sol)
        to_ret3.put(time_passed)
        to_ret4.put(iterations)

    return sol, mindist, time_passed, iterations


def main():
    instances = get_file()
    _, mindist, t, _ = solve_one(instances, 5)

    print("\n\n\n")

    print(f"Min distance {mindist}")
    print(f"Time passed {t}s")


if __name__ == "__main__":
    main()
