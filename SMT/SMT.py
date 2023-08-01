import math

from z3 import *
import itertools
from time import time
from .converter import get_file

MAXITER = 50


def array_max(vs):
    m = vs[0]
    for v in vs[1:]:
        m = If(v > m, v, m)
    return m


def lex_less(a, b):
    if not a:
        return True
    if not b:
        return False
    return Or(a[0] <= b[0], And(a[0] == b[0], lex_less(a[1:], b[1:])))


def convert_to_integer(y, time_range, courier_range, package_range):
    res = [[0 for _ in time_range] for _ in courier_range]

    for courier in courier_range:
        for _time in time_range:
            res[courier][_time] = Sum([package * y[package][_time][courier] for package in package_range])
    return res


def lex_less_single(a, b):
    if not a or not b:
        return True

    return Or(a[0] <= b[0], And(a[0] == b[0], lex_less_single(a[1:], b[1:])))


def lex_less_no_conversion(a, b):
    if not a:
        return True
    if not b:
        return False
    return Or(lex_less_single(a[0], b[0]), And(a[0] == b[0], lex_less_no_conversion(a[1:], b[1:])))


def multiple_couriers(m, n, D, l, s, to_ret1, to_ret2, to_ret3, to_ret4):
    solver = Solver()
    upper_bound, lower_bound = calculate_bound_package(m, n, D, l, s)

    # So that the package representing the base doens't count in the weight calculation
    s += [0]

    # Ranges
    package_range = range(n + 1)
    time_range = range(n + 2)
    time_range_no_zero = range(1, time_range[-1] + 1)
    courier_range = range(m)

    # Constant
    base_package = n
    last_time = n + 1

    # Variables
    # y[p][t][c] == 1 se c porta p al tempo t
    y = [[[Int(f"y_{package}_{_time}_{courier}")
           for courier in courier_range]
          for _time in time_range]
         for package in package_range]

    # Binary constraint
    # Value range of decision variable
    for package in package_range:
        for courier in courier_range:
            for _time in time_range:
                solver.add(y[package][_time][courier] <= 1)
                solver.add(y[package][_time][courier] >= 0)

    # weights vector, weights[c] is the amount transported by c
    weights = []
    for courier in courier_range:
        weights.append(
            Sum([s[package] * y[package][_time][courier] for _time in time_range for package in package_range])
        )

    # distances vector, distances[c] is the amount travelled by c
    distances = []
    for courier in courier_range:
        courier_distances = []
        for _time in time_range_no_zero:
            for p1 in package_range:
                for p2 in package_range:
                    a = y[p1][_time - 1][courier] == 1
                    b = y[p2][_time][courier] == 1
                    courier_distances.append(D[p1][p2] * And(a, b))

        distances.append(Sum(courier_distances))

    # Each package taken one time only
    for package in package_range:
        if package == base_package:
            continue
        solver.add(Sum([y[package][_time][courier] for _time in time_range for courier in courier_range]) == 1)

    # A courier carry only one package at _time
    for courier in courier_range:
        for _time in time_range:
            solver.add(Sum([y[package][_time][courier] for package in package_range]) == 1)

    # Every carried package must be delivered to destination and every courier must start from destination
    for courier in courier_range:
        solver.add(y[base_package][0][courier] == 1)
        solver.add(y[base_package][last_time][courier] == 1)

    # Each courier can hold packages with total weight <= max carriable weight
    for courier in courier_range:
        solver.add(weights[courier] <= l[courier])

    # Couriers must immediately start with a package after the base
    for courier in courier_range:
        for _time in time_range_no_zero:
            a = y[base_package][_time][courier]
            b = y[base_package][1][courier]

            solver.add(Implies(a != 1, b != 1))

    # Couriers cannot go back to the base before taking other packages
    for courier in courier_range:
        for _time in time_range_no_zero:
            a = y[base_package][_time][courier]

            for _time2 in range(_time + 1, last_time):
                b = y[base_package][_time2][courier]
                solver.add(Implies(a == 1, b == 1))

    # Symmetry breaking constraints
    # integer_matrix = convert_to_integer(y, time_range, courier_range, package_range)

    # If two couriers have the same capacity then they are symmetric, we impose an order between them
    # for c1 in courier_range:
    #     for c2 in courier_range:
    #         if c1 < c2 and l[c1] == l[c2]:
    #             solver.add(lex_less_no_conversion(integer_matrix[c1], integer_matrix[c2]))
    #
    # # Two couriers path are exchangeable if the maximum weight of the two is less than the minimum loading capacity
    # for c1 in courier_range:
    #     for c2 in courier_range:
    #         if c1 < c2:
    #             max_weight = If(weights[c1] > weights[c2], weights[c1], weights[c2])
    #             min_capacity = If(l[c1] < l[c2], l[c1], l[c2])
    #             condition = max_weight <= min_capacity
    #
    #             solver.add(Implies(condition, lex_less_no_conversion(y[c1], integer_matrix[c2])))

    # Heuristic (?)
    for courier in courier_range:
        package_transported = Sum([
            y[package][_time][courier] for _time in time_range for package in package_range if package != base_package
        ])

        solver.add(package_transported <= upper_bound)
        solver.add(package_transported >= lower_bound)

    # Getting maximum distance
    objective_value = array_max(distances)

    min_distance = math.inf
    for i in range(len(D)):
        for j in range(len(D[i])):
            if D[i][j] <= min_distance and D[i][j] != 0:
                min_distance = D[i][j]

    max_distance = 0
    for i in range(len(D)):
        max_distance += max(D[i])

    print(f"{max_distance=}, {min_distance=}")
    max_distance = math.ceil(max_distance)  # / upper_bound)
    min_distance = max(min_distance, math.floor(min_distance))  # * lower_bound))
    print(f"{max_distance=}, {min_distance=}")

    for a in solver.assertions():
        # print(a)
        pass

    print(f"constraint number: {len(solver.assertions())}")

    start_time = time()
    iter = 1
    last_sol = None
    while iter < MAXITER:

        weight = 0.5  # + (0.2 * math.exp(-0.2 * iter))

        k = int(((1 - weight) * min_distance + weight * max_distance))

        print(f"current k={k}, {weight=}")

        solver.push()
        solver.add(objective_value <= k)

        sol = solver.check()

        if sol != sat:
            min_distance = k
        else:
            max_distance = k
            last_sol = solver.model()

        to_ret = [[0 for _ in range(last_time + 1)] for _ in range(len(courier_range))]

        print(f"ITERATION: {iter} - TIME: {time() - start_time} - STATUS: {sol} - DISTANCE: {k}")

        if sol == sat:
            g = last_sol
            print("SMT SOLUTION: \n__________________\n")
            for courier in courier_range:
                t = ""
                for _time in time_range:
                    value = sum(package * g.eval(y[package][_time][courier]) for package in package_range)
                    t += f"{g.eval(value + 1)}, "
                    to_ret[courier][ _time] = g.eval(value + 1).as_long()
                print(t)
            print("\n__________________\n")
            if to_ret1 != None:
                to_ret1.put(k)
                to_ret2.put(to_ret)
                to_ret3.put(f"{time() - start_time:.2f}")
                to_ret4.put(iter)

        if abs(min_distance - max_distance) <= 1:
            g = last_sol
            print("SMT SOLUTION: \n__________________\n")
            for courier in courier_range:
                t = ""
                for _time in time_range:
                    value = sum(package * g.eval(y[package][_time][courier]) for package in package_range)
                    t += f"{g.eval(value + 1)}, "
                    to_ret[courier][ _time] = g.eval(value + 1).as_long()   
                print(t)             
            print("\n__________________\n")

            return max_distance, to_ret, f"{time() - start_time:.2f}", iter

        iter += 1
        solver.pop()

    return max_distance, "Out of _time", f"{time() - start_time:.2f}", iter


def calculate_bound_package(m, n, D, l, s):
    weight_sum = sum(s)
    capacity_min = min(l)

    capacity_max = max(l)
    weight_max = max(s)

    max_package = capacity_max // weight_max
    min_package = capacity_min // weight_max

    upper_bound = min(n, math.ceil(n / m))
    lower_bound = max(0, math.floor(n / m))

    if capacity_min >= weight_sum:
        delta = 0
    else:
        delta = max_package - min_package

    upper_bound = min(n, upper_bound + delta)
    lower_bound = max(0, lower_bound - delta)
    print(f"{delta=}, {upper_bound=}, {lower_bound=}")

    return upper_bound, lower_bound


def solve_one(instances, idx, to_ret1=None, to_ret2=None, to_ret3=None, to_ret4=None):

    m, n, D, l, s = instances[idx]['m'], instances[idx]['n'], instances[idx]['D'], instances[idx]['l'], instances[idx][
        's']

    mindist, sol, time_passed, iter = multiple_couriers(m, n, D, l, s, to_ret1, to_ret2, to_ret3, to_ret4)  # *instances[idx].values()) 27

    if to_ret1 != None:
        to_ret1.put(mindist)
        to_ret2.put(sol)
        to_ret3.put(time_passed)
        to_ret4.put(iter)
    return sol, mindist, time_passed, iter


def main():
    instances = get_file()
    _, mindist, t, _ = solve_one(instances, 5)
    print(f"Min distance {mindist}")
    print(f"Time passed {t}s")


if __name__ == "__main__":
    main()
