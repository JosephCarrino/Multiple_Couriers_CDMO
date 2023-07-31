import math

from z3 import *
import itertools
from time import time
from converter import get_file
import numpy as np

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
    # print(f"{delta=}, {upper_bound=}, {lower_bound=}")

    return upper_bound, lower_bound

def multiple_couriers(m, n, D, l, s):
    # solver = Solver()
    solver = Then('simplify', 'elim-term-ite', 'solve-eqs', 'smt').solver()

    upper_bound, lower_bound = calculate_bound_package(m, n, D, l, s)

    # So that the package representing the base doens't count in the weight calculation
    s += [0]

    # Ranges
    package_range = range(n + 1)
    # time_range = range(n + 2)
    time_range = range(upper_bound + 2)
    time_range_no_zero = range(1, time_range[-1] + 1)
    courier_range = range(m)

    # Constant
    base_package = n
    last_time = time_range[-1]

    # Variables
    # y[p][t][c] == 1 se c porta p al tempo t
    y = [[[Int(f"y_{courier}_{package}_{_time}")
           for _time in time_range]
          for package in package_range]
         for courier in courier_range]

    # print(y)

    # Binary constraint
    # Value range of decision variable
    for package in package_range:
        for courier in courier_range:
            for _time in time_range:
                solver.add(y[courier][package][_time] <= 1)
                solver.add(y[courier][package][_time] >= 0)

    # weights vector, weights[c] is the amount transported by c
    weights = []
    for courier in courier_range:
        weights.append(
            Sum([s[package] * y[courier][package][_time] for _time in time_range for package in package_range])
        )

    # distances vector, distances[c] is the amount travelled by c
    distances = []
    for courier in courier_range:
        courier_distances = []
        for _time in time_range_no_zero:
            for p1 in package_range:
                for p2 in package_range:
                    a = y[courier][p1][_time - 1] == 1
                    b = y[courier][p2][_time] == 1
                    courier_distances.append(D[p1][p2] * And(a, b))

        distances.append(Sum(courier_distances))

    # Each package taken one time only
    for package in package_range:
        if package == base_package:
            continue
        solver.add(Sum([y[courier][package][_time] for _time in time_range for courier in courier_range]) == 1)

    # A courier carry only one package at _time
    for courier in courier_range:
        for _time in time_range:
            solver.add(Sum([y[courier][package][_time] for package in package_range]) == 1)

    # Every carried package must be delivered to destination and every courier must start from destination
    for courier in courier_range:
        solver.add(y[courier][base_package][0] == 1)
        solver.add(y[courier][base_package][last_time] == 1)

    # Each courier can hold packages with total weight <= max carriable weight
    for courier in courier_range:
        solver.add(weights[courier] <= l[courier])

    # Couriers must immediately start with a package after the base
    for courier in courier_range:
        for _time in time_range_no_zero:
            a = y[courier][base_package][_time]
            b = y[courier][base_package][1]

            solver.add(Implies(a != 1, b != 1))

    # Couriers cannot go back to the base before taking other packages
    for courier in courier_range:
        for _time in time_range_no_zero:
            a = y[courier][base_package][_time]

            for _time2 in range(_time + 1, last_time):
                b = y[courier][base_package][_time2]
                solver.add(Implies(a == 1, b == 1))

    # Symmetry breaking constraints
    # integer_matrix = convert_to_integer(y, time_range, courier_range, package_range)

    #If two couriers have the same capacity then they are symmetric, we impose an order between them
    for c1 in courier_range:
        for c2 in courier_range:
            if c1 < c2 and l[c1] == l[c2]:
                solver.add(lex_less_no_conversion(y[c1], y[c2]))

    # Two couriers path are exchangeable if the maximum weight of the two is less than the minimum loading capacity
    for c1 in courier_range:
        for c2 in courier_range:
            if c1 < c2:
                max_weight = If(weights[c1] > weights[c2], weights[c1], weights[c2])
                min_capacity = If(l[c1] < l[c2], l[c1], l[c2])
                condition = max_weight <= min_capacity

                solver.add(Implies(condition, lex_less_no_conversion(y[c1], y[c2])))

    # Heuristic (?)
    for courier in courier_range:
        package_transported = Sum([
            y[courier][package][_time] for _time in time_range for package in package_range if package != base_package
        ])

        solver.add(package_transported <= upper_bound)
        solver.add(package_transported >= lower_bound)
    
    smtlib = solver.to_smt2()
    with(open("smtlib.txt", "w")) as f:
        f.write(smtlib)

if __name__ == "__main__":
    if os.argc < 2:
        print("Usage: python3 multiple_couriers.py <instance>")
        exit(1)
    else:
        instance = os.argv[1]