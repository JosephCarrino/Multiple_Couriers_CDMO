import math

from z3 import *
import itertools
from time import time
from converter import get_file

MAXITER = 50

def array_max(vs):
    m = vs[0]
    for v in vs[1:]:
        m = If(v > m, v, m)
    return m
def multiple_couriers(m, n, D, l, s):
    solver = Solver()

    # So that the package representing the base doens't count in the weight calculation
    s += [0]

    # Ranges
    package_range = range(n+1)
    time_range = range(n+2)
    time_range_no_zero = range(1, time_range[-1])
    courier_range = range(m)

    # Constant
    base_package = n
    last_time = n + 1
    coords = list(itertools.product(courier_range, time_range))

    # Variables
    # y[p][t][c] == 1 se c porta p al tempo t
    y = [[[Int( f"y_{package}_{_time}_{courier}")
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


    # Each package taken one _time only
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

    #Couriers must immediately start with a package after the base
    for courier in courier_range:
      for _time in time_range_no_zero:
        a = y[base_package][_time][courier]
        b = y[base_package][1][courier]

        solver.add(Implies(a != 1, b != 1))

    #Couriers cannot go back to the base before taking other packages
    for courier in courier_range:
      for _time in time_range_no_zero:
        a = y[base_package][_time][courier]

        for _time2 in range(_time + 1, last_time):
          b = y[base_package][_time2][courier]
          solver.add(Implies(a == 1, b == 1))

    # Heuristic (?)
    delta = calculate_delta(m, n, D, l, s)
    max_package = min(n, math.ceil(n / m) + delta)
    min_package = max(0, math.floor(n / m) - delta)

    print(f"{delta=}, {max_package=}, {min_package=}")

    for courier in courier_range:
      package_transported = Sum([
          y[package][_time][courier] for _time in time_range for package in package_range if package != base_package
      ])

      solver.add(package_transported <= max_package)
      solver.add(package_transported >= min_package)




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
    max_distance = max_distance / max_package
    min_distance = max(min_distance, min_distance * min_package)
    print(f"{max_distance=}, {min_distance=}")

    start_time = time()
    iter = 1
    last_sol = None
    while iter < MAXITER:
        k = int((min_distance + max_distance) / 2)

        print(f"current k={k}")

        solver.push()
        solver.add(objective_value <= k)

        sol = solver.check()

        if sol != sat:
            min_distance = k
        else:
            max_distance = k
            last_sol = sol


        if abs(min_distance - max_distance) <= 1:
            return max_distance, last_sol, f"{time() - start_time:.2f}", iter

        iter += 1
        solver.pop()


    return max_distance, "Out of _time", f"{time() - start_time:.2f}", iter



def calculate_delta(m, n, D, l, s):
    weight_sum = sum(s)
    capacity_min = min(l)

    if capacity_min >= weight_sum:
        return 0

    capacity_max = max(l)
    weight_max = max(s)

    max_package = capacity_max // weight_max
    min_package = capacity_min // weight_max




    return max_package - min_package    #delta

def solve_one(instances, idx, to_ret1 = None, to_ret2 = None, to_ret3 = None, to_ret4 = None):
    print(instances[idx])
    mindist, sol, time_passed, iter = multiple_couriers(*instances[idx].values())

    if to_ret1 != None:
        to_ret1.put(sol)
        to_ret2.put(mindist)
        to_ret3.put(time_passed)
        to_ret4.put(iter)
    return sol, mindist, time_passed, iter

def main():
    instances = get_file()
    _, mindist, t, _ = solve_one(instances, 3)
    print(f"Min distance {mindist}")
    print(f"Time passed {t}s")

if __name__ == "__main__":
    main()