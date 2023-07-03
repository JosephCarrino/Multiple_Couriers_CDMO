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
    upper_bound, lower_bound = calculate_bound_package(m, n, D, l, s)

    # So that the package representing the base doens't count in the weight calculation
    s += [0]

    # Ranges
    package_range = range(n+1)
    time_range = range(n+2)
    time_range_no_zero = range(1, time_range[-1] + 1)
    courier_range = range(m)

    # Constant
    base_package = n
    last_time = n + 1

    # Variables
    # y[p][t][c] == 1 se c porta p al tempo t
    y = [[[Bool( f"y_{package}_{_time}_{courier}")
          for courier in courier_range]
          for _time in time_range]
          for package in package_range]

    # # Binary constraint
    # # Value range of decision variable
    # for package in package_range:
    #   for courier in courier_range:
    #     for _time in time_range:
    #       solver.add(y[package][_time][courier] <= 1)
    #       solver.add(y[package][_time][courier] >= 0)


    # weights vector, weights[c] is the amount transported by c
    weights = []
    for courier in courier_range:
      weights.append(
          Sum([s[package] * If(y[package][_time][courier], 1, 0) for _time in time_range for package in package_range])
      )

    # distances vector, distances[c] is the amount travelled by c
    distances = []
    for courier in courier_range:
      courier_distances = []
      for _time in time_range_no_zero:
        for p1 in package_range:
          for p2 in package_range:
            a = y[p1][_time - 1][courier]
            b = y[p2][_time][courier]
            courier_distances.append(D[p1][p2] * If(And(a, b), 1, 0))

      distances.append(Sum(courier_distances))


    # Each package taken one time only
    for package in package_range:
      if package == base_package:
        continue
      solver.add(Sum([
          If(y[package][_time][courier], 1, 0) for _time in time_range for courier in courier_range]) == 1)

    # A courier carry only one package at _time
    for courier in courier_range:
      for _time in time_range:
        solver.add(Sum([If(y[package][_time][courier], 1, 0) for package in package_range]) == 1)


    # Every carried package must be delivered to destination and every courier must start from destination
    for courier in courier_range:
        solver.add(y[base_package][0][courier] == True)
        solver.add(y[base_package][last_time][courier] == True)

    # Each courier can hold packages with total weight <= max carriable weight
    for courier in courier_range:
        solver.add(weights[courier] <= l[courier])

    #Couriers must immediately start with a package after the base
    for courier in courier_range:
      for _time in time_range_no_zero:
        a = y[base_package][_time][courier]
        b = y[base_package][1][courier]

        solver.add(Implies(Not(a), Not(b)))

    #Couriers cannot go back to the base before taking other packages
    for courier in courier_range:
      for _time in time_range_no_zero:
        a = y[base_package][_time][courier]

        for _time2 in range(_time + 1, last_time):
          b = y[base_package][_time2][courier]
          solver.add(Implies(a, b))

    # Heuristic (?)
    for courier in courier_range:
      package_transported = Sum([
          If(y[package][_time][courier], 1, 0) for _time in time_range for package in package_range if package != base_package
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
    max_distance = math.ceil(max_distance)# / upper_bound)
    min_distance = max(min_distance, math.floor(min_distance))# * lower_bound))
    print(f"{max_distance=}, {min_distance=}")

    start_time = time()
    iter = 1
    last_sol = None
    while iter < MAXITER:

        weight = 0.5# + (0.2 * math.exp(-0.2 * iter))


        k = int(((1-weight) * min_distance + weight * max_distance))

        print(f"current k={k}, {weight=}")

        solver.push()
        solver.add(objective_value <= k)

        sol = solver.check()

        if sol != sat:
            min_distance = k
        else:
            max_distance = k
            last_sol = solver.model()

        if sol == sat:
            g = last_sol
            print("SMT SOLUTION: \n__________________\n")
            for courier in courier_range:
                t = ""
                for _time in time_range:
                    value = sum(package * g.eval(y[package][_time][courier]) for package in package_range)
                    t += f"{g.eval(value + 1)}, "

                print(t)

            print("\n__________________\n")


        if abs(min_distance - max_distance) <= 1:


            return max_distance, last_sol, f"{time() - start_time:.2f}", iter

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

def solve_one(instances, idx, to_ret1 = None, to_ret2 = None, to_ret3 = None, to_ret4 = None):
    print(instances[idx])
    m, n, D, l, s = instances[idx]['m'], instances[idx]['n'], instances[idx]['D'], instances[idx]['l'], instances[idx]['s']

    """
    import numpy as np
    sort = np.argsort(s)[::-1]
    s_sorted = np.array(s, dtype=np.int32)[sort]
    D_sorted = np.zeros((n+1, n+1), dtype=np.int32)

    print(f"{sort=}")
    print(f"s={np.array(s)}")
    print(f"{s_sorted=}")

    for x in range(n + 1):
        for y in range(n + 1):
            x_new = sort[x] if x != n else n
            y_new = sort[y] if y != n else n
            D_sorted[x][y] = D[x_new][y_new]

    print(D_sorted)
    print(np.array(D))

    s_sorted = s_sorted.tolist()
    D_sorted = D_sorted.tolist()
    """


    mindist, sol, time_passed, iter = multiple_couriers(m, n, D, l, s)#*instances[idx].values()) 27

    if to_ret1 != None:
        to_ret1.put(sol)
        to_ret2.put(mindist)
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