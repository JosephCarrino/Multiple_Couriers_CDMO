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


def lex_less(a, b):
    if not a:
        return True
    if not b:
        return False
    return Or(a[0] <= b[0], And(a[0] == b[0], lex_less(a[1:], b[1:])))

def distinct_except(G, ignored):
   if len(G) < 2:
       return BoolSort().cast(True)

   A = K(G[0].sort(), 0);
   for i in range(len(G)):
       A = Store(A, G[i], If(G[i] == ignored, 0, 1 + Select(A, G[i])))

   res = True
   for i in range(len(G)):
       res = And(res, Select(A, G[i]) <= 1)

   return res


def multiple_couriers(m, n, D, l, s):
    # i is the courier
    # j is the step
    c = [[Int(f"c_{i}_{j}") for j in range(n + 2)] for i in range(m)]

    # Ranges
    package_range = range(n + 1)
    time_range = range(n + 2)
    time_range_no_zero = range(1, time_range[-1] + 1)
    courier_range = range(m)

    # D_array = Array('D', IntSort(), ArraySort(IntSort(), IntSort()))
    # for i in range(n):
    #     D_i = Array('D_i', IntSort(), IntSort())
    #     for j in range(n):
    #         D_i = Store(D_i, j, D[i][j])
    #
    #     D_array = Store(D_array, i, D_i)
    #
    # # print(D_array)
    # # print(D_array[3][2])
    #
    # s_array = Array('s', IntSort(), IntSort())
    # for i, elem in enumerate(s):
    #     s_array = Store(s_array, i, elem)

    # Array indicating how much weight the courier i transport
    weights = []

    # Array indicating how much distance the courier i travel
    distances = []

    # list of coordinates of the matrix
    coords = list(itertools.product(range(m), range(n + 2)))

    solver = Solver()

    for i in range(m):
        # Since we should sum D[c[i,j], c[i,j+1]] but we can't use the array c as an index we have do the If to check every possible
        # combination of c[i,j], c[i,j+1]. The range of value for k and w is [1, n+1] since we have to include
        # the value for the base n+1
        distances.append(
            Sum(
                [
                    # D_array[c[i][j] - 1][c[i][j + 1] - 1] for j in range(0, n)
                    If(And(c[i][j] == k, c[i][j + 1] == w), D[k - 1][w - 1], 0)
                    for j in range(0, n)
                    for k in range(1, n + 2)
                    for w in range(1, n + 2)
                ]
            )
        )

        # We have the same problem for the index
        weights.append(
            Sum(
                # [s_array[c[i][j] - 1] for j in range(1, n)]
                [
                    If(c[i][j] == k, s[k - 1], 0)
                    for j in range(1, n)
                    for k in range(1, n + 1)
                ]
            )
        )

    # Constraint

    # Value range of decision variable
    for x, y in coords:
        solver.add(c[x][y] >= 1)
        solver.add(c[x][y] <= n + 1)

    # for x, y in coords:
    #     for xx, yy, in coords:
    #         if (x, y) != (xx, yy):
    #             solver.add(
    #                 Implies(
    #                     And(c[x][y] != n + 1, c[xx][yy] != n + 1),
    #                     Distinct(c[x][y], c[xx][yy])
    #                 ))

    # solver.add(distinct_except(c, n + 1))

    # for package in package_range:
    #     if package == n:
    #         continue
    #
    #     s = Sum([
    #         If(c[x][y] == package + 1, 1, 0)
    #         for x, y in coords
    #     ])
    #
    #     solver.add(s == 1)

    # print(s.sexpr())

    # A package is carried by only one courier only one time
    for x, y in coords:
        solver.add(
            [
                Or(c[x][y] == n + 1, c[x][y] != c[i][j])
                for i, j in coords
                if (x, y) != (i, j)
            ]
        )

    # Each package is assigned to a courier
    for package in range(1, n + 1):
        solver.add(Or([c[i][j] == package for i, j in coords]))


    # The total weight carried by each courier is less than or equal to his maximum carriable weight
    for i in range(m):
        solver.add(weights[i] <= l[i])

    # Every carried package must be delivered to destination and every courier must start from destination
    for i in range(m):
        solver.add(And(c[i][0] == n + 1, c[i][n + 1] == n + 1))

    # Couriers must immediately start with a package after the base
    for i in range(m):
        for j in range(1, n + 1):
            solver.add(Implies(c[i][j] != n + 1, c[i][1] != n + 1))

    # Couriers cannot go back to the base before taking other packages
    for i in range(m):
        for j in range(1, n):
            for k in range(j, n + 1):
                solver.add(Implies(c[i][j] == n + 1, c[i][k] == n + 1))

    # Symmetry breaking
    # If two couriers have the same capacity then they are symmetric, we impose an order between them
    for c1 in range(m):
        for c2 in range(m):
            if c1 < c2 and l[c1] == l[c2]:
                solver.add(lex_less(c[c1], c[c2]))

    # Two couriers path are exchangeable if the maximum weight of the two is less than the minimum loading capacity
    for c1 in range(m):
        for c2 in range(m):
            if c1 < c2:
                max_weight = If(weights[c1] > weights[c2], weights[c1], weights[c2])
                min_capacity = If(l[c1] < l[c2], l[c1], l[c2])
                condition = max_weight <= min_capacity

                solver.add(Implies(condition, lex_less(c[c1], c[c2])))

    # Optimization

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

    for a in solver.assertions():
        #print(a)
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

        if sol == sat:
            g = last_sol
            print("SMT SOLUTION: \n__________________\n")
            for courier in range(m):
                t = ""
                for _time in range(n + 2):
                    t += f"{g.eval(c[courier][_time])}, "
                print(t)

            print("\n__________________\n")

        if abs(min_distance - max_distance) <= 1:
            return max_distance, last_sol, f"{time() - start_time:.2f}", iter

        iter += 1
        solver.pop()

    return max_distance, "Out of _time", f"{time() - start_time:.2f}", iter


def solve_one(instances, idx, to_ret1=None, to_ret2=None, to_ret3=None, to_ret4=None):
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
    _, mindist, t, _ = solve_one(instances, 7)
    print(f"Min distance {mindist}")
    print(f"Time passed {t}s")


if __name__ == "__main__":
    main()
