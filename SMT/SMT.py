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
    # i is the courier
    # j is the step
    c = [[Int(f"c_{i}_{j}") for j in range(n + 2)] for i in range(m)]

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
    # TODO

    # Optimization

    # Getting maximum distance
    max_distance = array_max(distances)

    # The max distance must be less than the minimum distance

    min_dist = 9999999
    for i in range(len(D)):
        for j in range(len(D[i])):
            if D[i][j] <= min_dist and D[i][j] != 0:
                min_dist = D[i][j]
        
    max_dist = 0
    for i in range(len(D)):
        max_dist += max(D[i])
    
    start_time = time()
    iter = 1
    last_sol = None
    while iter < MAXITER:
        k = int((min_dist + max_dist) / 2)

        solver.push()
        solver.add(max_distance < k)

        sol = solver.check()

        if sol != sat:
            min_dist = k
        else:
            max_dist = k
            last_sol = sol
        
        if abs(min_dist - max_dist) <=1:
            if last_sol:
                return min_dist, last_sol, f"{time() - start_time:.2f}", iter
            else:
                return 0, "Unsat", f"{time() - start_time:.2f}", iter
        iter+=1
        solver.pop()

    return 0, "Unsat", f"{time() - start_time:.2f}", iter

def solve_one(instances, idx, to_ret1 = None, to_ret2 = None, to_ret3 = None, to_ret4 = None):

    mindist, sol, time_passed, iter = multiple_couriers(*instances[idx].values())

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