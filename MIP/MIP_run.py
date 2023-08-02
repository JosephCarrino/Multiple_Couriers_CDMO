import collections

from mip import Model, minimize, INTEGER, BINARY, xsum, Var
import itertools

from multiprocess.queues import Queue

from utils.converter import get_file
from time import time


def get_package_at_time_t(_time: int, courier: int, y: list[list[list[Var]]],
                          package_range: collections.Iterable) -> any:
    """
    Function that returns the package taken by a certain courier at a given time
    :param _time: Given time
    :param courier: Given courier
    :param y: Solver variables of booleans
    :param package_range: Number of packages
    :return: Package took at time 'time' by courier 'courier'
    """
    return sum(package * y[package][_time][courier] for package in package_range).x


def and_constraint(model, A, B):
    """
    Function that defines the and operator
    :param model: Solver where to add the constraint
    :param A: First formula
    :param B: Second formula
    :return: The built constraint
    """
    # A and B
    # A = 0, B = 0 => 0
    # A = 0, B = 1 => 0
    # A = 1, B = 0 => 0
    # A = 1, B = 0 => 1

    # v_and new variable
    # v_and <= A
    # v_and <= B
    # v_and >= A + B - 1

    v_and = model.add_var(name=f"and({A.name}, {B.name})", var_type=BINARY)

    model += v_and <= A
    model += v_and <= B
    model += v_and >= A + B - 1

    return v_and


def implies_constraint(model, A, B):
    """
    Function that defines the imply operator
    :param model: Solver where to add the constraint
    :param A: First formula
    :param B: Second formula
    :return: The built constraint
    """
    # A => B
    # A = {0, 1}

    # If A == 1 then B == 1
    # else B == {0, 1}

    # A <= B <= 1 + A
    # B <= 1

    # A = 0 ... 0 <= B <= 1
    # A = 1 ... 1 <= B <= 2 , B <= 1 => B = 1

    model += A <= B
    model += B <= 1 + A
    model += B <= 1


EMPH_TO_NAME = {0: "Balanced MIP", 1: "Feasibility MIP", 2: "Optimality MIP"}


def solve_multiple_couriers(m: int, n: int, D: list[list[int]], l: list[int], s: list[int],
                            emph: int = 0) -> (list[list[str]], int):
    """
    SMT solver with just carriabl weights heuristic
    :param m: Number of couriers
    :param n: Number of packages
    :param D: Matrix of packages distances
    :param l: List of carriable weight by each courier
    :param s: List of weights of the packages
    :param emph: Parameter of solver emphasis (Balanced, Feasibility, Optimality)
    :return: Found solution and minimized distance
    """
    # Create the MIP model
    model = Model()
    model.emphaisis = emph
    # So that the package representing the base doens't count in the weight calculation
    s += [0]

    # Ranges
    package_range = range(n + 1)
    time_range = range(n + 2)
    time_range_no_zero = range(1, time_range[-1])
    courier_range = range(m)

    # Constant
    base_package = n
    last_time = n + 1
    list(itertools.product(courier_range, time_range))

    # Variables
    # y[p][t][c] == 1 se c porta p al tempo t
    y = [
        [
            [
                model.add_var(name=f"y_{package}_{_time}_{courier}", var_type=BINARY)
                for courier in courier_range
            ]
            for _time in time_range
        ]
        for package in package_range
    ]

    # weights vector, weights[c] is the amount transported by c
    weights = []
    for courier in courier_range:
        weights.append(
            xsum(
                s[package] * y[package][_time][courier]
                for _time in time_range
                for package in package_range
            )
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
                    courier_distances.append(D[p1][p2] * and_constraint(model, a, b))

        distances.append(xsum(courier_distances))

    # Each package taken one time only
    for package in package_range:
        if package == base_package:
            continue
        model += (
                xsum(
                    y[package][_time][courier]
                    for _time in time_range
                    for courier in courier_range
                )
                == 1
        )

    # A courier carry only one package at time
    for courier in courier_range:
        for _time in time_range:
            model += xsum(y[package][_time][courier] for package in package_range) == 1

    # Every carried package must be delivered to destination and every courier must start from destination
    for courier in courier_range:
        model += y[base_package][0][courier] == 1
        model += y[base_package][last_time][courier] == 1

    # Each courier can hold packages with total weight <= max carriable weight
    for courier in courier_range:
        model += weights[courier] <= l[courier]

    # Couriers must immediately start with a package after the base
    for courier in courier_range:
        for _time in time_range_no_zero:
            a = y[base_package][_time][courier]
            b = y[base_package][1][courier]

            # 1 - a => 1 - b is saying a != 1 => b != 1
            implies_constraint(model, 1 - a, 1 - b)

    # Couriers cannot go back to the base before taking other packages
    for courier in courier_range:
        for _time in time_range_no_zero:
            a = y[base_package][_time][courier]

            for time2 in range(_time + 1, last_time):
                b = y[base_package][time2][courier]
                implies_constraint(model, a, b)

    # Calculate d_max
    d_max = model.add_var(var_type=INTEGER)
    for courier in courier_range:
        model += d_max >= distances[courier]

    model.objective = minimize(d_max)
    model.verbose = 0
    # Solve the MIP model
    model.optimize(max_seconds=300)

    # Extract the solution
    solution = [
        [
            int(get_package_at_time_t(_time, courier, y, package_range) + 1)
            for _time in time_range
        ]
        for courier in courier_range
    ]

    return solution, model.objective_value


MAXITER = 500


def minimizer_binary(instance: dict, solver: any = solve_multiple_couriers, emph: int = 0) -> (list[list[str]], int):
    """
    Function for solving an instance using the MIP solver
    :param instance: Instance to solve
    :param solver: MIP solver function
    :param emph: Emphasis MIP solver parameter
    :return: Found solution and minimized distance
    """
    m = instance["m"]
    n = instance["n"]
    D = instance["D"]
    l = instance["l"]
    s = instance["s"]

    return solver(m, n, D, l, s, emph=emph)


def solve_one(instances: list[dict], idx: int, to_ret1: Queue = None, to_ret2: Queue = None,
              to_ret3: Queue = None, emph: int = 0) -> (list[list[str]], int, float, int):
    """
    Function that, given an instance, solve it with provided solver and return solution and meta-data
    :param instances: List of available instances
    :param idx: Index of desired instance
    :param to_ret1: Queue for returning minimized distance
    :param to_ret2: Queue for returning found solution
    :param to_ret3: Queue for returning time of computation
    :param emph: MIP solver parameter for emphasis
    :return: Found solution, minimized distance, time of computation and number of iterations
    """
    start_time = time()
    sol, mindist = minimizer_binary(instances[idx], emph=emph)
    time_passed = time() - start_time
    print(f"TIME: {time_passed} - STATUS: {'sat'} - DISTANCE: {mindist}")
    print(f"{EMPH_TO_NAME[emph]} SOLUTION: \n__________________\n")
    for path in sol:
        for i in range(len(path)):
            if i != len(path) - 1:
                print(path[i], end=", ")
            else:
                print(path[i])
    print("\n__________________\n")
    if to_ret1 is not None:
        to_ret1.put(mindist)
        to_ret2.put([[int(elem) for elem in path] for path in sol])
        to_ret3.put(time_passed)
    return sol, mindist, f"{time_passed:.2f}", iter


def main():
    instances = get_file()
    solve_one(instances, 0)


if __name__ == "__main__":
    main()
