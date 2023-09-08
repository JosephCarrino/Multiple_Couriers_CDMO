import collections

from mip import Model, minimize, INTEGER, BINARY, xsum, Var, OptimizationStatus
import itertools

from multiprocess.queues import Queue

from utils.converter import get_instances
from time import time

EMPH_TO_NAME = {0: "Balanced MIP", 1: "Feasibility MIP", 2: "Optimality MIP"}


def solve_multiple_couriers(
        m: int,
        n: int,
        D: list[list[int]],
        l: list[int],
        s: list[int],
        model_result: dict,
        emph: int = 0,
        timeout: int = 300
) -> (list[list[str]], int):
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

    ### Model creation ###

    model = Model()
    model.emphaisis = emph

    # So that the package representing the base doens't count in the weight calculation
    s += [0]

    ### Ranges ###
    package_range = range(n + 1)
    package_range_no_base = range(n)
    time_range = range(n + 2)
    time_range_no_zero = range(1, time_range[-1])
    courier_range = range(m)

    ### Constant ###
    base_package = n
    last_time = n + 1
    list(itertools.product(courier_range, time_range))

    ### Variables ###

    # y_{courier}_{p1}_{p2} == 1 if courier goes from p1 to p2
    y = [[[model.add_var(name=f"y_{courier}_{p1}_{p2}", var_type=BINARY, lb=0, ub=1)
           for p2 in package_range]
          for p1 in package_range]
         for courier in courier_range]

    weights = []
    for courier in courier_range:
        # s[p1] * y[courier][p1][p2] for p1 in package_range for p2 in package_range
        weights.append(
            xsum(
                s[p1] * y[courier][p1][p2]
                for p1 in package_range
                for p2 in package_range
            )
        )

    distances = []
    for courier in courier_range:
        distances.append(
            xsum(
                D[p1][p2] * y[courier][p1][p2]
                for p1 in package_range
                for p2 in package_range
            )
        )

    ### Constraint ###

    # Correctness of the path
    for courier in courier_range:
        # If y[courier][p1][p2] == 1 then y[courier][p3][p1] == 1
        for p1 in package_range:
            for p2 in package_range:
                # Case 1
                # y[base/p3][p1] == 1
                # y[p1][base/p2] == 1

                # Case 2
                # y[base/p1][p2] == 1
                # y[p2/p3][base/p1] == 1

                # Case 3
                # y[p4/p1][p5/p2]
                # y[p5/p3][p4/p1]

                # Case 4
                # y[p4/p3][p5/p1]
                # y[p5/p1][p4/p2]
                condition = y[courier][p1][p2]
                p3_gen_base = (
                    p3 for p3 in package_range if
                    (p3 == base_package)  # Case 1
                    or (p1 == base_package)  # Case 2
                    or (p3 != p1 and p3 != p2)  # Case 3
                )
                results = xsum(y[courier][p3][p1] for p3 in p3_gen_base)

                # condition = 1 => results = 1
                # condition = 0 => results = {0, 1}
                model += results <= 1
                model += results >= condition

    path_increment = [[model.add_var(name=f"path_increment_{courier}_{p}", var_type=INTEGER, lb=0, ub=n)
                       for p in package_range]
                      for courier in courier_range]

    # Link between y and path_increment
    for courier in courier_range:
        path_increment[courier][base_package] = 0

    for courier in courier_range:
        for p1 in package_range:
            for p2 in package_range_no_base:
                model += path_increment[courier][p2] >= path_increment[courier][p1] + 1 - n * (1 - y[courier][p1][p2])
                model += path_increment[courier][p2] <= path_increment[courier][p1] + 1 + n * (1 - y[courier][p1][p2])

    for courier in courier_range:
        for p in package_range:
            model += path_increment[courier][p] <= xsum([y[courier][p][p2] for p2 in package_range]) * (n + 1)

    ### Optimization ###

    for courier in courier_range:
        for p1 in package_range:
            model += y[courier][p1][p1] == 0

    for courier in courier_range:
        for p1 in package_range:
            for p2 in package_range_no_base:
                condition = y[courier][p1][p2]
                results = xsum(y[courier][p2][p3] for p3 in package_range)

                model += results <= 1
                model += results >= condition

    # Every carried package must be delivered to destination and every courier must start from destination
    for courier in courier_range:
        # Parto dalla base 1 volta
        model += xsum(y[courier][base_package][package] for package in package_range) == 1

        # Torno alla base 1 volta
        model += xsum(y[courier][package][base_package] for package in package_range) == 1

    # Each package is carried only once
    for package in package_range_no_base:
        model += xsum(y[courier][package][p2] for courier in courier_range for p2 in package_range) == 1

    # Each courier can hold packages with total weight <= max carriable weight
    for courier in courier_range:
        model += weights[courier] <= l[courier]

    # Calculate d_max
    d_max = model.add_var(var_type=INTEGER)
    for courier in courier_range:
        model += d_max >= distances[courier]

    model.verbose = 0
    model.objective = minimize(d_max)

    start_time = time()

    model_result["ready"] = True

    # Solve the MIP model
    model.optimize(max_seconds=timeout)

    # Build the solution
    solution = [[base_package + 1
                 for _ in time_range]
                for _ in courier_range]

    for courier in courier_range:
        for p1 in package_range:
            try:
                z_value = int(path_increment[courier][p1].x)
            except:
                z_value = 0

            # print(f"Courier {courier}, package {p1 + 1}, z = {z_value}")
            if z_value != 0:
                solution[courier][z_value] = p1 + 1

    model_result["sol"] = solution
    model_result["time"] = time() - start_time
    model_result["iterations"] = None
    model_result["min_dist"] = model.objective_value
    model_result["optimal"] = model.status == OptimizationStatus.OPTIMAL

    return model_result["sol"], model_result["min_dist"]


def minimizer_binary(
        instance: dict,
        solver: any = solve_multiple_couriers,
        emph: int = 0,
        model_result: dict = None,
        timeout: int = 300,
) -> (list[list[str]], int):
    """
    Function for solving an instance using the MIP solver
    :param instance: Instance to solve
    :param solver: MIP solver function
    :param emph: Emphasis MIP solver parameter
    :return: Found solution and minimized distance
    """
    if model_result is None:
        model_result = {}

    m = instance["m"]
    n = instance["n"]
    D = instance["D"]
    l = instance["l"]
    s = instance["s"]

    return solver(m, n, D, l, s, emph=emph, model_result=model_result, timeout=timeout)


def solve_one(instance: dict, instance_index: int, model_result: dict = None, emph: int = 0,
              timeout: int = 300) -> dict:
    if model_result is None:
        model_result = {}

    minimizer_binary(instance, emph=emph, model_result=model_result, timeout=timeout)

    return model_result
