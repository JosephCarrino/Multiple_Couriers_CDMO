from multiprocess.queues import Queue
from z3 import *
from time import time
from utils.converter import get_instances

MAXITER: int = 50


def array_max(vs: list, context: Context = None) -> int:
    """
    Function for finding max in an array
    :param vs: Input array
    :return: Max element of vs
    """

    m = vs[0]

    if context is None:
        context = m.ctx

    for _v in vs[1:]:
        m = If(_v > m, _v, m, ctx=context)
    return m


# def lex_less(a: list[list], b: list[list]) -> bool:
#     """
#     Function for detecting lexicographic order between two strings
#     :param a: First string
#     :param b: Second string
#     :return: True if 'a' is ordered before 'b', False otherwise
#     """
#     if not a:
#         return True
#     if not b:
#         return False
#     return Or(a[0] <= b[0], And(a[0] == b[0], lex_less(a[1:], b[1:])))


def lex_less_single(a: list[list], b: list[list], context: Context = None) -> bool:
    if not a or not b:
        return True

    if context is None:
        context = a[0].ctx

    return Or(a[0] <= b[0], And(a[0] == b[0], lex_less_single(a[1:], b[1:], context), context), context)


def lex_less_no_conversion(a: list[list], b: list[list], context: Context = None) -> bool:
    if not a:
        return True
    if not b:
        return False

    if context is None:
        context = a[0][0].ctx

    # Added the last term because sometimes the And was from a different context
    return Or(lex_less_single(a[0], b[0], context),
              And(a[0] == b[0], lex_less_no_conversion(a[1:], b[1:], context), context), context)


def multiple_couriers(
        m: int,
        n: int,
        D: list[list[int]],
        l: list[int],
        s: list[int],
        model_result: dict,
) -> (int, list[list[str]], float, int):
    """
    SMT solver with addition of symmetry breaking constraints such as maintaining couriers lexicographic order
    :param m: Number of couriers
    :param n: Number of packages
    :param D: Matrix of packages distances
    :param l: List of carriable weight by each courier
    :param s: List of weights of the packages
    :return: Minimized distance, found solution, computation time and iterations number
    """

    upper_bound, lower_bound = calculate_bound_package(m, n, l, s)

    # print(f"Upper bound: {upper_bound}")
    # print(f"Lower bound: {lower_bound}")

    # So that the package representing the base doesn't count in the weight calculation
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

    ## Solver ##
    context = Context()
    solver = Then('simplify', 'elim-term-ite', 'solve-eqs', 'smt', ctx=context).solver()
    # solver = Solver(ctx=context)

    # Variables
    # y[p][t][c] == 1 se c porta p al tempo t
    y = [[[Int(f"y_{courier}_{package}_{_time}", ctx=context)
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

    # If two couriers have the same capacity then they are symmetric, we impose an order between them
    for c1 in courier_range:
        for c2 in courier_range:
            if c1 < c2 and l[c1] == l[c2]:
                solver.add(lex_less_no_conversion(y[c1], y[c2]))

    # Two couriers path are exchangeable if the maximum weight of the two is less than the minimum loading capacity
    # for c1 in courier_range:
    #     for c2 in courier_range:
    #         if c1 < c2:
    #             max_weight = If(weights[c1] > weights[c2], weights[c1], weights[c2], ctx=context)
    #             min_capacity = If(l[c1] < l[c2], l[c1], l[c2], ctx=context)
    #
    #             condition = max_weight <= min_capacity
    #             solver.add(Implies(condition, lex_less_no_conversion(y[c1], y[c2])))

    # Heuristic (?)
    for courier in courier_range:
        package_transported = Sum([
            y[courier][package][_time] for _time in time_range for package in package_range if package != base_package
        ])

        solver.add(package_transported <= upper_bound)
        solver.add(package_transported >= lower_bound)

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

    max_distance = math.ceil(max_distance)  # / upper_bound)
    min_distance = max(min_distance, math.floor(min_distance))  # * lower_bound))

    start_time = time()
    iterations = 1
    last_best_sol = None
    while iterations < MAXITER:
        # weight = 0.5  # + (0.2 * math.exp(-0.2 * iter))

        k = int((min_distance + max_distance) / 2)#int(((1 - weight) * min_distance + weight * max_distance))

        solver.push()
        solver.add(objective_value <= k)

        model_result["ready"] = True
        sol = solver.check()

        if sol != sat:
            min_distance = k
        else:
            last_best_sol = solver.model()
            max_distance = last_best_sol.eval(objective_value).as_long()

            # Building solution matrix and store the intermediate solution
            last_solution_matrix = [[0 for _ in range(last_time + 1)] for _ in range(len(courier_range))]

            for courier in courier_range:
                for _time in time_range:
                    value = sum(package * last_best_sol.eval(y[courier][package][_time]) for package in package_range)
                    last_solution_matrix[courier][_time] = last_best_sol.eval(value + 1).as_long()

            model_result["sol"] = last_solution_matrix
            model_result["time"] = time() - start_time
            model_result["iterations"] = iterations
            model_result["min_dist"] = max_distance
            model_result["optimal"] = False

        if abs(min_distance - max_distance) <= 1 or iterations >= MAXITER:
            model_result["optimal"] = True
            return model_result["min_dist"], model_result["sol"], model_result["time"], model_result["iterations"]
        else:
            iterations += 1
            solver.pop()


def calculate_bound_package(m: int, n: int, l: list[int], s: list[int]) -> (int, int):
    """
    Function for calculating bounds for the number of packages carriable by each courier
    :param m: Number of couriers
    :param n: Number of packages
    :param l: List of carriable weight by each courier
    :param s: List of weights of the packages
    :return: General upper and lower bound
    """
    weight_max = max(s)

    # Not able is the number of courier that are not able
    # to transport the CORRECT package
    not_able = len([cap for cap in l if cap < weight_max])

    upper_bound = n - m + not_able
    lower_bound = 1 if not_able == 0 else 0

    return upper_bound, lower_bound

def solve_one(instance: dict, instance_index: int, model_result: dict = None) -> dict:
    if model_result is None:
        model_result = {}

    m, n, D, l, s = instance['m'], instance['n'], instance['D'], instance['l'], instance['s']
    multiple_couriers(m, n, D, l, s, model_result=model_result)

    return model_result