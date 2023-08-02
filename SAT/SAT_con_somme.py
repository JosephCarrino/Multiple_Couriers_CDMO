from z3 import *
import numpy as np
from itertools import combinations
from ..utils.converter import get_file
from time import time

MAXITER = 500
MAX_BIT = 16


def at_least_one(bool_vars):
    return Or(bool_vars)


def at_most_one(bool_vars):
    return [Not(And(pair[0], pair[1])) for pair in combinations(bool_vars, 2)]


def exactly_one(bool_vars):
    return at_most_one(bool_vars) + [at_least_one(bool_vars)]


def at_most_k_seq(x, k, name):
    clauses = []
    n = len(x)
    s = [[Bool(f"s_{i}_{j}_{name}") for j in range(k)] for i in range(n - 1)]
    clauses.append(Or(Not(x[0]), s[0][0]))
    clauses += [Not(s[0][j]) for j in range(1, k)]
    clauses.append(Or(Not(x[n - 1]), Not(s[n - 2][k - 1])))
    for i in range(1, n - 1):
        clauses.append(Or(Not(x[i]), s[i][0]))
        clauses.append(Or(Not(s[i - 1][0]), s[i][0]))
        for j in range(1, k):
            clauses.append(Or(Not(s[i - 1][j]), s[i][j]))
            clauses.append(Or(Not(x[i]), Not(s[i - 1][j - 1]), s[i][j]))
        clauses.append(Or(Not(x[i]), Not(s[i - 1][k - 1])))
    return And(clauses)


# def adder(a, b, MAX_BIT=16):
#     """
#     a = [a_1, a_2, a_3, ..., a_{max_bit}]
#     b = [b_1, b_2, b_3, ..., b_{max_bit}]
#
#     d = [d_1, d_2, d_3, ..., d_{max_bit}]
#     :param a:
#     :param b:
#     :return:
#     """
#     clauses = []
#     carries = [Bool(f"carry") for _ in range(MAX_BIT)]
#     d = [Bool(f"res") for _ in range(MAX_BIT)]
#
#     # Set carries_0 = carries_max_bit = 0
#
#     clauses.append(And(Not(carries[0]), Not(carries[MAX_BIT - 1])))
#
#     for i in range(MAX_BIT):
#         c1 = And(a[i], Not(b[i]), Not(carries[i]))
#         c2 = And(Not(a[i]), b[i], Not(carries[i]))
#         c3 = And(Not(a[i]), Not(b[i]), carries[i])
#         c4 = And(a[i], b[i], carries[i])
#
#         c = d[i] == Or(c1, c2, c3, c4)
#
#         clauses.append(c)
#
#         if i > 0:
#             clauses.append(
#                 carries[i - 1] == Or(And(a[i], b[i]), And(a[i], c[i]), And(b[i], c[i]))
#             )


# def adder(a, b, MAX_BIT = 16):
#     a_bits = [Bool(f'a_{i}') for i in range(MAX_BIT)]
#     b_bits = [Bool(f'b_{i}') for i in range(MAX_BIT)]
#     result_bits = [Bool(f'result_{i}') for i in range(MAX_BIT)]
#     clauses = []
#
#     # Assert constraints for each bit of the input numbers
#     for i in range(MAX_BIT):
#         clauses.append(Implies(a_bits[i], a & (1 << i) != 0))  # Convert variable to bit constraint
#         clauses.append(Implies(b_bits[i], b & (1 << i) != 0))  # Convert variable to bit constraint
#
#     # Assert the adder constraints
#     carry = Bool('carry')
#     clauses.append(result_bits[0] == Xor(a_bits[0], b_bits[0]))  # XOR operation for the least significant bit
#     clauses.append(carry == And(a_bits[0], b_bits[0]))  # AND operation for the carry
#
#     for i in range(1, MAX_BIT):
#         # XOR operation for the current bit with the carry from the previous bit
#         clauses.append(result_bits[i] == Xor(Xor(a_bits[i], b_bits[i]), carry))
#         # AND operation for the carry
#         clauses.append(carry == (Or(And(a_bits[i], b_bits[i]), And(b_bits[i], carry), And(carry, a_bits[i]))))
#
#     # Convert the result bits to an integer
#     result = sum([(2 ** i) * If(result_bits[i], 1, 0) for i in range(MAX_BIT)])
#
#     return result, clauses

conversion_number = 0


def convert_to_binary(a, MAX_BIT=MAX_BIT):
    if type(a) != int:
        raise Exception("Input must be an integer")

    global conversion_number
    conversion_number += 1

    clauses = []
    a_bits = [Bool(f'a_conversion_{conversion_number}_{i}') for i in range(MAX_BIT)]

    for i in range(MAX_BIT):
        t = a & (1 << i) != 0
        clauses.append(a_bits[i] == t)#Implies(a_bits[i], a & (1 << i) != 0))  # Convert variable to bit constraint

    return a_bits, clauses


sum_number = 0


def adder(a, b, MAX_BIT=MAX_BIT):
    """
    a = [a_1, a_2, a_3, ..., a_{max_bit}]
    b = [b_1, b_2, b_3, ..., b_{max_bit}]
    """

    global sum_number
    sum_number += 1

    result = [Bool(f"res_{sum_number}_{i}") for i in range(MAX_BIT)]
    clauses = []

    # Assert the adder constraints
    carry = [Bool(f"carry_{sum_number}_{i}") for i in range(MAX_BIT)]
    clauses.append(result[0] == Xor(a[0], b[0]))  # XOR operation for the least significant bit
    clauses.append(carry[0] == And(a[0], b[0]))  # AND operation for the carry

    for i in range(1, MAX_BIT):
        # XOR operation for the current bit with the carry from the previous bit
        clauses.append(result[i] == Xor(Xor(a[i], b[i]), carry[i - 1]))
        # AND operation for the carry
        clauses.append(carry[i] == Or(And(a[i], b[i]), And(b[i], carry[i - 1]), And(carry[i - 1], a[i])))

    return result, clauses



def adder_list(l, MAX_BIT = MAX_BIT):
    # if len(l) == 1:
    #     return l[0]
    # else:
    #     return adder(l[0], adder_list(l[1:], MAX_BIT)) #([adder(l[0], l[1])] + l[2:])

    if len(l) == 1:
        return l[0], []

    if len(l) == 2:
        return adder(l[0], l[1], MAX_BIT)
    else:
        sub_result, sub_clauses = adder_list(l[1:], MAX_BIT)
        result, clauses = adder(l[0], sub_result, MAX_BIT)
        return result, sub_clauses + clauses




# less_than_number = 0
def less_than(a, b):
    """
    a = [a_1, a_2, a_3, ..., a_{max_bit}]
    b = [b_1, b_2, b_3, ..., b_{max_bit}]
    """

    # global less_than_number
    # less_than_number += 1

    # Assert the less than constraint
    # less_than_var = Bool(f'less_than_{less_than_number}')

    if not a:
        return [True]
    if not b:
        return [False]

    return [Or(
        And(b[-1], Not(a[-1])),       # a < b
        And(b[-1] == a[-1],  less_than(a[:-1], b[:-1])[0]))]

    # clauses.append(Or(
    #     And(b[0], Not(a[0])),       # a < b
    #     And(Not(b[0]), Not(a[0]), less_than(a[1:], b[1:], MAX_BIT - 1))
    # ))

    # return less_than_var, clauses


convert_to_zero_number = 0
def if_convert_to_zero(condition, a):
    clauses = []

    global convert_to_zero_number
    convert_to_zero_number += 1

    a_new = [Bool(f'if_convert_to_zero_{convert_to_zero_number}_{i}') for i in range(MAX_BIT)]
    for i in range(MAX_BIT):
        clauses.append(If(condition, a_new[i] == a[i], a_new[i] == False))

    return a_new, clauses


def multiple_couriers(m, n, D, l, s):
    solver = Solver()
    # upper_bound, lower_bound = calculate_bound_package(m, n, D, l, s)

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

    import numpy as np

    # Variables
    # y[c][t][p] == 1 se c porta p al tempo t
    y = np.array([np.array([np.array([Bool(f"y_{courier}_{_time}_{package}")
                                      for package in package_range])
                            for _time in time_range])
                  for courier in courier_range])

    # weights vector, weights[c] is the amount transported by c
    weights = []
    for courier in courier_range:
        _sum = [Bool(f"sum_weight_{courier}_{i}") for i in range(16)]

        # _sum = [
        #     [[Bool(f"sum_weight_{courier}_{_time}_{package}_{i}") for i in range(16)]
        #     for _time in time_range]
        #     for package in package_range
        # ]

        intermediate_results = [[Bool(f"intermediate_res_{courier}_{0}") for _ in range(16)]]
        for _time in time_range:
            for package in package_range:
                if package == base_package:
                    continue

                s_binary, clauses_convert = convert_to_binary(s[package])
                s_new, clauses_condition = if_convert_to_zero(y[courier, _time, package], s_binary)
                # print(clauses_condition)

                res, clauses = adder(intermediate_results[-1], s_new)
                intermediate_results.append(res)
                clauses += clauses_convert + clauses_condition
                solver.add(clauses)

        _sum = intermediate_results[-1]
        weights.append(_sum)

    # Distance vector, distances[c] is the distance traveled by c
    distances = []
    for courier in courier_range:
        _sum = [Bool(f"sum_distance_{courier}_{i}") for i in range(MAX_BIT)]

        intermediate_results = [[Bool(f"intermediate_res_{courier}_{0}") for _ in range(MAX_BIT)]]
        for _time in time_range_no_zero:
            for p1 in package_range:
                for p2 in package_range:
                    condition = And(y[courier, _time - 1, p1], y[courier, _time, p2])

                    d_binary, clauses_convert = convert_to_binary(D[p1][p2])
                    d_new, clauses_condition = if_convert_to_zero(condition, d_binary)

                    result, clauses = adder(intermediate_results[-1], d_new)
                    clauses += clauses_convert + clauses_condition
                    intermediate_results.append(result)
                    solver.add(clauses)

        _sum = intermediate_results[-1]
        distances.append(_sum)


    # At each time, the courier carries exactly one package or he is at base
    for courier in courier_range:
        for _time in time_range:
            solver.add(exactly_one(y[courier, _time, :].tolist()))

    # Each package is carried only once
    for package in package_range:
        if package == base_package:
            continue
        solver.add(exactly_one(y[:, :, package].flatten().tolist()))


    # The total weight carried by each courier must be less or equal than his
    # carriable weight
    for courier in courier_range:
        capacity_binary, clauses_conversion = convert_to_binary(l[courier])
        clauses = less_than(weights[courier], capacity_binary)

        clauses += clauses_conversion

        solver.add(clauses)

    # for i in range(m):
    #     vars = []
    #     for j in range(n):
    #         for k in range(s[j]):
    #             vars.append(w[i][j][k])
    #     solver.add(at_most_k_seq(vars, l[i], f"weight_{i}"))

    # At start/end the courier must be at the base
    for courier in courier_range:
        solver.add(y[courier, 0, base_package])
        solver.add(y[courier, last_time, base_package])

    # Couriers must immediately start with a package after the base
    for courier in courier_range:
        for _time in time_range_no_zero:
            a = y[courier, _time, base_package]
            b = y[courier, 1, base_package]

            solver.add(Implies(Not(a), Not(b)))

    # Couriers cannot go back to the base before taking other packages
    for courier in courier_range:
        for _time in time_range_no_zero:
            a = y[courier, _time, base_package]

            for _time2 in range(_time + 1, last_time):
                b = y[courier, _time2, base_package]
                solver.add(Implies(a, b))

    # Optimization

    objective_value = [Bool(f"objective_value_{i}") for i in range(MAX_BIT)]
    for courier in courier_range:
        clauses = less_than(distances[courier], objective_value)
        solver.add(clauses)

    min_distance = 9999999
    for i in range(len(D)):
        for j in range(len(D[i])):
            if D[i][j] <= min_distance and D[i][j] != 0:
                min_distance = D[i][j]

    max_distance = 0
    for i in range(len(D)):
        max_distance += max(D[i])


    # for a in solver.assertions():
    #     print(a)

    print(f"Assertion len {len(solver.assertions())}")

    iter = 1
    start_time = time()
    last_sol = None
    while iter < MAXITER:
        k = int((min_distance + max_distance) / 2)

        # Getting the maximum distance
        solver.push()

        # Convert k to binary
        k_binary, clauses_conversion = convert_to_binary(k)
        clauses = less_than(objective_value, k_binary)

        clauses += clauses_conversion

        solver.add(clauses)


        # solver.add(at_most_k_seq(dists, k, f"Courier_dist_{i}"))

        sol = solver.check()

        if sol != sat:
            min_distance = k
        else:
            max_distance = k
            last_sol = solver.model()

        print(f"ITERATION: {iter} - TIME: {time() - start_time} - STATUS: {sol} - DISTANCE: {k}")

        if sol == sat:
            g = last_sol
            print("SAT SOLUTION: \n__________________\n")
            for courier in courier_range:
                t = ""
                for _time in time_range:
                    value = sum(package * g.eval(y[courier][_time][package]) for package in package_range)
                    t += f"{g.eval(value + 1)}, "

                print(t)

            print("\n__________________\n")

        if abs(min_distance - max_distance) <= 1:
            g = last_sol
            print("SAT SOLUTION: \n__________________\n")
            for courier in courier_range:
                t = ""
                for _time in time_range:
                    value = sum(package * g.eval(y[courier][_time][package]) for package in package_range)
                    t += f"{g.eval(value + 1)}, "

                print(t)

            print("\n__________________\n")
            return 0, last_sol, f"{time() - start_time:.2f}", iter

        iter += 1


        solver.pop()

    return k, last_sol, f"{time() - start_time:.2f}", iter


def minimizer_binary(instance, solver=multiple_couriers, maxiter=MAXITER):
    m = instance["m"]
    n = instance["n"]
    D = instance["D"]
    l = instance["l"]
    s = instance["s"]
    l[0] = 15
    l[1] = 10
    return solver(m, n, D, l, s)


def solve_one(instances, idx, to_ret1=None, to_ret2=None, to_ret3=None, to_ret4=None):
    sol, mindist, time_passed, iter = minimizer_binary(instances[idx])
    if to_ret1 != None:
        to_ret1.put(sol)
        to_ret2.put(mindist)
        to_ret3.put(time_passed)
        to_ret4.put(iter)
    return sol, mindist, time_passed, iter


def test():
    bit = 5
    solver = Solver()
    x = [Bool(f"x_{i}") for i in range(bit)]
    y = [Bool(f"y_{i}") for i in range(bit)]



    result, clauses = adder(x, y, bit)
    solver.add(clauses)

    binary_10, clauses = convert_to_binary(10)
    solver.add(clauses)
    print(clauses)

    for i in range(bit):
        solver.add(binary_10[i] == result[i])



    # print(solver.assertions())
    print(solver.check())
    # print(solver.model())


    t = sum([2**i * solver.model().eval(x[i]) for i in range(bit)])
    x_value = solver.model().eval(t)
    print(f"x = {x_value}")

    t = sum([2**i * solver.model().eval(y[i]) for i in range(bit)])
    y_value = solver.model().eval(t)
    print(f"y = {y_value}")


    t = sum([2**i * solver.model().eval(binary_10[i]) for i in range(bit)])
    y_value = solver.model().eval(t)
    print(f"y = {y_value}")

def main():
    instances = get_file()
    _, mindist, t, _ = solve_one(instances, 0)
    # print(f"Min distance {mindist}")
    # test()
    print(f"Time passed {t}s")


if __name__ == "__main__":
    main()
