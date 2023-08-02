# from minizinc import Instance, Model, Solver
import time
import os

from multiprocess.queues import Queue

MODEL_PATH: str = "../model.mzn"
DATA_PATH: str = "instances/data1.dzn"


def main():
    # runner()
    cmd_runner()


# def runner(model_path: str = MODEL_PATH, data_path: str = DATA_PATH) -> (list[list[str]], int, float):
#     """
#     A first try of MiniZinc runner using minizinc python library.
#     Not working very well.
#     :param model_path: the path of the wanted model
#     :param data_path: the path of the .dat instance
#     :return: The solution, the minimized distance and the time passed
#     """
#     model = Model(model_path)
#     instance = Instance(Solver.lookup("chuffed"), model)
#     instance.add_file(data_path)
#     start = time.time()
#     try:
#         result = instance.solve()
#     except Exception as E:
#         print(E)
#         return [], 0, 0
#     if result is None or result is None:
#         return [], 0, 0
#     end = time.time()
#     moves = result["c"]
#     paths = get_paths(moves)
#     return paths, result["z"], end - start


def cmd_runner(model: str = MODEL_PATH, data_path: str = DATA_PATH,
               to_ret_1: Queue = None, to_ret_2: Queue = None, to_ret_3: Queue = None,
               solver: str = "Gecode") -> (list[list[str]], int, str):
    """
    The actual MiniZinc model runner using system commands.
    :param model: The path of the used model
    :param data_path: The path of .dat instance
    :param to_ret_1: Queue for async return of solution
    :param to_ret_2: Queue for async return of minimized distance
    :param to_ret_3: Queue for async return of time of computation
    :param solver: MiniZinc used solver
    :return: Solution, Minimized distance and time of computation
    """
    start = time.time()
    output = os.popen(f"minizinc -p 8 --solver {solver} {model} {data_path} --time-limit 30000 --intermediate").read()
    if output == "":
        return [], 0, 0
    splitter = "----------"
    if len(output.split(splitter)) < 2:
        return [], 0, 0
    result = output.split(splitter)[-2] + splitter + "\n=========="
    result = result.strip() + "\n"
    print(result)
    if result != "" and result[0] != "c":
        return [], 0, 0
    end = time.time()
    # paths, dist = out_formatter(result)
    paths, dist = default_out_formatter(result)
    if to_ret_1 is not None:
        to_ret_1.put(paths)
        to_ret_2.put(dist)
        to_ret_3.put(end - start)
    print(f"Model:  {model.split('/')[1]}")
    print(f"DISTANCE: {dist}\n")
    print("CSP SOLUTION: \n__________________\n")
    for path in paths:
        for i in range(len(path)):
            if i != len(path) - 1:
                print(path[i], end=", ")
            else:
                print(path[i])
    print("\n__________________\n")
    return paths, dist, end - start


def default_out_formatter(result: str) -> (list[list[str]], int):
    """
    Utility function for parsing cmd output into true solution
    :param result: CMD output of MiniZinc
    :return: Solution and minimized distance parsed
    """
    splitted = result.split("\n")[1:-3]
    dist = int(splitted[-1].split("=")[-1][1:-1])
    splitted = splitted[:-2]
    paths = []
    for row in splitted:
        path = []
        true_row = row.split("|")[-1]
        digits = true_row.split(",")
        for digit in digits:
            path.append(digit)
        paths.append(path)
    paths = get_paths(paths)
    return paths, dist


def get_paths(moves: list[list[str]]) -> list[list[str]]:
    """
    Utility function for parsing the paths of the solution
    :param moves: strings of MiniZinc path cmd output
    :return: Parsed path of solution
    """
    paths = []
    for courier in moves:
        path = ["s"]
        start = 1
        tmp = courier[start:]
        end = tmp.index(tmp[len(tmp) - 1])
        tmp = tmp[:end]
        for elem in tmp:
            path.append(str(elem))
        path.append("s")
        paths.append(path)
    return paths


if __name__ == "__main__":
    main()
