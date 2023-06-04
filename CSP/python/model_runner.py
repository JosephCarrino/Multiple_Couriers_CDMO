from minizinc import Instance, Model, Solver
import time
import os

MODEL_PATH = "../model.mzn"
DATA_PATH = "instances/data1.dzn"

def main():
    # runner()
    cmd_runner()

def runner(model_path=MODEL_PATH, data_path=DATA_PATH):
    model = Model(model_path)
    instance = Instance(Solver.lookup("chuffed"), model)
    instance.add_file(data_path)
    start = time.time()
    try:
        result = instance.solve()
    except Exception as E:
        print(E)
        return [], 0, 0
    if result is None or result == None:
        return [], 0, 0
    end = time.time()
    moves = result["c"]
    paths = get_paths(moves)
    return paths, result["z"], end-start

def cmd_runner(model=MODEL_PATH, data_path = DATA_PATH, to_ret_1 = None, to_ret_2 = None, to_ret_3 = None):
    solver = "Gecode"
    start = time.time()
    result = os.popen(f"minizinc -p 8 --solver {solver} {model} {data_path}").read()
    if result != "" and result[0] != "c":
        return [], 0 , 0
    end = time.time()
    # paths, dist = out_formatter(result)
    paths, dist = default_out_formatter(result)
    if to_ret_1 != None:
        to_ret_1.put(paths)
        to_ret_2.put(dist)
        to_ret_3.put(end-start)
    return paths, dist, end-start

def default_out_formatter(result):
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

def out_formatter(result):
    splitted = result.split("[")
    dist = int(splitted[-1].split("= ")[-1].split("-")[0].strip())
    paths = []
    for part in splitted:
        if "," in part:
            paths.append(part.split("]")[0])
    paths = paths[:-1]
    true_paths = []
    for path in paths:
        true_path = []
        digits = path.split(",")
        for elem in digits:
            true_path.append(int(elem))
        true_paths.append(true_path)
    paths = get_paths(true_paths)
    return paths, dist


def get_paths(moves):
    paths = []
    for courier in moves:
        path = ["s"]
        start = 1
        tmp = courier[start:]
        end = tmp.index(tmp[len(tmp)-1])
        tmp = tmp[:end]
        for elem in tmp:
            path.append(str(elem))
        path.append("s")
        paths.append(path)
    return paths

if __name__ == "__main__":
    main()