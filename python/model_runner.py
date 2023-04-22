from minizinc import Instance, Model, Solver
import time

MODEL_PATH = "../model.mzn"
DATA_PATH = "../data1.dzn"

def main():
    runner()

def runner(model_path=MODEL_PATH, data_path=DATA_PATH):
    model = Model(model_path)
    instance = Instance(Solver.lookup("chuffed"), model)
    instance.add_file(data_path)
    start = time.time()
    result = instance.solve()
    end = time.time()
    moves = result["c"]
    paths = get_paths(moves)
    return paths, result["z"], end-start

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