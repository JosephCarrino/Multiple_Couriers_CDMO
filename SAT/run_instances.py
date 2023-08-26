from SAT.SAT_run import solve_one
from utils.converter import get_file
import multiprocessing
import json
import matplotlib.pyplot as plt
import numpy as np

N_INST: int = 10
FONTSIZE = 14

MODELS: list[any] = [solve_one]

NAMES: list[str] = ["SAT"]

def main():
    """
    This main simply run all models on the whole batch of selected instances
    The output is a precise format of .json files with found solution, time of computation, minimized distance
    and optimality boolean.
    """
    instances = get_file()
    times = []
    for i in range(1, N_INST + 1):
        results = []
        j = 0
        to_print = run_instance(i, instances)
        results.append(to_print)
        times.append(to_print["time_passed"][0])
        j += 1

        to_out = {}
        optimal = True
        if to_print["time_passed"][0] >= 300:
            optimal = False
        to_out = {
            "time": int(to_print["time_passed"][0]),
            "optimal": optimal,
            "obj": to_print["min_dist"][0],
            "sol": [[int(elem) for elem in courier if int(elem) != int(courier[0])] for courier in
                    to_print["sol"][0]] if to_print["sol"][0] != "Unsat" else to_print["sol"][0]
        }
        with open(f"../res/SAT/{i}.json", "w+") as f:
            json.dump(to_out, f, indent=4)
    x = np.array([i for i in range(1, N_INST + 1)])
    plt.grid()
    plt.title("Time passed computing each instance", weight="bold", fontsize=FONTSIZE)
    plt.xlabel("Instance n°", weight="bold", fontsize=FONTSIZE)
    plt.ylabel("Average time (in sec)", weight="bold", fontsize=FONTSIZE)
    plt.plot(x, times, "-o", label="SAT", linewidth=5)
    plt.legend()
    plt.show()


def run_instance(i: int, instances: list[dict], model: any) -> dict:
    """
    Function that solve a single instance using SAT solver
    :param i: Instance index
    :param instances: List of available instances
    :return: A dictionary with found solution, minimized distance, time of computation and number of iterations
    """
    timeouted = False
    to_print = {"sol": [], "min_dist": [], "time_passed": [], "iter": []}
    sol = multiprocessing.Queue()
    min_dist = multiprocessing.Queue()
    time_passed = multiprocessing.Queue()
    iterations = multiprocessing.Queue()
    p = multiprocessing.Process(target=model, args=(instances, i - 1, min_dist, sol, time_passed, iterations))
    p.start()
    p.join(300)
    if p.is_alive():
        print("TIMEOUT")
        timeouted = True
        p.terminate()
        p.join()
    to_print["sol"].append(sol.get() if not sol.empty() else "Unsat")
    to_print["min_dist"].append(min_dist.get() if not min_dist.empty() else 0)
    to_print["time_passed"].append(float(time_passed.get()) if not time_passed.empty() and not timeouted else 300)
    to_print["iter"].append(iterations.get() if not iterations.empty() else 0)
    return to_print


if __name__ == "__main__":
    main()
