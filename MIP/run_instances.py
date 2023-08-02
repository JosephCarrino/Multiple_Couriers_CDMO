from MIP.MIP_run import solve_one
from utils.converter import get_file
import multiprocessing
import json
import matplotlib.pyplot as plt
import numpy as np

N_INST: int = 10

NAMES: list[str] = ["Balanced MIP", "Feasibility MIP", "Optimality MIP"]
MODELS = range(3)

FONTSIZE: int = 14


def main():
    """
    This main simply run all models on the whole batch of selected instances
    The output is a precise format of .json files with found solution, time of computation, minimized distance
    and optimality boolean.
    """
    instances = get_file()
    times = [[] for _ in range(len(MODELS))]
    all_results = []
    for i in range(1, N_INST + 1):
        results = []
        j = 0
        for param in MODELS:
            to_print = run_instance(i, instances, param)
            results.append(to_print)
            all_results.append(to_print)
            times[j].append(to_print["time_passed"][0])
            j += 1

        to_out = {}
        for to_print, name in zip(results, NAMES):
            optimal = True
            if to_print["time_passed"][0] >= 300:
                optimal = False
            to_out[name] = {
                "time": int(to_print["time_passed"][0]),
                "optimal": optimal,
                "obj": to_print["min_dist"][0],
                "sol": [[int(elem) for elem in courier if int(elem) != int(courier[0])] for courier in
                        to_print["sol"][0]] if to_print["sol"][0] != "Unsat" else to_print["sol"][0]
            }
        with open(f"../res/MIP/{i}.json", "w+") as f:
            json.dump(to_out, f, indent=4)
    # with open("MIP_data.json", "w+") as f:
    #     json.dump(all_results, f)
    x = np.array([i for i in range(1, N_INST + 1)])
    plt.grid()
    plt.title("Time passed computing each instance", weight="bold", fontsize=FONTSIZE)
    plt.xlabel("Instance n°", weight="bold", fontsize=FONTSIZE)
    plt.ylabel("Average time (in sec)", weight="bold", fontsize=FONTSIZE)
    for i in range(len(NAMES)):
        plt.plot(x, times[i], "-o", label=NAMES[i], linewidth=5)
    plt.legend()
    plt.show()


def run_instance(i: int, instances: list[dict], param: int) -> dict:
    """
    Function that solve a single instance using MIP solver with given parameter
    :param i: Instance index
    :param instances: List of available instances
    :param param: MIP solver emphasis parameter
    :return: A dictionary with found solution, minimized distance, time of computation and number of iterations
    """
    timeouted = False
    to_print = {"sol": [], "min_dist": [], "time_passed": [], "iter": []}
    sol = multiprocessing.Queue()
    min_dist = multiprocessing.Queue()
    time_passed = multiprocessing.Queue()
    iterations = multiprocessing.Queue()
    p = multiprocessing.Process(target=solve_one, args=(instances, i - 1, min_dist, sol, time_passed, param))
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
