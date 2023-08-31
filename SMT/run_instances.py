from SMT.SMTordinecorretto import solve_one as fixed_solve
from SMT.SMT_naive import solve_one as first_solve
from utils.converter import get_instances
import multiprocessing
import json
import matplotlib.pyplot as plt
import numpy as np

N_INST: int = 2

MODELS: list[any] = [first_solve, fixed_solve]

NAMES: list[str] = ["Naive SMT", "Fixed SMT"]
FONTSIZE: int = 14

def main():
    """
    In main we simply run each model on each instance of the chosen set
    The output are .json files in the 'res' directory with solution, minimized distance,
    boolean of optimality and time of computation
    """
    instances = get_instances()
    times = [[] for _ in range(len(MODELS))]
    all_results = []
    for i in range(1, N_INST + 1):
        results = []
        j = 0
        for model in MODELS:
            to_print = run_instance(i, instances, model)
            results.append(to_print)
            all_results.append(to_print)
            times[j].append(to_print["time_passed"][0])
            j += 1
        to_out = {}
        for to_print, name in zip(results, NAMES):
            print(to_print)
            optimal = True
            if to_print["time_passed"][0] == 300:
                optimal = False
            to_out[name] = {
                "time": int(to_print["time_passed"][0]),
                "optimal": optimal,
                "obj": to_print["min_dist"][0],
                "sol": [[int(elem) for elem in courier if int(elem) != int(courier[0])] for courier in
                        to_print["sol"][0]] if to_print["sol"][0] != "Unsat" else to_print["sol"][0]
            }
        with open(f"../res/SMT/{i + 1}.json", "w+") as f:
            json.dump(to_out, f, indent=4)
    # with open("SMT_data.json", "w+") as f:
    #     json.dump(all_results, f)

    # This is for plotting computation time
    x = np.array([i for i in range(1, N_INST + 1)])
    plt.grid()
    plt.title("Time passed computing each instance", weight="bold", fontsize=FONTSIZE)
    plt.xlabel("Instance n°", weight="bold", fontsize=FONTSIZE)
    plt.ylabel("Average time (in sec)", weight="bold", fontsize=FONTSIZE)
    for i in range(len(NAMES)):
        plt.plot(x, times[i], "-o", label=NAMES[i], linewidth=5)
    plt.legend()
    plt.show()


def run_instance(i: int, instances: list, model: any) -> dict:
    """
    Function which actually solve an instance
    :param i: Index of the instance to solve
    :param instances: List of all instances available
    :param model: SMT solver
    :return: A dictionary containing solution, minimized distance, time of computation and iterations
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
