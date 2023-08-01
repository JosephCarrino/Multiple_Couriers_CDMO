from .MIP import solve_one
from .converter import get_file
import multiprocessing
import json
import matplotlib.pyplot as plt
import numpy as np

N_INST = 5
FONTSIZE = 14

NAMES = ["Balanced MIP", "Feasibility MIP", "Optimality MIP"]
MODELS = range(3)

def main():
    instances = get_file()
    times = [[] for i in range(len(MODELS))]
    all_results = []
    for i in range(1, N_INST):
        results = []
        j = 0
        for param in MODELS:
            to_print = run_instance(i, instances, param)
            results.append(to_print)
            all_results.append(to_print)
            times[j].append(to_print["time_passed"][0])
            j+=1
        
        for to_print, name in zip(results, NAMES):
            print(to_print)
            optimal = True
            if to_print["time_passed"][0] == 300:
                optimal = False
            to_out[name] = {
                "time": int(to_print["time_passed"][0]),
                "optimal": optimal,
                "obj": to_print["min_dist"][0],
                "sol": [[int(elem) for elem in courier if int(elem) != int(courier[0])] for courier in  to_print["sol"][0]] if to_print["sol"][0] != "Unsat" else to_print["sol"][0]
            }
        with open(f"../res/MIP/{i+1}.json", "w+") as f:
            json.dump(to_out, f, indent=4)
    with open("MIP_data.json", "w+") as f:
        json.dump(all_results, f)
    x = np.array([i for i in range(1, N_INST+1)])
    plt.grid()
    plt.title("Time passed computing each instance", weight = "bold", fontsize = FONTSIZE)
    plt.xlabel("Instance n°", weight = "bold", fontsize = FONTSIZE)
    plt.ylabel("Average time (in sec)", weight = "bold", fontsize = FONTSIZE)
    for i in range(len(NAMES)):
        plt.plot(x, times[i], "-o", label=NAMES[i], linewidth=5)
    plt.legend()
    plt.show()


def run_instance(i, instances, param):
    timeouted = False
    to_print = {"sol": [], "min_dist": [], "time_passed": [], "iter": []}
    sol = multiprocessing.Queue()
    min_dist = multiprocessing.Queue()
    time_passed = multiprocessing.Queue()
    iter = multiprocessing.Queue()
    p = multiprocessing.Process(target=solve_one, args=(instances, i-1, min_dist, sol, time_passed, param))
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
    to_print["iter"].append(iter.get() if not iter.empty() else 0)
    return to_print


if __name__ == "__main__":
    main()