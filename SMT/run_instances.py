from SMT import solve_one
from converter import get_file
import multiprocessing
import json
import matplotlib.pyplot as plt
import numpy as np

N_INST = 2
FONTSIZE = 14

def main():
    instances = get_file()
    to_print = {"sol": [], "min_dist": [], "time_passed": [], "iter": []}
    for i in range(N_INST):
        sol = multiprocessing.Queue()
        min_dist = multiprocessing.Queue()
        time_passed = multiprocessing.Queue()
        iter = multiprocessing.Queue()

        p = multiprocessing.Process(target=solve_one, args=(instances, i, sol, min_dist, time_passed, iter))
        p.start()
        p.join(300)
        if p.is_alive():
            print("TIMEOUT")
            p.terminate()
            p.join()
        to_print["sol"].append(str(sol.get()) if not sol.empty() else "Unsat")
        to_print["min_dist"].append(min_dist.get() if not min_dist.empty() else 0)
        to_print["time_passed"].append(time_passed.get() if not time_passed.empty() else 300)
        to_print["iter"].append(iter.get() if not iter.empty() else 0)
        print(to_print)
    with open("SAT_data.json", "w+") as f:
        json.dump(to_print, f)
    x = np.array([i for i in range(1, N_INST+1)])
    plt.grid()
    plt.title("Time passed computing each instance", weight = "bold", fontsize = FONTSIZE)
    plt.xlabel("Instance n°", weight = "bold", fontsize = FONTSIZE)
    plt.ylabel("Average time (in sec)", weight = "bold", fontsize = FONTSIZE)
    plt.plot(x, to_print["time_passed"], "-o", linewidth=5)
    plt.show()

if __name__ == "__main__":
    main()