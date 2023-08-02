from CSP.python.model_runner import cmd_runner as get_stats
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import json
import os

INSTANCES_DIR = f"{os.path.abspath(__file__)}\\..\\instances"

MODELS: list[str] = ["../model.mzn",
                     "../dom_deg_min.mzn",
                     "../dom_deg_rand.mzn"]

MODELS_TO_EXP: list[str] = ["CSP/model.mzn",
                            "CSP/dom_deg_min.mzn",
                            "CSP/dom_deg_rand.mzn"
                            ]

model_to_name: dict[str, str] = {MODELS[0]: "Default Model",
                                 MODELS[1]: "Dom_w_deg, Indomain_min",
                                 MODELS[2]: "Dom_w_deg, Indomain_rand"}

model_to_name_exp: dict[str, str] = {MODELS_TO_EXP[0]: "Default Model",
                                     MODELS_TO_EXP[1]: "Dom_w_deg, Indomain_min",
                                     MODELS_TO_EXP[2]: "Dom_w_deg, Indomain_rand"}

FONTSIZE: int = 14
plt.rc("font", weight="bold", size=FONTSIZE)

N_INSTANCES: int = 10


def specific_runner(number: int, instances: list, model: str) -> (list[list[list[str]]], list[int], list[float]):
    """
    Function for running an instance on a specified model
    :param number: Index of the instance to run
    :param instances: List of all instances
    :param model: Model to be used
    :return: Found solutions, minimized distances and times of computation
    """

    # Variable for checking if solution found is the optimal one
    timeouted = False
    to_ret_paths = []
    to_ret_dist = []
    to_ret_times = []
    paths = multiprocessing.Queue()
    z = multiprocessing.Queue()
    time = multiprocessing.Queue()
    real_number = "0" + str(number) if number < 10 else str(number)
    p = multiprocessing.Process(target=get_stats, args=(model, f"{INSTANCES_DIR}\\inst{real_number}.dzn", paths, z, time))
    p.start()
    p.join(300)
    if p.is_alive():
        print("TIMEOUT")
        timeouted = True
        p.terminate()
        p.join()
    to_ret_paths.append(paths.get() if not paths.empty() else 0)
    to_ret_dist.append(z.get() if not z.empty() else 0)
    to_ret_times.append(time.get() if not time.empty() and not timeouted else 300)
    return to_ret_paths, to_ret_dist, to_ret_times


def get_results(number: int) -> list[int]:
    """
    Getting results on an instance using all CSP models available
    :param number: Instance index
    :return times: Time of computation of each model
    """
    to_out = {}
    times = []
    for model in MODELS:
        paths, z, time = specific_runner(number, [], model)
        optimal = True
        if time[0] == 300:
            optimal = False
        to_out[model_to_name[model]] = {
            "time": int(time[0]),
            "optimal": optimal,
            "obj": z[0],
            "sol": [[int(elem) for elem in courier if elem != "s"]
                    for courier in paths[0]]
            if paths[0] != "Unsat" and paths[0] != 0 else paths[0]
        }
        times.append(int(time[0]))
    with open(f"../../res/CSP/{number}.json", "w+") as f:
        json.dump(to_out, f, indent=4)

    return times


def get_all_results():
    """
    Getting solutions for all instances
    """
    times = [[] for _ in range(len(MODELS))]
    for i in range(1, N_INSTANCES + 1):
        tmp_times = get_results(i)
        for j in range(len(MODELS)):
            times[j].append(tmp_times[j])

    # Plotting of times of computation
    x = np.array([i for i in range(1, N_INSTANCES + 1)])
    plt.grid()
    plt.title("Time passed computing each instance", weight="bold", fontsize=FONTSIZE)
    plt.xlabel("Instance n°", weight="bold", fontsize=FONTSIZE)
    plt.ylabel("Average time (in sec)", weight="bold", fontsize=FONTSIZE)
    for i in range(len(MODELS)):
        plt.plot(x, times[i], "-o", label=model_to_name[MODELS[i]], linewidth=5)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    get_all_results()
