
from data_generator import multi_mixed_generator as get_instances
from model_runner import cmd_runner as get_stats
import matplotlib.pyplot as plt
import os
import numpy as np
import multiprocessing
import json

ORDERS = 5
INSTANCES_PER_ORDER = 2
MODELS = ["../model.mzn", 
          "../dom_deg_min.mzn", 
          "../dom_deg_rand.mzn"]

model_to_name = {MODELS[0]: "Default Model", 
                 MODELS[1]: "Dom_w_deg, Indomain_min",
                 MODELS[2]: "Dom_w_deg, Indomain_rand"}

FONTSIZE  = 14
plt.rc("font", weight = "bold", size= FONTSIZE)

N_INSTANCES = 21

def main():
    # get_instances(ORDERS, INSTANCES_PER_ORDER)
    to_save = {MODELS[0]: {}, MODELS[1]: {}, MODELS[2]: {}}
    plt.grid()
    plt.title("Comparison of different heuristics on increasing instance complexity", weight = "bold", fontsize = FONTSIZE)
    plt.xlabel("Instance n°", weight = "bold", fontsize = FONTSIZE)
    plt.ylabel("Average time (in sec)", weight = "bold", fontsize = FONTSIZE)
    for model in MODELS:
        paths, dists, times = mixed_runner(model)
        to_save[model]["paths"] = paths
        to_save[model]["dists"] = dists
        to_save[model]["times"] = times
        x = np.array([i for i in range(1, N_INSTANCES+1)])
        plt.plot(x, times, "-o", linewidth=5, label = model_to_name[model])
    with open("multi_model_data.json", "w+") as f:
        json.dump(to_save, f, indent=4)
    plt.legend()
    plt.show()

def mixed_runner(model, orders, instances_per_order):
    to_ret_paths = []
    to_ret_dist = []
    to_ret_times = []
    for i in range(orders):
        by_order_paths = []
        by_order_dist = []
        by_order_times = []
        for j in range(instances_per_order):
            print(f"instances/data{i}_{j}.dzn")
            paths, z, time = get_stats(model = model, data_path = f"instances/data{i}_{j}.dzn")
            by_order_paths.append(paths)
            by_order_dist.append(z)
            by_order_times.append(time)
        to_ret_paths.append(by_order_paths)
        to_ret_dist.append(by_order_dist)
        to_ret_times.append(by_order_times)
    return to_ret_paths, to_ret_dist, to_ret_times

def mixed_runner(model):
    to_ret_paths = []
    to_ret_dist = []
    to_ret_times = []
    for file in os.listdir("instances"):
        number = file.split("inst")[1].split(".")[0]
        if (int)(number) > N_INSTANCES:
            continue
        # I need to get output times from the process
        paths = multiprocessing.Queue()
        z = multiprocessing.Queue()
        time = multiprocessing.Queue()
        p = multiprocessing.Process(target=get_stats, args=(model, f"instances/{file}", paths, z, time))
        p.start()
        p.join(300)
        if p.is_alive():
            print("TIMEOUT")
            p.terminate()
            p.join()
            continue
        to_ret_paths.append(paths.get() if not paths.empty() else 0)
        to_ret_dist.append(z.get() if not z.empty() != 0 else 0)
        to_ret_times.append(time.get() if not time.empty() != 0 else 300)
    return to_ret_paths, to_ret_dist, to_ret_times

if __name__ == '__main__':
    main()