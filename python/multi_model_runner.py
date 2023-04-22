
from data_generator import multi_mixed_generator as get_instances
from model_runner import cmd_runner as get_stats
import matplotlib.pyplot as plt


ORDERS = 3
INSTANCES_PER_ORDER = 5
MODELS = ["../model.mzn"]

model_to_name = {MODELS[0]: "Default Model"}

FONTSIZE  = 14
plt.rc("font", weight = "bold", size= FONTSIZE)

def main():
    get_instances(ORDERS, INSTANCES_PER_ORDER)
    plt.grid()
    plt.title("Comparison of different heuristics on increasing instance complexity", weight = "bold", fontsize = FONTSIZE)
    plt.xlabel("Complexity factor", weight = "bold", fontsize = FONTSIZE)
    plt.ylabel("Average time", weight = "bold", fontsize = FONTSIZE)
    for model in MODELS:
        paths, dists, times = mixed_runner(model, ORDERS, INSTANCES_PER_ORDER)
        avg_times = []
        for i in range(len(times)):
            avg_times.append(sum(times[i]) / len(times[i]))
        plt.plot(range(ORDERS), avg_times, "-o", linewidth=5, label = model_to_name[model])
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

if __name__ == '__main__':
    main()