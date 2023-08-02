from CSP.python.data_generator import multi_generator as get_instances
from CSP.python.model_runner import cmd_runner as get_stats
import matplotlib.pyplot as plt

'''
    This file was used to create charts for the report using generated instances
'''

INSTANCES = 5
FONTSIZE = 14
plt.rc('font', size=FONTSIZE, weight="bold")


def main():
    get_instances(INSTANCES, increasingly=True)
    paths, dists, times = multi_runner(INSTANCES)
    print(dists)
    two_plots(range(INSTANCES), dists, times, "Complexity factor", "Min distance", "Time")


def two_plots(x, y1, y2, xlabel, y1label, y2label):
    fig, ax1 = plt.subplots()
    ax1.grid()
    fig.suptitle("Comparison of time and computed distance increasing instance complexity", weight="bold",
                 fontsize=FONTSIZE)
    color = 'tab:red'
    ax1.set_xlabel(xlabel, weight="bold", fontsize=FONTSIZE)
    ax1.set_ylabel(y1label, color=color, weight="bold", fontsize=FONTSIZE)
    ax1.plot(x, y1, "-o", color=color, linewidth=5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(y2label, color=color, weight="bold", fontsize=FONTSIZE)  # we already handled the x-label with ax1
    ax2.plot(x, y2, "-o", color=color, linewidth=5)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def multi_runner(instances):
    to_ret_paths = []
    to_ret_dist = []
    to_ret_times = []
    for i in range(instances):
        print(f"instances/data{i}.dzn")
        paths, z, time = get_stats(data_path=f"instances/data{i}.dzn")
        to_ret_paths.append(paths)
        to_ret_dist.append(z)
        to_ret_times.append(time)
    return to_ret_paths, to_ret_dist, to_ret_times


if __name__ == '__main__':
    main()
