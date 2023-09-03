import json
import pathlib as pl
import numpy as np
import matplotlib.pyplot as plt

results_dir = pl.Path('../res')

def make_graph(model):

    data_dir = results_dir / model

    # Get the data
    data = {}

    # load all json file
    for file in data_dir.glob('*.json'):
        with open(file) as f:
            data[file.stem] = json.load(f)

    instance_number = len(data)

    # extract the time
    data_model = {}

    for instance_id, instance_data in data.items():

        for model_name, model_data in instance_data.items():
            if model_name not in data_model:
                data_model[model_name] = [_ for _ in range(instance_number)]

            data_model[model_name][int(instance_id) - 1] = model_data['time']

    label = list(data_model.keys())
    times = np.array(list(data_model.values()))

    fontsize = 20
    # Plot the data
    plt.figure(figsize=(20, 10))
    plt.grid()
    plt.title("Time passed computing each instance", weight="bold", fontsize=fontsize)
    plt.xlabel("Instance n°", weight="bold", fontsize=fontsize)
    plt.ylabel("Average time (in sec)", weight="bold", fontsize=fontsize)

    # Make x axis ticks

    plt.xticks(np.arange(0, times.shape[1], 1), labels=np.arange(1, times.shape[1] + 1, 1),  fontsize=fontsize)
    plt.yticks(fontsize=fontsize)


    # Change marker style for each line

    markers = ['o', 'v', 's', 'p', 'P', '*', 'h', 'H', 'D', 'd', 'X', 'x', '+', '|', '_']
    for i, model_name in enumerate(label):
        # Change marker size
        # plt.plot(times[i], label=model_name, marker=markers[i])
        plt.plot(times[i], label=model_name, marker=markers[i], markersize=12, linewidth=3)

    plt.legend(fontsize=fontsize, loc='upper right')

    # save the graph to file
    plt.savefig(f"{model}_graph.png", dpi=500)


if __name__ == '__main__':
    make_graph("MIP")