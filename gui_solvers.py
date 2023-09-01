
from solvers import run_json_creator, AVAILABLE_MODELS
import pathlib as pl


# The main function of the project.
# It shows the possibilities for each solver and allow to run an instance on them.
def run_interface():
    print('''\
███████╗ █████╗ ████████╗███╗   ███╗ █████╗ ███╗   ██╗       ██╗       ██████╗  ██████╗ ███╗   ███╗██╗██████╗ ███████╗
██╔════╝██╔══██╗╚══██╔══╝████╗ ████║██╔══██╗████╗  ██║       ██║       ██╔══██╗██╔═══██╗████╗ ████║██║██╔══██╗██╔════╝
███████╗███████║   ██║   ██╔████╔██║███████║██╔██╗ ██║    ████████╗    ██████╔╝██║   ██║██╔████╔██║██║██████╔╝███████╗
╚════██║██╔══██║   ██║   ██║╚██╔╝██║██╔══██║██║╚██╗██║    ██╔═██╔═╝    ██╔══██╗██║   ██║██║╚██╔╝██║██║██╔═══╝ ╚════██║
███████║██║  ██║   ██║   ██║ ╚═╝ ██║██║  ██║██║ ╚████║    ██████║      ██║  ██║╚██████╔╝██║ ╚═╝ ██║██║██║     ███████║
╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝    ╚═════╝      ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝     ╚══════╝
                                                                                                                      ''')

    print("--- Choose the solving approach, either a number, a list of comma separated number or 'all' ---\n")
    for i, model_name in enumerate(AVAILABLE_MODELS):
        print(f"Press {i} for: {model_name}")

    print("\n")
    model_indexes = input()

    if model_indexes == "all":
        model_name_str = "all"
    else:
        model_indexes = [int(index) for index in model_indexes.split(",")]
        model = [AVAILABLE_MODELS[index] for index in model_indexes]
        model_name_str = ",".join(model)


    print(f"--- Choose the instances ---")
    print(f"\n~~ A number, a comma separated list or 'all'~~")

    instances_str = input()

    print(f"\n--- Choose the maximum number of process ---")
    max_process = int(input())

    print(f"\n--- Choose 1 if need output plots or 0 if not ---")
    output_plots = int(input()) == 1

    result_folder = pl.Path("resultsInterface")
    plot_folder = pl.Path("plots")

    print(model_name_str)
    print(instances_str)
    print(max_process)
    print(result_folder)

    run_json_creator(instances_str, model_name_str, max_process, result_folder, output_plots, plot_folder)


if __name__ == '__main__':
    run_interface()
