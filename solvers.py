import argparse
import pathlib as pl
from utils.json_creator import run_json_creator, AVAILABLE_MODELS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run instances in parallel and create a json file with the results for each instance')
    parser.add_argument('--instances', type=str,
                        help='Instances number to run or the string "all" to run on all available instances')
    parser.add_argument('--model', type=str,
                        help='Name of the model or models (separated by a comma) to run or the string "all". Possible models are: ' + ", ".join(AVAILABLE_MODELS),
                        default=AVAILABLE_MODELS[0])
    parser.add_argument('--max_process', type=int, default=4, help='Maximum number of process to run in parallel')
    parser.add_argument('--result_folder', type=pl.Path, default=pl.Path("res"),
                        help='Folder where to save the results')

    parser.add_argument('--build_plot', action="store_true", help='If True, build a plot with the results for each model')
    parser.add_argument('--plot_folder', type=pl.Path, default=pl.Path("plots"), help='Folder where to save the plot')

    parser.add_help = True

    args = parser.parse_args()
    run_json_creator(args.instances, args.model, args.max_process, args.result_folder, args.build_plot, args.plot_folder)



