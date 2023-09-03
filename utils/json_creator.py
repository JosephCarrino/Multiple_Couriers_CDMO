import json
import pathlib as pl
from typing import Callable

from utils.mulitprocessing_execution import run_instances_multiprocessing
from utils import get_instances

from SAT.SAT_model import solve_one as sat_solver
from MIP.MIP_model import solve_one as mip_solver
from SMT.STM_fixed import solve_one as smt_correct_solver
from SMT.SMT_naive import solve_one as smt_naive_solver
from CSP.CSP_model import solve_one as csp_solver

from functools import partial

import matplotlib.pyplot as plt
import numpy as np

AVAILABLE_MODELS = ["SAT", "NaiveSMT", "FixedSMT", "BalancedMIP", "FeasibilityMIP", "OptimalityMIP",
                    "CSP", "DomDegMinCSP", "DomDegRandCSP", "CSPnoSB"]


def get_model_from_name(model_name: str) -> Callable[[dict, dict], dict]:
    """
    Get the model function from the model name
    """

    if model_name == "SAT":
        return sat_solver
    elif model_name == "NaiveSMT":
        return smt_naive_solver
    elif model_name == "FixedSMT":
        return smt_correct_solver
    elif model_name == "BalancedMIP":
        return partial(mip_solver, emph=0, timeout=300)
    elif model_name == "FeasibilityMIP":
        return partial(mip_solver, emph=1, timeout=300)
    elif model_name == "OptimalityMIP":
        return partial(mip_solver, emph=2, timeout=300)
    elif model_name == "CSP":
        return partial(csp_solver,
                       model_path=pl.Path("CSP/model.mzn"),
                       instance_folder_path=pl.Path("CSP/instances"),
                       solver="Gecode",
                       timeout=300 * 1000,
                       )
    elif model_name == "DomDegMinCSP":
        return partial(csp_solver,
                       model_path=pl.Path("CSP/dom_deg_min.mzn"),
                       instance_folder_path=pl.Path("CSP/instances"),
                       solver="Gecode",
                       timeout=300 * 1000,
                       )
    elif model_name == "DomDegRandCSP":
        return partial(csp_solver,
                       model_path=pl.Path("CSP/dom_deg_rand.mzn"),
                       instance_folder_path=pl.Path("CSP/instances"),
                       solver="Gecode",
                       timeout=300 * 1000,
                       )
    elif model_name == "CSPnoSB":
        return partial(csp_solver,
                       model_path=pl.Path("CSP/model.mzn"),
                       instance_folder_path=pl.Path("CSP/instances"),
                       solver="Gecode",
                       symmetry_breaking=False,
                       timeout=300 * 1000,
                       )
    elif model_name == "DomDegMinCSPnoSBnoSB":
        return partial(csp_solver,
                       model_path=pl.Path("CSP/dom_deg_min.mzn"),
                       instance_folder_path=pl.Path("CSP/instances"),
                       solver="Gecode",
                       symmetry_breaking=False,
                       timeout=300 * 1000,
                       )
    elif model_name == "DomDegRandCSPnoSB":
        return partial(csp_solver,
                       model_path=pl.Path("CSP/dom_deg_rand.mzn"),
                       instance_folder_path=pl.Path("CSP/instances"),
                       solver="Gecode",
                       symmetry_breaking=False,
                       timeout=300 * 1000,
                       )
    else:
        raise ValueError(f"Model {model_name} not found")


def get_out_folder_for_model(model_name: str, result_folder: pl.Path) -> pl.Path:
    """
    Get the folder where to save the results for the given model name
    """

    if model_name == "SAT":
        return result_folder / "SAT"
    elif model_name in ["NaiveSMT", "FixedSMT"]:
        return result_folder / "SMT"
    elif model_name in ["BalancedMIP", "FeasibilityMIP", "OptimalityMIP"]:
        return result_folder / "MIP"
    elif model_name in ["CSP", "DomDegMinCSP", "DomDegRandCSP", "CSPnoSB", "DomDegMinCSPnoSB", "DomDegRandCSPnoSB"]:
        return result_folder / "CSP"
    else:
        raise ValueError(f"Model {model_name} not found")


def get_process_timeout(model_name: str) -> int:
    """
    For all model except MIP model we will have a timeout of 300 seconds
    :param model_name: name of the model
    :return:
    """

    # These models have already a timeout in their optimizer and we don't want to have a timeout in the process
    if model_name in ["BalancedMIP", "FeasibilityMIP", "OptimalityMIP", "CSP", "DomDegMinCSP", "DomDegRandCSP", "CSPnoSB", "DomDegMinCSPnoSB", "DomDegRandCSPnoSB"]:
        return 600
    else:
        return 300


def save_results(results: list[dict], instances_number: list[int], results_folder: pl.Path, model_name: str):
    """
    Save the results in a json file
    :param results: list of results
    :param instances_number: list of instances number
    :param results_folder: folder where to save the results
    :param model_name: name of the model
    :return:
    """

    print(f"Saving results in {results_folder}")
    results_folder.mkdir(parents=True, exist_ok=True)

    for instance_id, result in zip(instances_number, results):
        instance_result_file = results_folder / f"{instance_id}.json"

        if instance_result_file.exists():
            # Load previous results to append or update them
            with open(instance_result_file, "r") as f:
                try:
                    json_dict = json.load(f)
                except json.decoder.JSONDecodeError:
                    json_dict = {}

                json_dict[model_name] = result
        else:
            json_dict = {model_name: result}

        with open(instance_result_file, "w") as f:
            json.dump(json_dict, f, indent=4)


def save_plot(results: list[dict], instances_number: list[int], plot_folder: pl.Path, model_name: str) -> None:
    """
    Save the plot of the results
    :param results: list of results
    :param instances_number: list of instances number
    :param plot_folder: folder where to save the plot
    :param model_name: name of the model
    """

    plot_file = plot_folder / f"{model_name}.png"
    print(f"Saving plot in {plot_file}")

    plot_folder.mkdir(parents=True, exist_ok=True)

    x_axes = np.array(instances_number)
    y_axes = np.array([result["time"] for result in results])

    plt.grid()
    plt.title(f"Time for {model_name} for each instance", weight="bold", fontsize=16)
    plt.xlabel("Instance", weight="bold", fontsize=14)
    plt.ylabel("Time (s)", weight="bold", fontsize=14)
    plt.plot(x_axes, y_axes, "-o", color="red", linewidth=5)
    plt.savefig(plot_file)
    plt.show()
    plt.close()


def run_json_creator(
        instances_str: str,
        model_str: str,
        max_process: int,
        result_folder: pl.Path,
        build_plot: bool = True,
        plot_folder: pl.Path = pl.Path("plots")
):
    """
    Run instances in parallel and create a json file with the results for each instance and for each model
    :param instances_str: Instances number to run or the string "all" to run on all available instances
    :param model_str: name of the model or models (separated by a comma) to run or the string "all"
    :param max_process: maximum number of process to run in parallel
    :param result_folder: folder where to save the results
    :param build_plot: if True, build a plot with the results for each model
    :param plot_folder: folder where to save the plot
    :return:
    """

    all_instances = get_instances()

    if instances_str == "all":
        instances_number = list(range(1, len(all_instances) + 1))
    else:
        instances_number = [int(i) for i in instances_str.split(",")]

    if model_str == "all":
        models_name = AVAILABLE_MODELS
    else:
        models_name = model_str.split(",")
        models_name = [model_name.strip() for model_name in models_name]

    for model_name in models_name:
        print(f"Running instances {instances_number} on model {model_name}, max process {max_process}")

        model_str = get_model_from_name(model_name)
        results_folder = get_out_folder_for_model(model_name, result_folder)
        results_folder.mkdir(parents=True, exist_ok=True)

        selected_instances = [all_instances[i - 1] for i in instances_number]
        process_timeout = get_process_timeout(model_name)

        results = run_instances_multiprocessing(selected_instances, instances_number, model_str, max_process=max_process,
                                                timeout=process_timeout)

        print("\n")
        save_results(results, instances_number, results_folder, model_name)

        if build_plot:
            save_plot(results, instances_number, plot_folder, model_name)

        print("\n\n")
