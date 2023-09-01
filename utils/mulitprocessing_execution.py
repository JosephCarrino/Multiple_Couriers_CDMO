import multiprocessing
import threading
from typing import Callable
from functools import partial

model_type = Callable[[dict, int, dict], dict]


def run_instances_multiprocessing(
        instances: list[dict],
        model: model_type,
        max_process: int = 4,
        model_kwargs: dict = None,
        timeout: int = 300
) -> list[dict]:
    """
    Function that solve a list of instances using a solver
    :param instances: the list of instances to be solved
    :param model: model to be used
    :param max_process: maximum number of process to be used
    :param model_kwargs: model kwargs
    :param timeout: maximum number of seconds to run the model
    :return: list of results of the computation using the model
    """

    if model_kwargs is None:
        model_kwargs = {}

    n_instances = len(instances)
    process_number = min(n_instances, max_process)

    partial_run_instance = partial(run_instance, timeout=timeout, model_kwargs=model_kwargs)

    with multiprocessing.Pool(processes=process_number) as pool:
        results = pool.starmap(partial_run_instance, [(instances[i], i, model) for i in range(n_instances)])

    return results


def clean_result(model_result: dict, timeout: int = 300) -> dict:
    """
    Clean the result of the model to be saved in the json file
    """

    results = {
        "time": int(model_result["time"]) if model_result["optimal"] else 300, # TODO doesn't work if the timeout change
        "optimal": model_result["optimal"],
        "obj": int(model_result["min_dist"]) if model_result["min_dist"] else 0,
        "sol": model_result["sol"] if model_result["sol"] else "Unsat",
    }
    # Remove element referring to the base node
    results["sol"] = [[elem for elem in courier_elems if elem != courier_elems[0]] for courier_elems in results["sol"]]

    return results


def run_instance(
        instance: dict,
        instance_index: int,
        model: model_type,
        timeout: int = 300,
        model_kwargs: dict = None
) -> dict:
    """
    Function that solve a single instance using a solver
    :param instance: the instance to be solved
    :param instance_index: the instance index
    :param model: model to be used
    :param timeout: maximum number of seconds to run the model
    :param model_kwargs: model kwargs
    :return: result of the computation using the model
    """

    print(f"Starting instance n°{instance_index}")

    # Start a thread that run the model and terminate it after timeout seconds

    model_result = {
        "time": None,
        "min_dist": None,
        "sol": None,
        "optimal": None,
        "iterations": None,
    }

    # Start a thread that run the model and terminate it after timeout seconds
    thread = threading.Thread(target=model, args=(instance, instance_index, model_result), kwargs=model_kwargs)
    thread.start()
    thread.join(timeout=timeout)

    result = clean_result(model_result, timeout)

    print(f"Completed instance n°{instance_index}")

    return result
