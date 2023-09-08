import datetime
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor as Pool
from typing import Callable
from functools import partial

model_type = Callable[[dict, int, dict], dict]


def run_instances_multiprocessing(
        instances: list[dict],
        instances_index: list[int],
        model: model_type,
        max_process: int = 4,
        model_kwargs: dict = None,
        timeout: int = 300
) -> list[dict]:
    """
    Function that solve a list of instances using a solver
    :param instances: the list of instances to be solved
    :param instances_index: the list of instances index
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

    with Pool(max_workers=process_number) as pool:
        results = list(
            pool.map(partial_run_instance, [(instances[i], instances_index[i], model) for i in range(n_instances)]))

    print(f"---Completed {n_instances} instances")

    return results


def clean_result(model_result: dict) -> dict:
    """
    Clean the result of the model to be saved in the json file
    """

    results = {
        "time": int(model_result["time"]) if model_result["optimal"] else 300,
        # TODO doesn't work if the timeout change
        "optimal": model_result["optimal"],
        "obj": int(model_result["min_dist"]) if model_result["min_dist"] else 0,
        "sol": model_result["sol"] if model_result["sol"] else [],
    }

    if results["sol"] == []:
        results["sol"] = []
    else:
        # Remove element referring to the base node
        results["sol"] = [[elem for elem in courier_elems if elem != courier_elems[0]] for courier_elems in
                          results["sol"]]

    return results


def run_instance(
        *args,
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

    instance, instance_index, model = args[0]

    print(f"Starting instance n°{instance_index} at {datetime.datetime.now().strftime('%H:%M:%S')}")

    # Start a thread that run the model and terminate it after timeout seconds

    model_result_dict = {
        "time": None,
        "min_dist": None,
        "sol": None,
        "optimal": None,
        "iterations": None,
        "ready": False,
    }

    # We use process instead of thread to be able to terminate them after timeout

    model_result = multiprocessing.Manager().dict(model_result_dict)
    process = multiprocessing.Process(target=model, args=(instance, instance_index, model_result), kwargs=model_kwargs)
    process.start()

    # TODO add a safe guard to avoid infinite loop

    time_start = time.time()
    while not model_result["ready"] and time.time() - time_start < 300:
        # print(f"waiting for model to be ready {datetime.datetime.now().strftime('%H:%M:%S')}")
        pass

    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        model_result["optimal"] = False
        print(f"---Completed instance n°{instance_index} at {datetime.datetime.now().strftime('%H:%M:%S')} -- timeout")
    else:
        print(f"---Completed instance n°{instance_index} at {datetime.datetime.now().strftime('%H:%M:%S')}")
    # print(model_result)
    result = clean_result(model_result)

    return result
