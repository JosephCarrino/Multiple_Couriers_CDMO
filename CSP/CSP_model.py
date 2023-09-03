import pathlib as pl
import os
import json
import time


def minizinc_solution_parser(solution_json: dict, time_passed: int, model_result: dict = None) -> dict:
    """
    Parse a solution from a JSON object
    """

    if model_result is None:
        model_result = {}

    model_result["iterations"] = None
    # print(f"Before int cast {solution_json['time']=}")
    # model_result["time"] = int(solution_json["time"]) / 1000
    # print(f"{solution_json['time']=}")
    model_result["time"] = time_passed
    print(f"{model_result['time']=}")

    model_result["optimal"] = model_result["time"] < 300  # TODO hardcoded timeoutù
    print(f"{model_result['optimal']=}")

    model_result["min_dist"] = None

    # Get the solution
    splitted = solution_json["output"]["dzn"].splitlines()[1:]
    model_result["min_dist"] = int(splitted[-1].split("=")[-1][1:-1])

    splitted = splitted[:-2]
    solution_matrix = []
    for row in splitted:
        true_row = row.split("|")[-1]
        digits = true_row.split(",")

        courier_path = [int(digit) for digit in digits]
        solution_matrix.append(courier_path)

    model_result["sol"] = solution_matrix

    return model_result


def minizinc_output_parser(
        output_strings: list[str],
        model_result: dict = None,
) -> dict:
    """
    Parse the output of a minizinc run
    :param output_strings: The list of strings containing the output of the minizinc run
    :return: A dictionary containing the information about the best solution
    """

    data = []
    for output_string in output_strings:
        try:
            data.append(json.loads(output_string))
        except json.JSONDecodeError:
            pass

    # There are more then one solution since we are using the --intermediate flag
    solutions = [d for d in data if d["type"] == "solution"]

    if len(solutions) == 0:
        return {"iterations": None, "time": None, "optimal": False, "min_dist": None, "sol": None}

    statuses = [d for d in data if d["type"] == "status"]
    status_optimal = [s for s in statuses if s["status"] == "OPTIMAL_SOLUTION"]

    if len(status_optimal) == 0:
        time_passed = 300
    else:
        time_passed = int(status_optimal[-1]["time"]) / 1000

    # The best is the last one
    best_solution = solutions[-1]

    return minizinc_solution_parser(best_solution, time_passed, model_result)


def solve_one(
        instance: dict,
        instance_index: int,
        model_result: dict = None,
        model_path: pl.Path = pl.Path("CSP/model.mzn"),
        instance_folder_path: pl.Path = pl.Path("CSP/instances"),
        solver: str = "Gecode",
        symmetry_breaking: bool = True,
        timeout: int = 300 * 1000,
) -> dict:
    instance_index_str = str(instance_index).zfill(2)
    instance_path = instance_folder_path / f"inst{instance_index_str}.dzn"

    # Run the model
    minizinc_cmd_args = [
        f"--solver {solver}",
        f"{model_path}",
        f"{instance_path}",
        f"-p {1}",
        f"--solver-time-limit  {timeout}",  # Only for solving
        "--json-stream",
        "--output-time",
        "--intermediate",
        f"{'-D mzn_ignore_symmetry_breaking_constraints=true' if not symmetry_breaking else ''}"
    ]

    cmd = "minizinc " + " ".join(minizinc_cmd_args)

    model_result["ready"] = True
    output = os.popen(cmd).readlines()

    # Parse the output
    return minizinc_output_parser(output,
                                  model_result)
