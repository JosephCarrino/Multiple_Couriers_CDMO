import pathlib as pl
import os
import json


def minizinc_solution_parser(solution_json: dict, model_result: dict = None) -> dict:
    """
    Parse a solution from a JSON object
    """

    if model_result is None:
        model_result = {}

    model_result["iterations"] = None
    model_result["time"] = solution_json["time"]
    model_result["optimal"] = None
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


def minizinc_output_parser(output_strings: list[str], model_result: dict = None) -> dict:
    """
    Parse the output of a minizinc run
    :param output_strings: The list of strings containing the output of the minizinc run
    :return: A dictionary containing the information about the best solution
    """

    data = []
    for output_string in output_strings:
        print(f"{output_string=}")
        try:
            data.append(json.loads(output_string))
        except json.JSONDecodeError:
            pass

    print(f"{data=}")

    # There are more then one solution since we are using the --intermediate flag
    solutions = [d for d in data if d["type"] == "solution"]

    # The best is the last one
    best_solution = solutions[-1]

    return minizinc_solution_parser(best_solution, model_result)


def solve_one_new(
        instance: dict,
        instance_index: int,
        model_result: dict = None,
        model_path: pl.Path = pl.Path("CSP/model.mzn"),
        instance_folder_path: pl.Path = pl.Path("CSP/instances"),
        solver: str = "Gecode",
        timeout: int = 30_000,
) -> dict:
    instance_index_str = str(instance_index + 1).zfill(2)
    instance_path = instance_folder_path / f"inst{instance_index_str}.dzn"

    # Run the model
    minizinc_cmd_args = [
        f"--solver {solver}",
        f"{model_path}",
        f"{instance_path}",
        f"-p {1}",
        f"--time-limit {timeout}",
        "--json-stream",
        "--output-time",
        "--intermediate",
    ]

    cmd = "minizinc " + " ".join(minizinc_cmd_args)

    print(f"Running cmd {cmd}")
    output = os.popen(cmd).readlines()

    # Parse the output
    return minizinc_output_parser(output, model_result)
