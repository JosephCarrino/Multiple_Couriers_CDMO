import os
import re
import numpy as np
import pathlib as pl


#INSTANCES_DIR = f"{os.path.abspath(__file__)}\\..\\..\\instances"
INSTANCES_DIR = pl.Path(__file__).parent.parent / "instances"


def get_instances(instances_dir: pl.Path = INSTANCES_DIR) -> list[dict]:
    """
    Function for extracting all instances and getting them as dictionaries
    :return: List of all instances
    """
    instances_count = len([0 for x in os.listdir(instances_dir) if x.endswith(".dat")])


    instances = [{} for _ in range(instances_count)]
    for instance in os.listdir(instances_dir):
        if not instance.endswith(".dat"):
            continue

        instance_number = int(instance[4:6])
        instance_path = pl.Path(instances_dir) / instance

        with open(instance_path, "r") as f:
            text = f.read()


        lines = text.split("\n")
        # regex to get a number from a string
        num_regex = r"(\d+)"
        lines = [x for x in lines if x != ""]
        lines = [x.strip() for x in lines]
        m = int(re.findall(num_regex, lines[0])[0])
        n = int(re.findall(num_regex, lines[1])[0])
        l = [int(x) for x in re.findall(num_regex, lines[2])]
        s = [int(x) for x in re.findall(num_regex, lines[3])]
        distances = []
        for i in range(4, 4 + n + 1):
            distances.append([int(x) for x in re.findall(num_regex, lines[i])])

        distances = np.array(distances)
        l = [int(x) for x in l]
        s = [int(x) for x in s]

        problem_data = {"m": m,
                  "n": n,
                  "D": distances.tolist(),
                  "l": l,
                  "s": s, }

        instances[instance_number - 1] = problem_data
    return instances
