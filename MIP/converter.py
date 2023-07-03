import os
import re
import numpy as np

def get_file():
    instances = []
    for instance in os.listdir("./instances"):
        if not instance.endswith(".dat"):
            continue
        with open(f"./instances/{instance}") as f:
            text = f.read()
        lines = text.split("\n")
        # regex to get a number from a string
        num_regex = r"(\d+)"
        lines = [x for x in lines if x != ""]
        lines = [x.strip() for x in lines]
        m = int(re.findall(num_regex, lines[0])[0])
        n = int(re.findall(num_regex, lines[1])[0])
        time = n-m+2
        l = [int(x) for x in re.findall(num_regex, lines[2])]
        s = [int(x) for x in re.findall(num_regex, lines[3])]
        distances = []
        for i in range(4, 4 + n + 1):
            distances.append([int(x) for x in re.findall(num_regex, lines[i])])
        
        distances = np.array(distances)

        text = ''
        l = [int(x) for x in l]
        s = [int(x) for x in s]

        to_ret = {"m": m,
                  "n": n,
                  "D": distances.tolist(),
                  "l": l,
                  "s": s,}
        instances.append(to_ret)
    return instances
        