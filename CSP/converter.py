import os
import re
import numpy as np


def main():
    for instance in os.listdir("./instances"):
        if not instance.endswith(".dat"):
            continue
        print(f"Instance: {instance}")
        with open(f"./instances/{instance}") as f:
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

        text = ''
        l = [str(x) for x in l]
        s = [str(x) for x in s]

        text += f'm = {m};\n'
        text += f'n = {n};\n'
        text += f'l = [{", ".join(l)}];\n'
        text += f's = [{", ".join(s)}];\n'

        text += 'D = [| '
        for i in range(n + 1):
            text += ', '.join([str(x) for x in distances[i]])
            if i != n:
                text += ', \n \t\t     | '
        text += ' |];\n'

        # change instance extension into .dzn
        instance = instance.replace(".dat", ".dzn")

        with open(f"Formatted_Instances/{instance}", "w") as f:
            f.write(text)


if __name__ == '__main__':
    main()
