import random

'''
    Written this generator for testing CSP before true instances were available
'''

MIN_COURIERS = 2
COURIERS_BOUND = 3
MIN_PACKAGES = 3
PACKAGES_BOUND = 5
MIN_DISTANCE = 1
DISTANCE_BOUND = 4
MIN_WEIGHT = 1
WEIGHT_BOUND = 3
COURIER_FACTOR = 2

INSTANCES = 10

def main():
    multi_generator()


def multi_generator(instances: int = INSTANCES, increasingly: bool = False):
    """
    Generate a number of instances
    :param instances: How many instances to generate
    :param increasingly: True if each instance must be more complex than the previous one
    """
    for i in range(instances):
        if increasingly:
            generator(i, i+1, f"instances/data{i}.dzn")
        else:
            generator(i, 1)

def multi_mixed_generator(orders: int = 5, instances_per_order: int = INSTANCES):
    """
    Generate different instances for with increasing complexity
    :param orders: Number of complexity batches of instances
    :param instances_per_order: Number of instances to be generated with each complexity order
    """
    for i in range(orders):
        for j in range(instances_per_order):
            generator(i+1, f"instances/data{i}_{j}.dzn")


def generator(factor: int, out_name: str):
    """
    Function for generating an example instance
    :param factor: the complexity facto
    :param out_name: the name of the instance file
    """
    random.seed()
    sum_pack_weights = float("inf")
    sum_courier_weights = 0
    while sum_pack_weights > sum_courier_weights:
        n_couriers = random.randint(MIN_COURIERS+factor, COURIERS_BOUND+factor)
        n_packages = random.randint(MIN_PACKAGES+factor, PACKAGES_BOUND+factor)
        packages_weights = [random.randint(MIN_WEIGHT*factor, WEIGHT_BOUND*factor) for _ in range(n_packages)]
        couriers_weights = [random.randint(MIN_WEIGHT*COURIER_FACTOR*factor, WEIGHT_BOUND*COURIER_FACTOR*factor) for _ in range(n_couriers)]
        sum_pack_weights = sum(packages_weights)
        sum_courier_weights = sum(couriers_weights)

    is_symmetric = random.randint(0, 1)
    if is_symmetric:
        # a symmetric matrix of distances
        distances = [[random.randint(MIN_DISTANCE*factor, DISTANCE_BOUND*factor) for _ in range(n_packages+1)] for _ in range(n_packages+1)]
        for i in range(n_packages):
            for j in range(i):
                distances[i][j] = distances[j][i]
    else:
        # a non-symmetric matrix of distances
        distances = [[random.randint(MIN_DISTANCE*factor, DISTANCE_BOUND*factor) for _ in range(n_packages+1)] for _ in range(n_packages+1)]
    
    # write to file
    with open(out_name, "w") as f:
        f.write(f"m = {n_couriers}; % number of couriers\n")
        f.write(f"n = {n_packages}; % number of packages\n\n")
        f.write(f"l = {couriers_weights}; % maximum carriable weight per courier\n")
        f.write(f"s = {packages_weights}; % weight of each package\n\n")
        f.write(f"D = % distance matrix \n")
        f.write("\t [")
        for line in distances:
            f.write("| ")
            for j in range(len(line)):
                f.write(f"{line[j]}")
                if j < len(line)-1:
                    f.write(", ")
                else:
                    f.write("\n\t ")
        f.write("|];")
         

if __name__ == '__main__':
    main()