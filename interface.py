# Importing CSP solver
from CSP.python.multi_model_runner import specific_runner as CSP_solve
from CSP.python.multi_model_runner import MODELS_TO_EXP as CSP_models, model_to_name_exp as CSP_names

# Importing SAT solver
# TODO ...

# Importing SMT solver
from SMT.run_instances import run_instance as SMT_solve
from SMT.run_instances import MODELS as SMT_models, NAMES as SMT_names

# Importing MIP solver
from MIP.run_instances import run_instance as MIP_solve
from MIP.run_instances import MODELS as MIP_params, NAMES as MIP_names

# Importing instances getter
from utils.converter import get_file as get_instances

MIN_INSTANCE = 1
MAX_INSTANCE = 21


# The main function of the project.
# It shows the possibilities for each solver and allow to run an instance on them.
def run_interface():
    print('''\
███████╗ █████╗ ████████╗███╗   ███╗ █████╗ ███╗   ██╗       ██╗       ██████╗  ██████╗ ███╗   ███╗██╗██████╗ ███████╗
██╔════╝██╔══██╗╚══██╔══╝████╗ ████║██╔══██╗████╗  ██║       ██║       ██╔══██╗██╔═══██╗████╗ ████║██║██╔══██╗██╔════╝
███████╗███████║   ██║   ██╔████╔██║███████║██╔██╗ ██║    ████████╗    ██████╔╝██║   ██║██╔████╔██║██║██████╔╝███████╗
╚════██║██╔══██║   ██║   ██║╚██╔╝██║██╔══██║██║╚██╗██║    ██╔═██╔═╝    ██╔══██╗██║   ██║██║╚██╔╝██║██║██╔═══╝ ╚════██║
███████║██║  ██║   ██║   ██║ ╚═╝ ██║██║  ██║██║ ╚████║    ██████║      ██║  ██║╚██████╔╝██║ ╚═╝ ██║██║██║     ███████║
╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝    ╚═════╝      ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝     ╚══════╝
                                                                                                                      ''')
    csp_poss = [(CSP_names[CSP_models[i]], CSP_models[i], CSP_solve) for i in range(len(CSP_models))]
    # sat_poss = ...
    smt_poss = [(SMT_names[i], SMT_models[i], SMT_solve) for i in range(len(SMT_models))]
    mip_poss = [(MIP_names[MIP_params[i]], MIP_params[i], MIP_solve) for i in range(len(MIP_params))]

    all_poss = {
        '0': ("CSP", csp_poss),
        # '1': sat_poss,
        '2': ("SMT", smt_poss),
        '3': ("MIP", mip_poss)
    }

    print("--- Choose the solving approach ---\n")
    for key, value in all_poss.items():
        print(f"Press {key} for: {value[0]}")
        print("~~~~~~~~\n")
    approach = input()
    poss = all_poss[approach]

    print(f"\n--- You have chosen {poss[0]} ---\n")
    print(f"--- Choose the solving strategy ---")

    poss = poss[1]

    for i in range(len(poss)):
        strat_name, strat, _ = poss[i]
        print(f"Press {i} for: {strat_name}")
        print("~~~~~~~~\n")

    strategy = int(input())
    model = poss[strategy][1]

    print(f"\n--- Choose the instance ---")
    print(f"\n~~ A number between {MIN_INSTANCE} and {MAX_INSTANCE} ~~")

    inst = int(input())

    fun = poss[strategy][2]

    print(f"\n ~~ Running {poss[strategy][0]} on instance {inst} ~~ \n")

    fun(inst, get_instances(), model)


if __name__ == '__main__':
    run_interface()
