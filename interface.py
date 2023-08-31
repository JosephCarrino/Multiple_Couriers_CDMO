# Importing CSP solver
from CSP.python.multi_model_runner import specific_runner as CSP_solve
from CSP.python.multi_model_runner import MODELS_TO_EXP as CSP_models, model_to_name_exp as CSP_names

# Importing SAT solver
from SAT.run_instances import run_instance as SAT_solve
from SAT.run_instances import MODELS as SAT_models, NAMES as SAT_names

# Importing SMT solver
from SMT.run_instances import run_instance as SMT_solve
from SMT.run_instances import MODELS as SMT_models, NAMES as SMT_names

# Importing MIP solver
from MIP.run_instances import run_instance as MIP_solve
from MIP.run_instances import MODELS as MIP_params, NAMES as MIP_names

# Importing instances getter
from utils.converter import get_instances as get_instances

MIN_INSTANCE = 1
MAX_INSTANCE = 21


def convert_result_to_json(result):
    out = {}

    out["time"] = int(result["time_passed"][0])
    out["optimal"] = out["time"] < 300
    out["obj"] = result["min_dist"][0]
    out["sol"] = [[int(elem) for elem in courier if int(elem) != int(courier[0])] for courier in
                  result["sol"][0]] if result["sol"][0] != "Unsat" else result["sol"][0]


    return out

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
    sat_poss = [(SAT_names[i], SAT_models[i], SAT_solve) for i in range(len(SAT_models))]
    smt_poss = [(SMT_names[i], SMT_models[i], SMT_solve) for i in range(len(SMT_models))]
    mip_poss = [(MIP_names[MIP_params[i]], MIP_params[i], MIP_solve) for i in range(len(MIP_params))]

    all_poss = {
        '0': ("CSP", csp_poss),
        '1': ("SAT", sat_poss),
        '2': ("SMT", smt_poss),
        '3': ("MIP", mip_poss)
    }

    print("--- Choose the solving approach ---\n")
    for key, value in all_poss.items():
        print(f"Press {key} for: {value[0]}")
        print("~~~~~~~~\n")
    approach = input()
    poss = all_poss[approach]

    method_name = poss[0]

    print(f"\n--- You have chosen {method_name} ---\n")
    print(f"--- Choose the solving strategy ---")

    poss = poss[1]

    for i in range(len(poss)):
        strat_name, strat, _ = poss[i]
        print(f"Press {i} for: {strat_name}")
        print("~~~~~~~~\n")

    strategy_number = int(input())

    strategy_name, model = poss[strategy_number][0], poss[strategy_number][1]

    print(f"\n--- Choose the instance ---")
    print(f"\n~~ A number between {MIN_INSTANCE} and {MAX_INSTANCE} ~~")

    inst = int(input())

    print(f"\n--- You have chosen instance {inst} ---")

    fun = poss[strategy_number][2]

    print(f"\n ~~ Running {poss[strategy_number][0]} on instance {inst} ~~ \n")


    result = fun(inst, get_instances(), model)
    json_data = convert_result_to_json(result)

    print(f"\n ~~ Result ~~ \n")
    print(json_data)


if __name__ == '__main__':
    run_interface()
