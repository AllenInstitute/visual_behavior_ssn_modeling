# %% This will do perturbation of the target data.
# solve for 6 configurations for G + image response and H + non-image response.
# the penalty value is fixed to 10.

import numpy as np
# from SSN_model_funcs import *  # tentatively...
import SSN_model_funcs as ssn
import create_new_target_data as cnd
import sys
import pickle

# types = ["G", "H", "F"] # G + image, H + non-image, Full G-H interpolation
types = ["H"] # G + image, H + non-image, Full G-H interpolation
homedir = "/home/shinya.ito/realistic-model/visual_behavior_modeling/"


ctypes = ["RS", "FS", "SST", "VIP"]

seed = int(sys.argv[1])
# seed = 1
np.random.seed(seed)

# solutions = {"G": [], "H": [], "F": []}
solutions = {"H": []}

# penalties = np.logspace(-1, 4, 6)
# penalties = [10]
fracs = np.linspace(-0.5, 1.5, 9)

for type in types:
    data, stim = cnd.prepare_interp_data_for_fit(fracs, type)
    for i, frac in enumerate(fracs):
        for t in range(10):
            print(f"Working on frac:{frac}, t:{t}")
            m = ssn.form_minuit(
                data[i]['data'],
                data_y_err=data[i]['err'],
                penalty_coef=10.0,
                predefined_stim=stim[i],
            )
            m = ssn.randomize_fit(m)
            m.migrad(ncall=100_000)
            print(f"fval: {m.fval}")
            solutions[type].append((m.values.to_dict(), m.fval))

store_file_name = homedir + f"ssn_model/fit_result5/result_{seed:04d}.pkl"

with open(store_file_name, "wb") as f:
    pickle.dump(solutions, f)


# %% plotting the results
# data_array = np.concatenate((np.zeros_like(data_reformatted), data_reformatted))

# r = sim_to_result_orig([], **m.values.to_dict())
# plot_results(r["output"], data_array, r["stims"], time_index)

# r = sim_to_result_orig([], **m.values.to_dict(), double_omission=True)
# plot_results(r["output"], data_array, r["stims"], time_index)


# # %% evaluate a few things about the new results

# # fraction between the main loss vs constructed loss
# infl_names = [
#     ["e_to_e", "p_to_e", "s_to_e", "v_to_e"],
#     ["e_to_p", "p_to_p", "s_to_p", "v_to_p"],
#     ["e_to_s", "p_to_s", "s_to_s", "v_to_s"],
#     ["e_to_v", "p_to_v", "s_to_v", "v_to_v"],
# ]


# # infl_flat = [item for sublist in infl_names for item in sublist]
# vdict = m.values.to_dict()
# infl_div = np.array([[vdict[i] for i in j] for j in infl_names])
# influence_matrix_fracerr

# np.sum(((infl_div - 1) / influence_matrix_fracerr) ** 2)

