# %%

import numpy as np
from SSN_model_funcs import *  # tentatively...
import sys
import pickle
import model_data

types = ["G", "H"]
homedir = "/home/shinya.ito/realistic-model/visual_behavior_modeling/"

dpath = homedir + "neuropixels_data/l23_mean_traces.pkl"
with open(dpath, "rb") as f:
    l23data = pickle.load(f)

ctypes = ["RS", "FS", "SST", "VIP"]

seed = int(sys.argv[1])
# seed = 1
np.random.seed(seed)

solutions = {"G": [], "H": []}

penalties = np.logspace(-1, 4, 6)

for type in types:
    data_all = [l23data[f"{ctype}_{type}"] for ctype in ctypes]
    data = np.array([d["mean"] for d in data_all])
    data_err = np.array([d["std"] for d in data_all])

    data_reformatted = data[:, 0 : (default_end - default_start)]
    data_reformatted = np.transpose(data_reformatted)
    data_err_reformatted = data_err[:, 0 : (default_end - default_start)]
    data_err_reformatted = np.transpose(data_err_reformatted)
    pre_stim = load_predefined_stimulus(type)
    for p in penalties:
        for t in range(10):
            print(p, t)
            m = form_minuit(
                data_reformatted,
                data_y_err=data_err_reformatted,
                penalty_coef=p,
                predefined_stim=pre_stim,
            )
            m = randomize_fit(m)
            m.migrad(ncall=100_000)
            solutions[type].append((m.values.to_dict(), m.fval))

store_file_name = homedir + f"ssn_model/fit_result4/result_{seed:04d}.pkl"

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

