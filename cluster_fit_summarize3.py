# %% a script for summarizing fit_result3 data

# the data structure is
# solutions['G' or 'H'][0-50] (each 10 of them have different value of penalty)
# the penalty values are 1 to 100 in log space

import numpy as np
import pickle
import pandas as pd
from numba import njit

# penalties = np.logspace(0, 2, 5) this is for v1
penalties = np.logspace(-1, 4, 6)  # this is for v2 and 4
sessions = ["G", "H"]


# %% load up one solution

path = "fit_result4_test/result_0000.pkl"

with open(path, "rb") as f:
    solutions = pickle.load(f)

params = solutions["G"][0][0].keys()


# %% load all solution and make dataframe

datadict = {}
datadict["penalty"] = []
datadict["loss"] = []
datadict["seed"] = []
datadict["iteration"] = []
datadict["session"] = []
for p in params:
    datadict[p] = []

for seed in range(100):
    path = f"fit_result4_test/result_{seed:04d}.pkl"
    try:
        with open(path, "rb") as f:
            solutions = pickle.load(f)
    except:
        print(f"file {seed} not found")
        continue

    for i, p in enumerate(penalties):
        for iter, ii in enumerate(range(i * 10, (i + 1) * 10)):
            for s in sessions:
                sol = solutions[s][ii][0]
                datadict["penalty"].append(p)
                datadict["loss"].append(solutions[s][ii][1])
                datadict["seed"].append(seed)
                datadict["iteration"].append(iter)
                datadict["session"].append(s)
                for par in params:
                    datadict[par].append(sol[par])


df = pd.DataFrame(datadict)
df
# df.to_hdf("fit_result3_v2.h5", "results", complib="bzip2")
# %%

df.to_feather("fit_result4_test.feather")

# %% read speed check
df2 = pd.read_feather("fit_result3_v2.feather")
