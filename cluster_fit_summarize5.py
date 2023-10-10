# %%
import numpy as np
import pickle
import pandas as pd
# from numba import njit

# sessions = ["G", "H", "F"]
sessions = ["H"]  # round 2

path = "fit_result5/result_0001.pkl"

with open(path, "rb") as f:
    solutions = pickle.load(f)

params = solutions["H"][0][0].keys()

# %% load all solution and make dataframe

fractions = np.linspace(-0.5, 1.5, 9)
datadict = {}
datadict['fraction'] = []
datadict['loss'] = []
datadict['seed'] = []
datadict['iteration'] = []
datadict['session'] = []
for p in params:
    datadict[p] = []

for seed in range(1, 1000):
    path = f"fit_result5/result_{seed:04d}.pkl"
    with open(path, "rb") as f:
        solutions = pickle.load(f)

    for i, f in enumerate(fractions):
        for iter, ii in enumerate(range(i * 10, (i + 1) * 10)):
            for s in sessions:
                sol = solutions[s][ii][0]
                datadict['fraction'].append(f)
                datadict['loss'].append(solutions[s][ii][1])
                datadict['seed'].append(seed)
                datadict['iteration'].append(iter)
                datadict['session'].append(s)
                for par in params:
                    datadict[par].append(sol[par])

# %%

df = pd.DataFrame(datadict)
df.to_feather("fit_result5_r2.feather")