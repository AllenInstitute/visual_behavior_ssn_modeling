# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from SSN_model_funcs import *
import SSN_model_funcs as ssn
from tqdm import tqdm
import pickle

tqdm.pandas()

df = pd.read_feather("fit_result4.feather")
with pd.option_context("mode.use_inf_as_null", True):
    df = df.dropna()

df["log10(loss)"] = np.log10(df["loss"])


# prepare data for plotting
dpath = "./neuropixels_data/l23_mean_traces.pkl"
with open(dpath, "rb") as f:
    l23data = pickle.load(f)

ctypes = ["RS", "FS", "SST", "VIP"]
types = ["G", "H"]
data_array = {}
data_array_comparison = {}
data_array_comparison_err = {}
for type in types:
    data_all = [l23data[f"{ctype}_{type}"] for ctype in ctypes]
    data = np.array([d["mean"] for d in data_all])
    data_err = np.array([d["std"] for d in data_all])

    data_reformatted = data[:, 0 : (ssn.default_end - ssn.default_start)]
    data_reformatted = np.transpose(data_reformatted)
    data_array[type] = np.concatenate(
        (np.zeros_like(data_reformatted), data_reformatted)
    )
    data_array_comparison[type] = data_reformatted
    data_array_comparison_err[type] = data_err[
        :, 0 : (ssn.default_end - ssn.default_start)
    ].transpose()


# %% calculate broken down cost function for each parameter

params = df.keys()[5:-1]
params
infl_names = params[9:]


row = df.iloc[24]


def get_cost(row):
    fitp = row[params]
    pre_stim = ssn.load_predefined_stimulus(row["session"])
    r = ssn.sim_to_result_orig(pre_stim, **fitp)

    # calcualte the deviation for each cell type
    diff = (
        r["output"][ssn.default_start : ssn.default_end, :]
        - data_array_comparison[row["session"]]
    )
    lsq = diff**2 / data_array_comparison_err[row["session"]] ** 2
    cost_cell_type = lsq.mean(axis=0)
    # if you multiply this by 6750, you get the original cost for this part.

    # calculate the cost for the influence matrix
    infl_errs = ssn.stim_infl_fracerr + list(
        ssn.influence_matrix_fracerr.transpose().flatten()
    )
    cost_infl = ((fitp[infl_names] - 1) / infl_errs) ** 2
    cost_infl
    fitp[infl_names]
    infl_errs
    cell_types = ["e", "p", "s", "v"]
    cost_series = pd.Series(cost_cell_type, index=[f"cost_{t}" for t in cell_types])

    # calculate image response
    response_onset = 70
    response_offset = 305
    first_onset = ssn.default_start + response_onset
    first_offset = ssn.default_start + response_offset
    second_onset = ssn.default_start + 750 + response_onset
    response = r["output"][first_onset:first_offset, :].mean(axis=0)
    nonresponse = r["output"][first_offset:second_onset, :].mean(axis=0)

    resp_ratio = response / nonresponse
    resp_series = pd.Series(resp_ratio, index=[f"resp_{t}" for t in cell_types])

    return pd.concat((cost_series, cost_infl, resp_series))
    # return(cost_infl)


get_cost(row)

# %% think about applying it to all the columns
# %%time
# may take several hours
# subdf = df.iloc[range(1000)]
# subdf.apply(get_cost, axis=1, result_type='expand')

all_cost = df.progress_apply(get_cost, axis=1, result_type="expand")
all_cost.to_hdf("fit4_all_cost_resp.hdf", "cost")
#

# the code below were used for sanity check
# cost_infl_pen = cost_infl * row["penalty"]
# cost = (cost_cell_type * 6750).sum() + cost_infl_pen.sum()
# (cost_cell_type * 6750).sum()
# cost_infl_pen.sum()
# (row["loss"], cost, row["loss"] - cost)


# %%

df
all_cost
# if not defined, load the data
all_cost = pd.read_hdf("fit4_all_cost.hdf", "cost")

missing = np.setdiff1d(range(1200000), all_cost.index)


# %% start with the basic things

# all_cost['fit loss'] = all_cost['fit']
all_cost["fit_loss"] = all_cost[["cost_e", "cost_p", "cost_s", "cost_v"]].mean(axis=1)

all_cost["log10(fit_loss)"] = np.log10(all_cost["fit_loss"])
all_cost["penalty"] = df["penalty"].astype(str)

# sns.histplot(data=all_cost, x='fit_loss', bins=np.linspace(0, 2, 100), hue='penalty')
sns.histplot(
    data=all_cost, x="log10(fit_loss)", bins=np.linspace(3, 5, 100), hue="penalty"
)

all_cost["fit_loss"].min()

df["log10(fit_loss)"] = all_cost["log10(fit_loss)"]

# %%
df["penalty_str"] = df["penalty"].astype(str)

for i, subdf in df.groupby("session"):
    plt.figure(figsize=(8, 3))
    sns.histplot(
        subdf[subdf["penalty"] < 100],
        x="log10(fit_loss)",
        hue="penalty_str",
        bins=np.linspace(3, 5, 100),
    )
    plt.title(i)
    plt.figure(figsize=(8, 3))
    sns.histplot(
        subdf[subdf["penalty"] >= 100],
        x="log10(fit_loss)",
        hue="penalty_str",
        bins=np.linspace(3, 5, 100),
    )
    plt.title(i)


# %% choosing relatively wide range of solution
df["log10(fit_loss)"]

# %% showing histogram for each of them
all_cost["session"] = df["session"]
all_cost["penalty_str"] = df["penalty_str"]
all_cost["penalty"] = df["penalty"]
all_cost["param_loss"] = (
    (df["loss"] - all_cost["fit_loss"] * 4 * 6750) / df["penalty"] / 19
)

fig, ax = plt.subplots(2, 2, figsize=(12, 5))
c = 0


for i, subdf in all_cost.groupby("session"):
    xelem = "param_loss"
    # plt.figure(figsize=(8, 3))
    sns.histplot(
        subdf[subdf["penalty"] < 100],
        x=xelem,
        hue="penalty_str",
        bins=np.linspace(0, 10, 100),
        ax=ax[0, c],
    )
    # plt.title(i)
    ax[0, c].set_title(i)
    # plt.figure(figsize=(8, 3))
    sns.histplot(
        subdf[subdf["penalty"] >= 100],
        x=xelem,
        hue="penalty_str",
        bins=np.linspace(0, 10, 100),
        ax=ax[1, c],
    )
    # plt.title(i)
    ax[1, c].set_title(i)
    c += 1

plt.suptitle(xelem)
plt.tight_layout()
