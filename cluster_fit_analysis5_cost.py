# %%
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import create_new_target_data as cnd
import SSN_model_funcs as ssn
from cycler import cycler
import pickle

from tqdm import tqdm

tqdm.pandas()

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


df = pd.read_feather("fit_result5.feather")
df2 = pd.read_feather("fit_result5_r2.feather")
df3 = pd.read_feather("fit_result6.feather")
df["generation"] = 5
df2["generation"] = 5
df3["generation"] = 6

# replace the H sessions  from df2
df = pd.concat([df.query("session != 'H'"), df2, df3])

with pd.option_context("mode.use_inf_as_null", True):
    df = df.dropna()

df["log10(loss)"] = np.log10(df["loss"])

fracs = np.linspace(-0.5, 1.5, 9)


# for each session and fraction, prepare fit data.
sessions = ["G", "H", "F"]
fractions = np.linspace(-0.5, 1.5, 9)
d, s = cnd.prepare_interp_data_for_fit([0], "G")
d[0]
s[0]

fit_data = {}
for s in ["G", "H", "F"]:
    fit_data[s] = {}
    for f in fractions:
        data, stim = cnd.prepare_interp_data_for_fit([f], s)
        fit_data[s][f] = {}
        fit_data[s][f]["traces"] = data[0]
        fit_data[s][f]["stim"] = stim[0]

# special case... for the generation 6, we used G based, fraction 1.0, alternate=True.
g6data, g6stim = cnd.prepare_interp_data_for_fit([1.0], "G", alternate=True)
# now fit_data is usable with fit_data[session][fraction]['traces'] and fit
fit_data[s][f]["traces"]["data"]

# %%
params = df.keys()[5:-2]  # as we added generation, it became -2
params
infl_names = params[9:]


row = df.iloc[-255]


def get_cost(row):
    s = row["session"]
    f = row["fraction"]
    fitp = row[params]
    # pre_stim = ssn.load_predefined_stimulus(row["session"])
    if row["generation"] == 6:
        pre_stim = g6stim[0]
        trace_data = g6data[0]["data"]
        trace_err = g6data[0]["err"]
    else:
        pre_stim = fit_data[s][f]["stim"]
        trace_data = fit_data[s][f]["traces"]["data"]
        trace_err = fit_data[s][f]["traces"]["err"]
    r = ssn.sim_to_result_orig(pre_stim, **fitp)

    # calcualte the deviation for each cell type
    # diff = r["output"][ssn.default_start:ssn.default_end, :] - fit_data[s][f]['traces']['data']
    diff = r["output"][ssn.default_start : ssn.default_end, :] - trace_data
    # lsq = diff ** 2 / data_array_comparison_err[row["session"]] ** 2
    # lsq = diff ** 2 / fit_data[s][f]['traces']['err'] ** 2
    lsq = diff**2 / trace_err**2
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


# %%
# %%time
all_cost = df.parallel_apply(get_cost, axis=1, result_type="expand")
all_cost.to_hdf("fit5_all_cost_resp.h5", "cost")
# %%
