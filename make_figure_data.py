# %% this script creates formatted data for figures.

import pickle
from importlib import reload

from matplotlib.lines import Line2D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import umap
import create_new_target_data as cnd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import SSN_model_funcs as ssn
import support_funcs as sf
from tqdm import tqdm
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

reload(cnd)
reload(sf)
reload(ssn)

tqdm.pandas()


# %% create the data for figure 3 (found solutions)
reload(sf)
# Necessary data are:
#   - dataframe of all of the cost functions (with penalty 10)
#   - dataframe of all of the parameters (+ LDA and UMAP)
#   - same dataframes for the selected data

# processing
#   - df need to come with different flavors
#      - all of the successful fits
#      - all cost < 0.3 (for UMAP)
#      - any cost but passed the parameter tests (color for umap)
#      - passed the parameter tests and cost < 0.2 (for LDA)
#      - ideally, all are in one df, and we can choose the elements from it.


def make_fig_data(df_raw, cost_df_raw, penalty_filter=None, make_umap=True):
    # df_raw = sf.load_result4_df()
    # cost_df_raw = pd.read_hdf("fit4_all_cost_resp.hdf")
    # pick only penalty 10
    if penalty_filter is not None:
        df_raw.query(f"penalty == {penalty_filter}", inplace=True)
        # df_raw.query("penalty == ", inplace=True)
    cost_df_raw = cost_df_raw.loc[df_raw.index]

    cost_df_raw["params_pass"] = cost_df_raw.parallel_apply(sf.filter_params, axis=1)
    # now I want to rename the columns so that it won't conflict with dhe df_raw.
    for pname in sf.sim_params:
        if pname in cost_df_raw.keys():
            cost_df_raw.rename(columns={pname: pname + "_cost"}, inplace=True)

    df_fig = df_raw.merge(cost_df_raw, left_index=True, right_index=True)
    df_fig["cost_pass_loose"] = df_fig.apply(sf.filter_cost_single, axis=1, thval=0.3)
    df_fig["cost_pass_tight"] = df_fig.apply(sf.filter_cost_single, axis=1, thval=0.2)
    df_fig["selected"] = df_fig["cost_pass_tight"] & df_fig["params_pass"]
    # df_fig.value_counts(['selected', 'session'])

    # now some of the simulation parameters are adjusted for further analysis.
    df_fig = sf.calculate_adjusted_connectivity(df_fig)
    df_fig = sf.calculate_final_connectivity(df_fig)

    # % now calculate umap using cost_pass_loose.
    if make_umap:
        scaler = StandardScaler()
        orig_data = df_fig.loc[df_fig["cost_pass_loose"]].loc[:, sf.sim_params_adj_nopv]
        scaled_data = scaler.fit_transform(orig_data)
        reduser = umap.UMAP(random_state=0)
        emb = reduser.fit_transform(scaled_data)  # may take several minutes.
        # store the umap embedding into the dataframe.
        df_fig.loc[df_fig["cost_pass_loose"], "umap_x"] = emb[:, 0]
        df_fig.loc[df_fig["cost_pass_loose"], "umap_y"] = emb[:, 1]

    # LDA analysis (Let's do it on the fly at the figure script)
    # orig_data_selected = df_fig.loc[df_fig["selected"]].loc[:, sf.sim_params_adj_nopv]
    # scaled_data_selected = scaler.fit_transform()
    # lda = LinearDiscriminantAnalysis(n_components=1)

    # Also, let's replace the session names from G, H to Familiar, Novel.
    # df_fig["session"] = df_fig["session"].replace({"G": "Familiar", "H": "Novel"})

    # to make it feather ready, I need to reset the index.
    df_fig.reset_index(inplace=True)
    # df_fig.to_feather("figure_data/df_gen4.feather")
    return df_fig


# %%
df_raw_gen4 = sf.load_result4_df()
cost_df_raw_gen4 = pd.read_hdf("fit4_all_cost_resp.hdf")
df_fig_gen4 = make_fig_data(df_raw_gen4, cost_df_raw_gen4, penalty_filter=10)
df_fig_gen4.to_feather("figure_data/df_gen4.feather")


# %% Now let's prepare gen5 and 6 data (target data manipulation)
# gen5 data contains
def type_data(row):
    # based on the info, return the type of the data.
    if row["session"] == "G":
        if row["fraction"] == 0.0:
            return "Familiar Original"
        elif row["fraction"] == 1.0:
            if row["generation"] == 5:
                return "Familiar + Novel image"
            if row["generation"] == 6:
                return "Familiar + Novel non-image"
    elif row["session"] == "H":
        if row["fraction"] == 0.0:
            return "Novel Original"
    return "Unused"


reload(sf)
df_raw_gen5 = sf.load_result5_df()
cost_df_raw_gen5 = pd.read_hdf("fit5_all_cost_resp.h5")
# reindex both before merging.
df_raw_gen5.reset_index(inplace=True, drop=True)
cost_df_raw_gen5.reset_index(inplace=True, drop=True)


df_fig_gen5 = make_fig_data(df_raw_gen5, cost_df_raw_gen5, make_umap=False)
# apply typing using pandarallel
df_fig_gen5["data_type"] = df_fig_gen5.parallel_apply(type_data, axis=1)

df_fig_gen5.to_feather("figure_data/df_gen5.feather")

# %%
df_fig_gen5.keys().tolist()

df_fig_gen5.data_type.value_counts()


df_raw_gen5
cost_df_raw_gen5
df_fig_gen5
