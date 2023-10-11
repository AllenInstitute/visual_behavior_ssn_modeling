import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import SSN_model_funcs as ssn
import pickle

# import umap  # This will be imported on the fly
import hdbscan
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from importlib import reload
from tqdm import tqdm
import xarray as xr
import create_new_target_data as cnd

tqdm.pandas()


cell_types = ["e", "p", "s", "v"]
cell_costs = {
    "Exc": "cost_e",
    "PV": "cost_p",
    "SST": "cost_s",
    "VIP": "cost_v",
}
taus = [f"tau_{t}" for t in cell_types]
inputs = [f"input_{t}" for t in cell_types]
stims = [f"stim_{t}" for t in cell_types]
conns = [f"{t1}_to_{t2}" for t1 in cell_types for t2 in cell_types]
sim_params = ["conn_scale"] + taus + inputs + stims + conns
sim_params_rm1_nopv = sim_params[1:20] + sim_params[21:]
# sim_params_adj = sim_params[1:9].append(sim_params[9:] + "_adj")
# I'm getting error, so rewriting with the list comprehension
sim_params_adj = sim_params[1:9] + [f"{t}_adj" for t in sim_params[9:]]
sim_params_fin = sim_params[1:9] + [f"{t}_fin" for t in sim_params[9:]]
# drop p_to_v (element 19)
sim_params_adj_nopv = sim_params_adj[:19] + sim_params_adj[20:]
sim_params_fin_nopv = sim_params_fin[:19] + sim_params_fin[20:]


# color scheme for the sessions
ghcolor = [(0.66, 0.06, 0.086), (0.044, 0.33, 0.62)]
ghcolor_bright = [(0.96, 0.36, 0.386), (0.344, 0.63, 0.92)]
ghcolor_bad1 = [(1.0, 0.9, 0.5), (0.5, 0.9, 1.0)]
ghcolor_bad2 = [(1.0, 0.9, 0.9), (0.9, 0.9, 1.0)]
gpmap = sns.diverging_palette(120, 300, l=40, s=90, sep=5, center="light", as_cmap=True)

# a nice range to show for traces.
range_to_show = [750 * 1 + 500, 750 * 7 - 100]


def get_l23data():
    dpath = "./neuropixels_data/l23_mean_traces.pkl"
    with open(dpath, "rb") as f:
        l23data = pickle.load(f)
    return l23data


def prepare_fit_data(errormode="std"):
    # copy of cluster fit 4...
    l23data = get_l23data()
    ctypes = ["RS", "FS", "SST", "VIP"]
    types = ["G", "H"]
    data_both = {}
    for type in types:
        data_all = [l23data[f"{ctype}_{type}"] for ctype in ctypes]
        data = np.array([d["mean"] for d in data_all])
        data_err = np.array([d[errormode] for d in data_all])

        data_reformatted = data[:, 0 : (ssn.default_end - ssn.default_start)]
        data_reformatted = np.transpose(data_reformatted)
        data_err_reformatted = data_err[:, 0 : (ssn.default_end - ssn.default_start)]
        data_err_reformatted = np.transpose(data_err_reformatted)
        data_both[type] = {"data": data_reformatted, "err": data_err_reformatted}
    return data_both


def prepare_data_array():
    l23data = get_l23data()

    ctypes = ["RS", "FS", "SST", "VIP"]
    types = ["G", "H"]
    data_array = {}
    for type in types:
        data_all = [l23data[f"{ctype}_{type}"] for ctype in ctypes]
        data = np.array([d["mean"] for d in data_all])

        data_reformatted = data[:, 0 : (ssn.default_end - ssn.default_start)]
        data_reformatted = np.transpose(data_reformatted)
        data_array[type] = np.concatenate(
            (np.zeros_like(data_reformatted), data_reformatted)
        )
    return data_array


def load_result4_df():
    df = pd.read_feather("fit_result4.feather")
    with pd.option_context("mode.use_inf_as_null", True):
        df = df.dropna()
    df["log10(loss)"] = np.log10(df["loss"])
    return df


def load_result5_df():
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
    return df


def get_close_n(df, measure, value, n=10):
    m = np.abs(df[measure] - value)
    ms = m.sort_values()
    ind = ms[0:n].index
    # return df.loc[ind]
    return ind


def plot_cell_type_activity(df_merged, params):
    celltypes = {
        "sst": ("cost_s", "SST"),
        "vip": ("cost_v", "VIP"),
        "e": ("cost_e", "RS"),
        "pv": ("cost_p", "FS"),
    }

    for celltype in celltypes.values():
        target_costs = np.linspace(0.1, 2.0, 20)
        for target_cost in target_costs:
            subdf = df_merged.query("session == 'H' & penalty == 10.0")
            inds = get_close_n(subdf, celltype[0], target_cost, 1)

            for ind in inds:
                row = subdf.loc[ind]
                fitp = row[params]
                sessiontype = row["session"]
                pre_stim = ssn.load_predefined_stimulus(sessiontype)
                r = ssn.sim_to_result_orig(pre_stim, **fitp)
                ssn.plot_results_one(
                    r["output"],
                    df_merged[sessiontype],
                    r["stims"],
                    ssn.time_index,
                    type=celltype[1],
                )
                plt.title(f"{celltype[1]}, target_cost: {target_cost:.1f}")
                plt.savefig(f"target_costs_h/{celltype[1]}_{target_cost:.1f}.png")
    return


def filter_params(row):
    keys = row.keys()
    accept = True
    th = 3.0
    for key in keys:
        if ("_to_" in key) or ("stim_" in key):
            accept &= np.sqrt(row[key]) < th
    return accept


def get_ncells():
    cost_names = {"RS": "cost_e", "FS": "cost_p", "SST": "cost_s", "VIP": "cost_v"}
    ncells = {}
    ncells["G"] = {}
    ncells["H"] = {}
    l23data = get_l23data()
    for k, d in l23data.items():
        # the below line seems wrong... fixed on 11/22/2022...
        # nc = np.round(d["std"][0] / d["sem"][0]) # incorrect
        nc = np.round((d["std"][0] / d["sem"][0]) ** 2)  # correct
        if "G" in k:
            ncells["G"][cost_names[k[:-2]]] = nc
        else:
            ncells["H"][cost_names[k[:-2]]] = nc
    return cost_names, ncells


def adjusted_cost(row, name, ncells):
    return row[name] * ncells[row["session"]][name]


def filter_cost_single(row, thval=0.45):
    cells = ["e", "p", "s", "v"]
    accept = True
    for s in cells:
        key = f"cost_{s}"
        accept &= row[key] < thval
    return accept


def construct_p10df(df_merged):
    """prepare necessary penalty values for passing good dataset for penalty 10."""
    # p10df = df_merged[df_merged["penalty"] == 10]
    p10df = df_merged.query("penalty == 10").copy()

    p10df["params_pass"] = p10df.apply(filter_params, axis=1)
    p10df["cost_pass_loose"] = p10df.apply(filter_cost_single, axis=1, thval=0.3)
    p10df["cost_pass_tight"] = p10df.apply(filter_cost_single, axis=1, thval=0.2)
    p10df["selected"] = p10df["params_pass"] & p10df["cost_pass_tight"]

    # cost_names = {"RS": "cost_e", "FS": "cost_p", "SST": "cost_s", "VIP": "cost_v"}
    # cost_names, ncells = get_ncells()
    # for n in cost_names.values():
    #     p10df.loc[:, n + "_sem"] = p10df.apply(
    #         lambda r: adjusted_cost(r, n, ncells), axis=1
    #     )

    # p10df.loc[:, "cost_pass_sem"] = p10df.apply(filter_cost_single, axis=1)
    # p10df.value_counts(["session", "params_pass", "cost_pass_sem"]).sort_index()
    return p10df


def get_umap_embedding(params, passdf_params, pcalda=False, skipumap=False):
    # taus = params[1:5]
    # stims = params[5:13]
    # infls = params[13:]
    taus = params[0:4]
    stims = params[4:12]
    infls = params[12:]

    # lin_elements = stims[[0, 1, 2, 3]] # only inputs
    # log_elements = taus.append(infls)
    lin_elements = np.concatenate([taus, stims, infls])
    # log_elements = []

    scaler = StandardScaler()
    orig_data = passdf_params.loc[:, lin_elements]
    scaled_lin = scaler.fit_transform(orig_data)
    # scaled_lin = StandardScaler().fit_transform(passdf_params.loc[:, lin_elements])
    # scaled_log = StandardScaler().fit_transform(
    #     np.log(passdf_params.loc[:, log_elements])
    # )
    # scaled_all = np.concatenate([scaled_lin, scaled_log], axis=1)
    scaled_all = scaled_lin

    if skipumap:
        emb = np.zeros((passdf_params.shape[0], 2))
    else:
        import umap  # only import when necessary...

        reducer = umap.UMAP(random_state=0)
        emb = reducer.fit_transform(scaled_all)
    # do also pca if the option is given
    if pcalda:
        pca = PCA(n_components=4)
        pcalda_result = {}
        pcalda_result["pca_emb"] = pca.fit_transform(scaled_all)
        # pcalda_result["pca_emb"] = pca.fit_transform(orig_data)
        pcalda_result["pca"] = pca
        pcalda_result["scaled_all"] = scaled_all
        pcalda_result["scaler"] = scaler

        # also do lda and get the hyperplane
        lda = LinearDiscriminantAnalysis(n_components=1)
        labels = passdf_params["session"] == "H"
        lda.fit(scaled_all, labels)  # X is the scaled_all, y is the celltype

        # lda_result = {}
        # lda_result["accuracy"] = lda.score(scaled_all, labels)
        # lda_result["lda"] = lda
        lda_emb = lda.transform(scaled_all)
        # lda_result["emb"] = lda_emb

        pcalda_result["lda_emb"] = lda_emb
        pcalda_result["lda"] = lda
        pcalda_result["lda_accuracy"] = lda.score(scaled_all, labels)

        return emb, pcalda_result
    return emb


def plot_umap(passdf_params, mode="pca", **kwargs):
    if mode == "pca":
        x = "PCA1"
        y = "PCA2"
    elif mode == "lda":
        x = "LDA1"
        y = "LDA2"
    else:
        x = "UMAP1"
        y = "UMAP2"
    a = sns.scatterplot(
        data=passdf_params,
        x=x,
        y=y,
        linewidth=0,
        s=3,
        # palette="viridis",
        **kwargs,
    )
    a.set_aspect("equal")
    return a


def make_passdf_params(df, passdf, params, minsize=500, skipumap=False, pcalda=False):
    passdf_params = df.loc[passdf.index]
    # insert adjustment
    passdf_params = calculate_adjusted_connectivity(passdf_params)
    passdf_params = calculate_final_connectivity(passdf_params)
    params_adj = params[1:9].append(params[9:] + "_adj")

    emb = get_umap_embedding(
        params_adj, passdf_params, pcalda=pcalda, skipumap=skipumap
    )
    if pcalda:
        emb, pcalda_result = emb
        passdf_params["UMAP1"] = emb[:, 0]
        passdf_params["UMAP2"] = emb[:, 1]
        passdf_params["PCA1"] = pcalda_result["pca_emb"][:, 0]
        passdf_params["PCA2"] = pcalda_result["pca_emb"][:, 1]
        passdf_params["PCA3"] = pcalda_result["pca_emb"][:, 2]
        passdf_params["PCA4"] = pcalda_result["pca_emb"][:, 3]
        passdf_params["LDA1"] = pcalda_result["lda_emb"][:, 0]
    else:
        if not skipumap:
            passdf_params["UMAP1"] = emb[:, 0]
            passdf_params["UMAP2"] = emb[:, 1]
        pcalda_result = None

    # also do cluster analysis
    if not skipumap:
        labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=minsize).fit_predict(
            emb
        )
        passdf_params["labels"] = labels
        passdf_params["labels_str"] = passdf_params["labels"].astype(str)
    # passdf_params.value_counts("labels_str")
    return passdf_params, params_adj, pcalda_result


def double_omission_all(df, params, plot=True):
    def get_diff(row):
        return ssn.test_double_omission(row[params], row["session"])

    print("evaluating double omission response...")
    diffs = df.progress_apply(get_diff, axis=1)
    print("done.")
    if plot:
        diffs.hist()
    return diffs


def calculate_adjusted_connectivity(df):
    types = ["e", "p", "s", "v"]
    conn_params = [f"{t1}_to_{t2}" for t1 in types for t2 in types]
    conn_params = conn_params + [f"stim_{t}" for t in types]
    for param in conn_params:
        df[param + "_adj"] = df[param] * df["conn_scale"]
    return df


def calculate_final_connectivity(df):
    types = ["e", "p", "s", "v"]
    conn_params = [f"{t1}_to_{t2}" for t1 in types for t2 in types]
    conn_params = conn_params + [f"stim_{t}" for t in types]
    infl_mat = ssn.influence_matrix.T.flatten()
    stim_mat = ssn.stim_infl
    infl_all = np.concatenate([infl_mat, stim_mat])
    for i, param in enumerate(conn_params):
        df[param + "_fin"] = df[param] * df["conn_scale"] * np.abs(infl_all[i])
    return df


# heatmap and annotate_heatmap is copied from matplotlib tutorial.
def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    txtdata=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    voffset=0,
    hoffset=0,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    txtdata
        Text data used to annotate.  If None, it's the same as data.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if txtdata is None:
        txtdata = data

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.

    t_col_th = im.get_clim()[1] / 2
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # if the cmap of im is not viridis, then the color will be
            # white for absolute value more than 0.5
            if im.cmap.name != "viridis":
                # unless the type is 'numpy.ma.core.MaskedConstant'
                if type(data[i, j]) != np.ma.core.MaskedConstant:
                    kw.update(color=textcolors[int(np.abs(data[i, j]) > t_col_th)])
            else:
                kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
            text = im.axes.text(
                j + hoffset, i + voffset, valfmt(txtdata[i, j], None), **kw
            )
            texts.append(text)

    return texts


def fr_traces_one(
    data,
    figaxs=None,
    voffset=30,
    just_traces=False,
    simulation=False,
    stim=True,
    **kwargs,
):
    # Make a nice plot of the firing rates.
    if figaxs is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        fig, ax = figaxs

    if simulation:
        datakey = "output"
        xoffset = 6750
    else:
        datakey = "data"
        xoffset = 0

    cl = ssn.color_list

    cell_types = ["Exc", "PV", "SST", "VIP"]
    for j, cell_type in enumerate(cell_types):
        fr = data[datakey][xoffset:, j] - j * voffset
        if "color" in kwargs:
            ax.plot(fr, **kwargs)
        else:
            ax.plot(fr, color=cl[j], **kwargs)
        if just_traces:
            continue
        ax.axhline(-j * voffset, color="k", linestyle=":")
    if stim:
        ssn.add_flash(ax, start=750 * 2, end=750 * 6.5)
    ax.set_ylim(-3 * voffset, 10)
    ax.set_xlim(750 * 1 + 500, 750 * 7 - 100)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # no ticks
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])

    return fig, ax


def fr_traces(
    data, figaxs=None, voffset=30, just_traces=False, simulation=False, **kwargs
):
    # Make a nice plot of the firing rates.
    if figaxs is None:
        fig, axs = plt.subplots(1, 2, figsize=(5, 5))
    else:
        fig, axs = figaxs

    if simulation:
        datakey = "output"
        xoffset = 6750
    else:
        datakey = "data"
        xoffset = 0

    cl = ssn.color_list

    sessions = ["G", "H"]
    cell_types = ["Exc", "PV", "SST", "VIP"]
    for i, session in enumerate(sessions):
        for j, cell_type in enumerate(cell_types):
            fr = data[session][datakey][:, j] - j * voffset
            # if kwargs contains color, use that color.
            if "color" in kwargs:
                axs[i].plot(fr, **kwargs)
            else:
                axs[i].plot(fr, color=cl[j], **kwargs)
            if just_traces:
                continue
            axs[i].set_xlim(750 * 1 + 500 + xoffset, 750 * 7 - 100 + xoffset)
            axs[i].axhline(-j * voffset, color="k", linestyle=":")
        ssn.add_flash(axs[i], start=750 * 2 + xoffset, end=750 * 6.5 + xoffset)

        axs[i].set_ylim(-3 * voffset, 10)
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
        # no ticks
        axs[i].tick_params(axis="both", which="both", length=0)
        axs[i].set_yticklabels([])
        axs[i].set_xticklabels([])
        # add legends to the right of the figure
        # axs[i].set_title(titles[t])
    # place a scale bar outside of the plot
    hend = 750 * 7 + 100 + xoffset
    vpos = -3 * voffset + 5
    axs[1].plot([hend + 500, hend + 1500], [vpos, vpos], linewidth=2, color="k")
    axs[1].plot([hend + 500, hend + 500], [vpos, vpos + 20], linewidth=2, color="k")
    # add scale bar text
    axs[1].text(hend + 700, vpos - 5, "1 s", fontsize=12)
    axs[1].text(hend + 100, vpos + 5, "20 Hz", fontsize=12, rotation=90)

    axs[1].legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

    return fig, axs


def get_simulation_results(session, params, stim_ratio=1.0):
    stimulus = ssn.load_predefined_stimulus(session, stim_ratio=stim_ratio)
    solution = ssn.sim_to_result_orig(stimulus, **params)
    return solution


def plot_one_reslut(session, params, figax=None):
    solution = get_simulation_results(session, params)
    fig, axs = fr_traces_one(solution, simulation=True, just_traces=True, figaxs=figax)
    return fig, axs


def get_best_params(df):
    # get the best parameters
    best_row = df.loc[df["loss"].idxmin()]
    session = best_row["session"]
    # params can be picked up by sim_params
    params = best_row[sim_params]
    return session, params


def plot_best_traces(df, figax=None):
    session, params = get_best_params(df)
    fig, axs = plot_one_reslut(session, params, figax)
    return fig, axs


def plot_original_traces(session, figax=None):
    fig, axs = fr_traces_one(
        ssn.prepare_fit_data()[session], simulation=False, color="k", figaxs=figax
    )
    return fig, axs


# def GHcolor():
#     """make a sns color palette with predefined colors"""
#     return [(0.66, 0.06, 0.086), (0.044, 0.33, 0.62)]


def nobox(ax):
    """remove top and right spines and ticks"""
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


def data_selection_figure_simple(df, **kwargs):
    return data_selection_figure(df, df, **kwargs)


def data_selection_figure_single(df, session, near=0.0, responsive=False):
    # do it only for the excitatory cells
    l23name = {"Exc": "RS"}
    cl = ssn.color_list
    l23data = ssn.get_l23data()
    subdf = df[df["session"] == session]
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.5))

    # plot the target
    target_data = l23data[f"RS_G"]["mean"]
    x = np.arange(len(target_data)) - 1300
    ax[1].plot(x, target_data, color="k", label="Target data")
    ax[1].set_xlim(0, 0 + 750 * 5)
    # ax.set_ylim(0.0, 25)

    # plot the simulation result
    stim = ssn.load_predefined_stimulus(session)

    costdf = subdf.query(f"cost_e > {near}").sort_values("cost_e", ascending=True)
    if costdf.empty:
        costdf = subdf.sort_values("cost_e", ascending=True)
    row = costdf.iloc[0]
    # run simulation
    sim_result = ssn.sim_to_result(stim, **row[sim_params])
    # plot the simulation result
    x = np.arange(len(sim_result[:, 0])) - 1300
    ax[1].plot(x, sim_result[:, 0], color="r", label="Model output")
    ax[1].set_xlim(0, 0 + 750 * 5)

    # plot the histogram of the cost function with semilog y.
    xrange = np.linspace(0.0, 0.5, 50)
    ax[0].hist(subdf["cost_e"], bins=xrange, color=cl[0])

    ax[0].set_xlabel("cost_e")
    ax[0].set_ylabel("loss")
    ax[0].set_yscale("log")
    # ax[0].set_xlim(0, 0.2)
    # ax[0].set_ylim(0, 0.2)
    # place a vertical line at the cost_e value
    ax[0].axvline(row["cost_e"], color="r", linestyle="--")

    if responsive:
        # plot the shading for the non-responsive cells
        subdf_noresp = subdf.query(f"resp_e < 1.2")
        ax[0].hist(subdf_noresp["cost_e"], bins=xrange, color="gray", alpha=0.8)

    # remove box
    nobox(ax[0])
    nobox(ax[1])

    leg = ax[1].legend()
    # place it above the figure
    leg.set_bbox_to_anchor((0.6, 1.0))

    stim_onsets = np.array([0, 750, 2250, 3000]) + 200
    stim_dur = [250] * 4
    add_shading(ax[1], stim_onsets, stim_dur)

    plt.tight_layout()
    return fig, ax


def data_selection_figure(
    df,
    p10df,
    fraction=None,
    alternate=False,
    shading=None,
    threshold=0.2,
    nocost=False,
    selection="normal",
    targetcol="gray",
    targetstyle="--",
):
    cell_costs = {
        "Exc": "cost_e",
        "PV": "cost_p",
        "SST": "cost_s",
        "VIP": "cost_v",
    }

    l23name = {"Exc": "RS", "PV": "FS", "SST": "SST", "VIP": "VIP"}
    cl = ssn.color_list
    l23data = ssn.get_l23data()

    if selection == "normal":
        col_num = 7
        costs = np.linspace(0.00, 0.30, col_num)
        # cost_cols = matplotlib.cm.get_cmap("tab10", col_num)
        cost_cols = matplotlib.cm.get_cmap("viridis", col_num).colors
        method = "normal"
    elif selection == "paper":
        costs = np.array([0])
        cost_cols = ["green"]
        method = "best"  # just show the best solution

    figs = []
    axss = []
    for group, subdf in p10df.groupby("session"):
        fig, axs = plt.subplots(4, 2, figsize=(8, 6))
        if fraction is None:
            stim = ssn.load_predefined_stimulus(group)
        else:
            data, stim = cnd.prepare_interp_data_for_fit(
                [fraction], group, alternate=alternate
            )
            data = data[0]
            stim = stim[0]
        for i, (key, value) in enumerate(cell_costs.items()):
            xrange = np.linspace(0.0, 0.5, 50)
            axs[i, 0].hist(subdf[value], bins=xrange, color=cl[i])

            # also pick up the df for simulations with no image responses
            subdf_noresp = subdf.query(f"resp_{key[0].lower()} < 1.2")
            axs[i, 0].hist(subdf_noresp[value], bins=xrange, color="gray", alpha=0.8)
            # set y axis to log scale
            axs[i, 0].set_yscale("log")
            # remove right and top spines
            nobox(axs[i, 0])
            nobox(axs[i, 1])

            for j, cost in enumerate(costs):
                if method == "normal":
                    # get the parameter set for the given cost
                    costdf = subdf.query(f"{value} < {cost}").sort_values(
                        value, ascending=False
                    )
                    # if not empty plot the simulation results
                    if costdf.empty:
                        continue
                elif method == "best":
                    costdf = subdf.sort_values(value, ascending=True).head(1)
                    # override the cost to make the plot accurate
                    cost = costdf[value].values[0]
                # getting the index of the best simulation
                idx = costdf.index[0]
                # sim_params = df.loc[idx].iloc[5 : 5 + 29].to_dict()
                sim_params_dict = df.loc[idx, sim_params].to_dict()
                # sim_params
                sim_result = ssn.sim_to_result(stim, **sim_params_dict)
                x = np.arange(len(sim_result[:, i])) - 1300
                if not nocost:
                    axs[i, 1].plot(
                        x, sim_result[:, i], color=cost_cols[j], label=f"{cost:.2f}"
                    )
                    axs[i, 0].axvline(cost, color=cost_cols[j], linestyle="--")

            # special case for setting a threshold
            # th = 0.17
            if threshold is not None:
                th = threshold
                thcol = "orange"
                axs[i, 0].axvline(th, color=thcol, linestyle="--")
                idx = (
                    subdf.query(f"{value} < {th}")
                    .sort_values(value, ascending=False)
                    .index[0]
                )

                # sim_params = df.loc[idx].iloc[5 : 5 + 29].to_dict()
                sim_params_dict = df.loc[idx, sim_params].to_dict()
                sim_result = ssn.sim_to_result(stim, **sim_params_dict)
                axs[i, 1].plot(x, sim_result[:, i], color=thcol, label=f"{th:.2f}")

            # plot the target data
            if fraction is None:
                target_data = l23data[f"{l23name[key]}_{group}"]["mean"]
            else:
                target_data = data["data"][:, i]
            x = np.arange(len(target_data)) - 1300
            axs[i, 1].plot(
                x, target_data, color=targetcol, linestyle=targetstyle, label="target"
            )
            axs[i, 1].set_xlim(0, 0 + 750 * 5)
            axs[i, 1].set_ylim(0.0, 25)
            # add shading
            if shading == "stim":
                stim_onsets = np.array([0, 750, 2250, 3000]) + 200
                stim_dur = [250] * 4
                add_shading(axs[i, 1], stim_onsets, stim_dur)
            elif shading == "session":
                # make distinction between image vs non-image periods.
                sess_onsets = np.array([0, 750, 2250, 3000]) + 270
                sess_dur = [305] * 4
                # print(sess_onsets)
                # fill the other area with anothre color.
                non_sess_onsets = [0] + list((sess_onsets + 305))
                nsdur = 750 - 305
                non_sess_durs = [270, nsdur, nsdur + 750, nsdur, nsdur]
                # print(non_sess_onsets)

                if fraction == 0.0:
                    # fill the entire session.
                    if group == "G":
                        add_shading(
                            axs[i, 1], [0], [4000], color=ghcolor[0], alpha=0.07
                        )
                    elif group == "H":
                        add_shading(
                            axs[i, 1], [0], [4000], color=ghcolor[1], alpha=0.07
                        )
                else:
                    if alternate:
                        c = ghcolor
                    else:
                        c = [ghcolor[1], ghcolor[0]]
                    # print(c)
                    add_shading(
                        axs[i, 1], sess_onsets, sess_dur, color=c[0], alpha=0.07
                    )
                    add_shading(
                        axs[i, 1],
                        non_sess_onsets,
                        non_sess_durs,
                        color=c[1],
                        alpha=0.07,
                    )

        plt.tight_layout()
        # save the figs
        figs.append(fig)
        axss.append(axs)
    return figs, axss


def add_shading(ax, onsets, durations, color="gray", alpha=0.15):
    """add shading to the plot"""
    for onset, duration in zip(onsets, durations):
        ax.axvspan(onset, onset + duration, color=color, alpha=alpha, zorder=-1)
    return ax


def number_of_solutions(df):
    df_dict = []
    for group, subdf in df.groupby("session"):
        for th in np.linspace(0.01, 0.31, 31):
            good_inds = []
            for key, value in cell_costs.items():
                # print(f"{group} {key}")
                good_df = subdf.query(f"{value} < {th}")
                good_inds.append(good_df.index)
                num_good = good_df.shape[0]
                # print(f"{th:.2f} {len(subdf.query(f'{value} < {th}'))}")
                df_elem = {
                    "session": group,
                    "cell_type": key,
                    "threshold": th,
                    "num_good": num_good,
                }
                # make this into a dataframe and plot the results
                df_dict.append(df_elem)
            # also cauculate the intersection of the solutions
            num_good = len(set(good_inds[0]).intersection(*good_inds[1:]))
            inter_elem = {
                "session": group,
                "cell_type": "All",
                "threshold": th,
                "num_good": num_good,
            }
            df_dict.append(inter_elem)

    df_numsol = pd.DataFrame(df_dict)
    return df_numsol


def plot_number_of_solutions(df_numsol, vertical=False):
    titles = {"G": "Familiar", "H": "Novel"}
    if vertical:
        fig, axs = plt.subplots(2, 1, figsize=(5, 3))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(7, 2.5))
    for i, (group, subdf) in enumerate(df_numsol.groupby("session")):
        sns.lineplot(
            data=subdf,
            x="threshold",
            y="num_good",
            hue="cell_type",
            hue_order=["Exc", "PV", "SST", "VIP", "All"],
            ax=axs[i],
            palette=ssn.color_list + ["k"],
        )
        axs[i].set_xlabel("Acceptance threshold")
        axs[i].set_ylabel("Number of solutions")
        axs[i].set_title(f"{group}")
        axs[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axs[i].set_xlim(0.0, 0.3)
        # axs[i].set_ylim(0, 5)
        axs[i].set_yscale("log")
        axs[i].set_ylim(1, 1e5)
        axs[i].set_title(titles[group])
        nobox(axs[i])
        # for the first one, no legend, ylabels, and xlabels
        if vertical:
            if i == 1:
                axs[i].legend_.remove()
            else:
                axs[i].legend_.get_frame().set_linewidth(0.0)
        else:
            if i == 0:
                axs[i].legend_.remove()
            else:
                axs[i].set_ylabel("")
                # axs[i].set_xlabel("")
                # no box for the legend
                axs[i].legend_.get_frame().set_linewidth(0.0)

        # plot vertical line at the threshold
        axs[i].axvline(0.2, color="orange", linestyle="--")

    plt.tight_layout()
    return fig, axs
    # plt.savefig("paper_figures/fig3_num_solutions.svg", bbox_inches="tight")


def zscore_df(df, elements):
    # copy the origial structure, and calculate the zscores for the desired elements.
    df_norm = df.copy()
    for e in elements:
        df_norm[e] = (df[e] - df[e].mean()) / df[e].std()
    return df_norm


def normalize_vector(vec):
    return vec / np.linalg.norm(vec)


def reshape_4vec(vec):
    return vec.reshape(7, 4).T


def plot_heatmap_with_anno(
    ax,
    matrix,
    alt_txt=None,
    addzero=True,
    normalize=True,
    reshape=True,
    toptxt=None,
    bottomtxt=None,
    sigfig=2,
    sigfig_sub=3,
    cmap="seismic",
):
    if addzero:
        matrix = np.insert(matrix, 19, 0)
        if toptxt is not None:
            toptxt = np.insert(toptxt, 19, 0)
        if bottomtxt is not None:
            bottomtxt = np.insert(bottomtxt, 19, 0)
        if alt_txt is not None:
            alt_txt = np.insert(alt_txt, 19, 0)

    if normalize:
        matrix = normalize_vector(matrix)
    if reshape:
        matrix = reshape_4vec(matrix)
        if toptxt is not None:
            toptxt = reshape_4vec(toptxt)
        if bottomtxt is not None:
            bottomtxt = reshape_4vec(bottomtxt)
        if alt_txt is not None:
            alt_txt = reshape_4vec(alt_txt)
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im, cbar = heatmap(
        matrix,
        ax=ax,
        col_labels=[
            "tau",
            "baseline",
            "L4 Exc to",
            "Exc to",
            "PV to",
            "SST to",
            "VIP to",
        ],
        row_labels=["Exc", "PV", "SST", "VIP"],
        cmap=cmap,
    )

    # annotate_heatmap(im, matrix, valfmt="{x:.3f}")
    # use sigfig
    if alt_txt is not None:
        txt = annotate_heatmap(im, matrix, txtdata=alt_txt, valfmt="{x}")
    else:
        txt = annotate_heatmap(im, matrix, valfmt=f"{{x:.{sigfig}f}}")

    # add text to the top and bottom
    if toptxt is not None:
        annotate_heatmap(
            im,
            matrix,
            valfmt=f"{{x:.{sigfig_sub}f}}",
            txtdata=toptxt,
            hoffset=0.45,
            voffset=-0.45,
            va="top",
            ha="right",
            fontsize=7,
        )
    if bottomtxt is not None:
        annotate_heatmap(
            im,
            matrix,
            valfmt=f"{{x:.{sigfig_sub}f}}",
            txtdata=bottomtxt,
            hoffset=0.45,
            voffset=0.45,
            va="bottom",
            ha="right",
            fontsize=7,
        )

    maxval = np.max(np.abs(matrix))
    # if not viridis
    if cmap != "viridis":
        im.set_clim(-maxval, maxval)
    im.get_clim()
    # rotate the x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    # plot a white vertical line between the 1st and 2nd columns, and between the 3rd and 4th columns
    ax.axvline(0.5, linewidth=2, color="white", linestyle="-")
    ax.axvline(2.5, linewidth=2, color="white", linestyle="-")
    return im, txt
