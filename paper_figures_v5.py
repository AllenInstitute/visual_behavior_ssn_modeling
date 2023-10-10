# simplified version of the figure script.
# %%
import numpy as np
import pandas as pd
import support_funcs as sf
import seaborn as sns
from importlib import reload
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from tqdm import tqdm
from joblib import Parallel, delayed
import SSN_model_funcs as ssn
import create_new_target_data as ctd
import os


# %% Load necessary data and do a bit of computations
df_fig_gen4 = pd.read_feather("figure_data/df_gen4.feather")

scaler = StandardScaler()
orig_data = df_fig_gen4.loc[df_fig_gen4["cost_pass_loose"]].loc[
    :, sf.sim_params_adj_nopv
]
scaled_data = scaler.fit_transform(orig_data)
pca = PCA(n_components=3)
emb = pca.fit_transform(scaled_data)
df_fig_gen4.loc[df_fig_gen4["cost_pass_loose"], "pca1"] = emb[:, 0]
df_fig_gen4.loc[df_fig_gen4["cost_pass_loose"], "pca2"] = emb[:, 1]
df_fig_gen4.loc[df_fig_gen4["cost_pass_loose"], "pca3"] = emb[:, 2]


# %% target data traces (Fig. 2 component)
mode = "sem"
target_data = sf.prepare_fit_data(errormode=mode)

pops = ["Exc", "PV", "SST", "VIP"]
fig, axs = plt.subplots(4, 1, figsize=(4, 7), sharex=True)
for i, pop in enumerate(pops):
    axs[i].plot(
        target_data["G"]["data"][:, i],
        color=sf.ghcolor[0],
        label="G",
    )
    axs[i].plot(
        target_data["H"]["data"][:, i],
        color=sf.ghcolor[1],
        label="H",
    )
    axs[i].fill_between(
        np.arange(6750),
        target_data["G"]["data"][:, i] - target_data["G"]["err"][:, i],
        target_data["G"]["data"][:, i] + target_data["G"]["err"][:, i],
        color=sf.ghcolor[0],
        alpha=0.15,
    )
    axs[i].fill_between(
        np.arange(6750),
        target_data["H"]["data"][:, i] - target_data["H"]["err"][:, i],
        target_data["H"]["data"][:, i] + target_data["H"]["err"][:, i],
        color=sf.ghcolor[1],
        alpha=0.15,
    )
    # axs[i].set_title(pop)
    # set it on y axis instead of title
    axs[i].set_ylabel(pop + " FR (Hz)", fontsize=14)
    if i == 0:
        axs[i].legend()
        # no legend box
        leg = axs[i].get_legend()
        leg.set_title("")
        leg.get_texts()[0].set_text("Familiar")
        leg.get_texts()[1].set_text("Novel")
        leg.get_frame().set_linewidth(0.0)

    ssn.add_flash(axs[i], color="gray", alpha=0.1)

# zoom into a specific range.
axs[0].set_xlim(750 * 2 - 200, 750 * 7 - 300)
axs[0].set_xticks(np.arange(750 * 2, 750 * 7, 750))
axs[0].set_xticklabels([-1500, -750, 0, 750, 1500])
# set box off for all axes
for ax in axs:
    sf.nobox(ax)

axs[3].set_xlabel("Time (ms)", fontsize=14)
plt.tight_layout()

# save figure
plt.savefig(f"paper_figures_v5/target_data_{mode}.pdf")
# plt.savefig(f"paper_figures_v5/target_data_{mode}.png", dpi=300)

# %% Model Input (L4Exc activity, Fig. 2 component)
stim_g = ssn.load_predefined_stimulus("G")
stim_h = ssn.load_predefined_stimulus("H")

fig, axs = plt.subplots(1, 1, figsize=(4.2, 2))
fig.patch.set_alpha(0.0)
axs.plot(stim_g, color=sf.ghcolor[0], label="G")
axs.plot(stim_h, color=sf.ghcolor[1], label="H")
axs.set_xlabel("Time (ms)")
axs.set_ylabel("L4 Exc activity (Hz)")
sf.nobox(axs)
axs.legend()

# no legend box
leg = axs.get_legend()
leg.set_title("")
leg.get_texts()[0].set_text("Familiar")
leg.get_texts()[1].set_text("Novel")
leg.get_frame().set_linewidth(0.0)

ssn.add_flash(axs, color="gray", alpha=0.1)

axs.set_xlim(750 * 2 - 200, 750 * 7 - 300)
axs.set_xticks(np.arange(750 * 2, 750 * 7, 750))
axs.set_xticklabels([-1500, -750, 0, 750, 1500])


plt.tight_layout()
plt.savefig("paper_figures_v5/L4E_input.pdf")

# %% Solution selection (Fig. 3 A-D)

df_selection = df_fig_gen4.query("params_pass == True")
figs, axss = sf.data_selection_figure_simple(
    df_selection, shading="stim", selection="paper"
)
figs[0].savefig("paper_figures_v5/selection_fam.pdf")
figs[1].savefig("paper_figures_v5/selection_nov.pdf")


# %% Number of selected solutions (Fig. 3 E-F)

reload(sf)
df_numsol = sf.number_of_solutions(df_selection)
sf.plot_number_of_solutions(df_numsol)
plt.savefig("paper_figures_v2/number_of_solutions.svg")
plt.savefig("paper_figures_v2/number_of_solutions.pdf")


# %% PCA solution selection figure (Fig. 3 G-I)

# preparing the selection criteria
cpl = "cost_pass_loose == True"
ncpl = "cost_pass_loose == False"
cpt = "cost_pass_tight == True"
ncpt = "cost_pass_tight == False"
pp = "params_pass == True"
npp = "params_pass == False"

common_settings = {
    "x": "pca1",
    "y": "pca2",
    "hue": "session",
    "hue_order": ["G", "H"],
    "s": 3,
    "alpha": 0.5,
}

df_cpl = df_fig_gen4.query(cpl)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.0), sharex=False, sharey=False)

df_pca_sel = df_fig_gen4.query(" and ".join([cpt, pp]))
df_pca_pbad = df_fig_gen4.query(" and ".join([cpl, npp, cpt]))
df_pca_cbad = df_fig_gen4.query(" and ".join([ncpt, cpl]))
df_fig_gen4.query(cpl).value_counts("session")

gray = (0.7, 0.7, 0.7)
for a in ax:
    # use the common setting above
    sns.scatterplot(
        data=df_cpl,
        **common_settings,
        palette=[gray, gray],
        ax=a,
    )
    a.set_xlabel("PCA 1")
    a.set_ylabel("PCA 2")
    sf.nobox(a)
    a.axis("equal")
    # legend off
    a.get_legend().remove()

# then, for each panel, plot the rejected and selected solutions.

um_cbad = sns.scatterplot(
    data=df_pca_cbad, **common_settings, palette=sf.ghcolor, ax=ax[0]
)
um_pbad = sns.scatterplot(
    data=df_pca_pbad, **common_settings, palette=sf.ghcolor, ax=ax[1]
)
um_g = sns.scatterplot(
    data=df_pca_sel.query('session=="G"'),
    **common_settings,
    palette=sf.ghcolor,
    ax=ax[2],
)
um_h = sns.scatterplot(
    data=df_pca_sel.query('session=="H"'),
    **common_settings,
    palette=sf.ghcolor,
    ax=ax[2],
)
# um_pbad.collections[5].zorder = -1


ax[0].set_title("Poor fit cost")
ax[1].set_title("Poor parameter cost")
ax[2].set_title("Accepted")

# turn off legends for all axes
for a in ax:
    a.get_legend().remove()
    a.set_xlim([-5, 30])
    a.set_ylim([-3, 20])

plt.tight_layout()
plt.savefig("paper_figures_v5/pca_selection_3panel.pdf")
plt.savefig("paper_figures_v5/pca_selection_3panel.png", dpi=300)

# %% Historgrams of all parameters (Fig. 4 A)
df_accept = df_fig_gen4.query("selected == True")

top_labels = [
    "tau (s)",
    "baseline",
    "L4 input",
    "Exc to",
    "PV to",
    "SST to",
    "VIP to",
]
# one for two rows
left_labels = ["Exc", "PV", "SST", "VIP"]
bindef = [
    np.linspace(0, 0.15, 31),  # tau
    np.linspace(0, 2, 31),  # baseline
    np.linspace(0, 3, 31),  # L4 input
    np.linspace(0, 5, 31),  # Exc to
    np.linspace(0, 3, 31),  # PV to
    np.linspace(0, 1, 31),  # SST to
    np.linspace(0, 0.5, 31),  # VIP to
]


fig, axs = plt.subplots(8, 7, figsize=(10, 4), sharex="col", sharey=True)
for i, p in enumerate(sf.sim_params_fin):  # 4 x 7
    # plot histogram with seaborn.
    # Two sessions are vertically stacked.
    # each plot is tightly close to each other, and the title is shown only at the top.
    # xaxis is shown only at the bottom. (but keep the ticks for each plot.)
    # For each vertical stack, the x axes are the same.
    # The Y is probability distribution in log, and they are the same for all plots.
    # The y axis is shown only at the left side.
    # There 8 figure rows. Each two rows are for one parameter.
    ri_f = (i % 4) * 2
    ri_n = (i % 4) * 2 + 1
    ci = i // 4
    ax_f = axs[ri_f, ci]
    ax_n = axs[ri_n, ci]

    # setting the axis properties
    ax_f.set_yscale("log")
    ax_n.set_yscale("log")

    # I want to make ax_f and ax_n closer, so move ax_f down a bit.
    ax_f.set_position(
        [
            ax_f.get_position().x0,
            ax_f.get_position().y0 - 0.012,
            ax_f.get_position().width,
            ax_f.get_position().height,
        ]
    )

    if p == "p_to_v_fin":  # no parameter tuning for this condition
        ax_f.set_visible(False)
        ax_n.tick_params(
            axis="y", which="both", left=False, right=False, labelleft=False
        )
        ax_n.tick_params(axis="x", which="both", colors=(0, 0, 0, 0))
        for spine in ax_n.spines.values():
            spine.set_visible(False)
        # make xtick invisible
        ax_n.set_xlabel(top_labels[ci])

        ax_n_above = axs[ri_n - 2, ci]
        ax_n_above.tick_params(axis="x", which="both", bottom=True, labelbottom=True)

        continue

    if ci == 0:
        # set the label across two rows
        ax_f.set_ylabel(left_labels[ri_f // 2], rotation=0, ha="right", va="center")
        # move the label a bit lower so that it goes across two rows
        ax_f.yaxis.set_label_coords(-0.5, 0.05)

    sf.nobox(ax_f)
    sf.nobox(ax_n)

    # set ylabel across two rows if it's the lestmost figure

    common_options = {
        "kde": False,
        "bins": bindef[ci],
        "alpha": 1,
        "stat": "probability",
    }
    sns.histplot(
        df_accept.query("session == 'G'")[p],
        ax=ax_f,
        color=sf.ghcolor[0],
        **common_options,
    )
    sns.histplot(
        df_accept.query("session == 'H'")[p],
        ax=ax_n,
        color=sf.ghcolor[1],
        **common_options,
    )
    ax_n.set_ylabel("")
    ax_n.set_xlabel("")
    # put the top label at the bottom as x axis label.
    if ri_n == 7:
        ax_n.set_xlabel(top_labels[ci])
plt.savefig("paper_figures_v5/parameter_histograms.pdf")

# %% Average shift with confidence interval (Fig. 4B)
# %%time
df_accept = df_fig_gen4.query("selected == True").copy()
# evaluate the average shift of the parameter.
# first, get the mean of the parameters for each session.
df_mean = df_accept.groupby("session").mean()[sf.sim_params_fin_nopv]

# difference of the parameters, expressed
nopvdf = df_mean[sf.sim_params_fin_nopv]
diff = nopvdf.loc["H"] - nopvdf.loc["G"]


# sample many random pairs to get the confidence interval.
n_sample = 1000000
n_param = len(sf.sim_params_fin_nopv)
diff_boot = np.zeros((n_sample, n_param))
df_g = df_accept.query("session == 'G'")[sf.sim_params_fin_nopv].to_numpy()
df_h = df_accept.query("session == 'H'")[sf.sim_params_fin_nopv].to_numpy()

# for i in tqdm(range(n_sample)):
#     # bootstrap sample (pick one sample)
#     df_g_one = df_g.sample(1)
#     df_h_one = df_h.sample(1)
#     diff_boot[i] = df_h_one.values - df_g_one.values

# make a jittable function version of the above. now df_g and df_h are numpy array.
from numba import njit, jit


@njit
def diff_bootstrap(df_g, df_h, n_sample):
    n_param = len(df_g[0])
    diff_boot = np.zeros((n_sample, n_param))
    for i in range(n_sample):
        # bootstrap sample (pick one sample)
        df_g_one = df_g[np.random.randint(0, len(df_g))]
        df_h_one = df_h[np.random.randint(0, len(df_h))]
        diff_boot[i] = df_h_one - df_g_one
    return diff_boot


diff_boot = diff_bootstrap(df_g, df_h, n_sample)
# %timeit diff_bootstrap(df_g, df_h, n_sample)


# get the 95% confidence interval
diff_boot_sorted = np.sort(diff_boot, axis=0)
diff_boot_025 = diff_boot_sorted[int(n_sample * 0.025), :]
diff_boot_975 = diff_boot_sorted[int(n_sample * 0.975), :]
diff_boot_95 = np.vstack((diff_boot_025, diff_boot_975))


# plot the vector
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
im = sf.plot_heatmap_with_anno(
    ax,
    np.array(diff),
    toptxt=diff_boot_975,
    bottomtxt=diff_boot_025,
    cmap=sf.gpmap,
    normalize=False,
)
plt.tight_layout()
# save the figure
plt.savefig("paper_figures_v5/mean_diff_gp.pdf")


# %% LDA Projection result (Fig. 5 B)
df_accept = df_fig_gen4.query("selected == True").copy()


# do LDA here
lda_data = df_accept[sf.sim_params_fin_nopv]
lda_data_norm = (lda_data - lda_data.mean()) / lda_data.std()
lda = LinearDiscriminantAnalysis(n_components=1, store_covariance=True)
# fit the model
lda.fit(lda_data_norm, df_accept["session"])
df_accept["LDA projection"] = lda.transform(lda_data_norm)
cov = lda.covariance_
#
lda.transform(lda_data_norm)
drange = np.arange(0, lda.scalings_.shape[0])

projections = np.dot(np.array(lda_data_norm)[:, drange], lda.scalings_[drange])
g_index = df_accept["session"] == "G"
h_index = df_accept["session"] == "H"

# plotting the histogram of the projected solutions
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
sns.histplot(
    df_accept,
    x="LDA projection",
    ax=ax,
    hue="session",
    hue_order=["G", "H"],
    bins=30,
    stat="probability",
    alpha=1,
    palette=sf.ghcolor,
)


def clean_gh_legend(ax):
    ax.get_legend().get_texts()[0].set_text("Familiar")
    ax.get_legend().get_texts()[1].set_text("Novel")
    # remove the title of the legend box
    ax.get_legend().set_title("")
    return ax


clean_gh_legend(ax)

ax.set_yscale("log")
ax.set_xlabel("LDA projection")
ax.set_ylabel("Probability")
sf.nobox(ax)

plt.tight_layout()
plt.savefig("paper_figures_v5/lda_histogram.pdf")


# %% within-class covariance matrix, derived by the LDA analysis. (Fig. 4 C)
# show in the same format as before for PCA.

# insert 19 th element in the cov matrix as nans.
cov_in = np.insert(cov, 19, np.nan, axis=0)
cov_in = np.insert(cov_in, 19, np.nan, axis=1)

# green to purple diverging color map

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
sns.heatmap(
    cov_in,
    ax=ax,
    cmap=sf.gpmap,
    # cmap='viridis',
    vmin=-1,
    vmax=1,
    center=0,
    cbar_kws={"shrink": 0.5},
)
ax.set_title("Within-class correlation matrix")
sf.nobox(ax)
# axis image
plt.axis("image")

# tick labels are the parameter names
ax.set_xticklabels(sf.sim_params_fin)
ax.set_yticklabels(sf.sim_params_fin, rotation=0)


# remove _fin
def apply_text_rule(text):
    new_text = text.replace("_fin", "")
    new_text = new_text.replace("input", "baseline to")
    new_text = new_text.replace("stim", "L4Exc to")
    new_text = new_text.replace("_e", " Exc")
    new_text = new_text.replace("_p", " PV ")
    new_text = new_text.replace("_s", " SST")
    new_text = new_text.replace("_v", " VIP")
    new_text = new_text.replace("e_to", "Exc to")
    new_text = new_text.replace("p_to", "PV  to")
    new_text = new_text.replace("s_to", "SST to")
    new_text = new_text.replace("v_to", "VIP to")
    return new_text


ax.set_xticklabels([apply_text_rule(x.get_text()) for x in ax.get_xticklabels()])
ax.set_yticklabels([apply_text_rule(x.get_text()) for x in ax.get_yticklabels()])

# change the font to monospace
for tick in ax.get_xticklabels():
    tick.set_fontname("monospace")
for tick in ax.get_yticklabels():
    tick.set_fontname("monospace")


plt.tight_layout()
# plt.savefig("paper_figures_v5/lda_correlation_matrix.svg")
plt.savefig("paper_figures_v5/lda_correlation_matrix.pdf")

# %% Example 2D parameter plot with the projection vector (Fig. 5 C)
df_accept_z = sf.zscore_df(df_accept, sf.sim_params_fin_nopv)


def plot_scatter(df_accept, param1, param2):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    df_accept_reorder = df_accept.sort_values(by=["session"]).copy()
    sns.scatterplot(
        data=df_accept_reorder,
        x=param1,
        y=param2,
        ax=ax,
        hue="session",
        hue_order=["G", "H"],
        palette=sf.ghcolor_bright,
        alpha=1,
        s=5,
    )
    clean_gh_legend(ax)

    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    sf.nobox(ax)

    plt.tight_layout()
    return ax


# plot_scatter(df_accept, "tau_v", "e_to_e_fin")


# add the mean values of the two parameters
def place_lda_vector(df_accept, ax, lda, param1, param2, flip=False):
    for i, s in enumerate(["G", "H"]):
        ax.scatter(
            df_accept[df_accept["session"] == s][param1].mean(),
            df_accept[df_accept["session"] == s][param2].mean(),
            color=sf.ghcolor[i],
            s=100,
            marker="*",
            # make it visible
            zorder=100,
        )

    # also a plot a vector that shows the direction of th hyperplane that LDA predicts.
    p1ind = sf.sim_params_fin_nopv.index(param1)
    p2ind = sf.sim_params_fin_nopv.index(param2)
    # arrow origin is the mean value of the Familiar session.
    arrow_origin = np.array(
        [
            df_accept[df_accept["session"] == "G"][param1].mean(),
            df_accept[df_accept["session"] == "G"][param2].mean(),
        ]
    )
    # arrow vector is derived from the LDA coefficients.
    coef_norm = lda.coef_[0] / np.linalg.norm(lda.coef_[0])
    arrow_vec = np.array([coef_norm[p1ind], coef_norm[p2ind]])
    # exaggerate the vector
    arrow_vec = arrow_vec * 5
    # normalize the vector
    # arrow_vec = arrow_vec / np.linalg.norm(arrow_vec) * 2
    if flip:
        arrow_vec = -arrow_vec
    # plot the vector
    head_size = 0.4
    ax.arrow(
        arrow_origin[0],
        arrow_origin[1],
        arrow_vec[0],
        arrow_vec[1],
        head_width=head_size,
        head_length=head_size,
        fc="k",
        ec="k",
        width=0.004,
        length_includes_head=True,
    )
    return ax


def place_mean_diff_vector(df_accept, ax, p1, p2):
    for i, s in enumerate(["G", "H"]):
        ax.scatter(
            df_accept[df_accept["session"] == s][p1].mean(),
            df_accept[df_accept["session"] == s][p2].mean(),
            color=sf.ghcolor[i],
            s=100,
            marker="*",
        )

    # also a plot a vector that shows the direction of th hyperplane that LDA predicts.
    arrow_origin = np.array(
        [
            df_accept[df_accept["session"] == "G"][p1].mean(),
            df_accept[df_accept["session"] == "G"][p2].mean(),
        ]
    )
    # arrow vector is derived from the LDA coefficients.
    arrow_vec = np.array(
        [
            df_accept[df_accept["session"] == "H"][p1].mean()
            - df_accept[df_accept["session"] == "G"][p1].mean(),
            df_accept[df_accept["session"] == "H"][p2].mean()
            - df_accept[df_accept["session"] == "G"][p2].mean(),
        ]
    )
    # plot the vector
    head_size = 0.4
    ax.arrow(
        arrow_origin[0],
        arrow_origin[1],
        arrow_vec[0],
        arrow_vec[1],
        head_width=head_size,
        head_length=head_size,
        fc="gray",
        ec="gray",
        width=0.002,
        length_includes_head=True,
    )
    return ax


def comp_scatter(df_accept, lda, p1, p2, flip=False):
    ax = plot_scatter(df_accept, p1, p2)
    ax = place_lda_vector(df_accept, ax, lda, p1, p2, flip=flip)
    ax = place_mean_diff_vector(df_accept, ax, p1, p2)


comp_scatter(df_accept_z, lda, "e_to_e_fin", "e_to_p_fin")
# zoom in alittle more.
plt.axis("equal")
plt.xlim(-5, 5)
plt.ylim(-3, 5)


plt.tight_layout()
plt.savefig("paper_figures_v5/scatter_EE_vs_EP.pdf")

# %% LDA Projection vector with confidence interval (Fig. 5 D)
# Do bootstrap analysis and get confidence intervals of the LDA projection vector.
# This can take minutes to run.
n_boot = 1000

n_sample = lda_data_norm.shape[0]
n_param = lda_data_norm.shape[1]
coef_boot = np.zeros((n_boot, n_param))


# function for getting one bootstrap sample
def lda_bootstrap(i):
    # bootstrap sample
    boot_idx = np.random.randint(0, n_sample, n_sample)
    # fit the model
    lda.fit(lda_data_norm.iloc[boot_idx], df_accept["session"].iloc[boot_idx])
    # normalize the coefficients
    return sf.normalize_vector(lda.coef_)


coef_boot = Parallel(n_jobs=8)(delayed(lda_bootstrap)(i) for i in tqdm(range(n_boot)))

# get the 95% confidence interval
coef_boot_sorted = np.sort(coef_boot, axis=0)
coef_boot_025 = coef_boot_sorted[int(n_boot * 0.025), :]
coef_boot_975 = coef_boot_sorted[int(n_boot * 0.975), :]
coef_boot_95 = np.vstack((coef_boot_025, coef_boot_975))


# plot coef in the matrix format.
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
im = sf.plot_heatmap_with_anno(
    ax,
    lda.coef_,
    toptxt=coef_boot_975,
    bottomtxt=coef_boot_025,
    cmap=sf.gpmap,
)
plt.tight_layout()
# save the figure
plt.savefig("paper_figures_v5/lda_coef_gp.pdf")


# %% Target data manipulation results: 2D LDA Projection (Fig. 6E)
df_fig_gen5 = pd.read_feather("figure_data/df_gen5.feather")

# drop if data_type is Unused
df_fig_gen5 = df_fig_gen5[df_fig_gen5["data_type"] != "Unused"].copy()
df_fig_gen5.value_counts(["data_type", "selected"])


def plot_biplot(ax, lda, headsize=1, offset=[0, 0], topnum=8):
    proj2d = lda.scalings_[:, :2]
    # get the dscending sort index with the length in the 2D space
    sort_idx = np.argsort(np.linalg.norm(proj2d, axis=1))[::-1]
    # plot the arrows of the top some
    ax.set_aspect("equal")
    for i in sort_idx[:topnum]:
        ax.arrow(
            offset[0],
            offset[1],
            proj2d[i, 0],
            proj2d[i, 1],
            head_width=headsize,
            head_length=headsize,
            fc="k",
            ec="k",
            length_includes_head=True,
        )
        # anchor text top left.
        ax.text(
            proj2d[i, 0] + offset[0],
            proj2d[i, 1] + offset[1],
            sf.sim_params_rm1_nopv[i],
            fontsize=8,
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    return ax


df5_accept = df_fig_gen5.query("selected == True").copy()

# multidimensional LDA on the selected dataset
# and show the results color coded by session using sns.
lda = LinearDiscriminantAnalysis(n_components=3, store_covariance=True)
data_scaled = StandardScaler().fit_transform(df5_accept[sf.sim_params_fin_nopv])

proj = lda.fit_transform(data_scaled, df5_accept["data_type"])
df5_accept["LDA 1"] = proj[:, 0]
df5_accept["LDA 2"] = proj[:, 1]
hue_order = [
    "Familiar Original",
    "Novel Original",
    "Familiar + Novel image",
    "Familiar + Novel non-image",
]
colors = [sf.ghcolor_bright[0], sf.ghcolor_bright[1], "limegreen", "violet"]
mean_colors = [sf.ghcolor[0], sf.ghcolor[1], "green", "purple"]
# combine the mean colors with hue_order to make a dictionary
mean_colors = dict(zip(hue_order, mean_colors))


fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
sns.scatterplot(
    ax=ax,
    data=df5_accept,
    x="LDA 1",
    y="LDA 2",
    hue="data_type",
    hue_order=hue_order,
    palette=colors,
    s=20,
)
sf.nobox(ax)
# revome legend title
ax.get_legend().set_title(None)
ax.set_aspect("equal")
ax.set_xlim(-40, 35)
ax.set_ylim(-22, 50)

# calculate the means of each cluster
ldameans = df5_accept.groupby("data_type")[["LDA 1", "LDA 2"]].mean()
for i, (data_type, subdf) in enumerate(df5_accept.groupby("data_type")):
    ax.scatter(
        subdf["LDA 1"].mean(),
        subdf["LDA 2"].mean(),
        color=mean_colors[data_type],
        s=100,
        marker="*",
    )

# add legend elements, indicating that a black star is the mean of each cluster.
handles, labels = ax.get_legend_handles_labels()
handles.append(plt.scatter([], [], color="black", marker="*"))
labels.append("Cluster mean")
ax.legend(handles, labels, loc="upper right")
# change the first two legend elements to "Familiar" and "Novel"
ax.get_legend().get_texts()[0].set_text("Familiar")
ax.get_legend().get_texts()[1].set_text("Novel")
ax.get_legend().get_texts()[2].set_text("Familiar non-img + Novel img")
ax.get_legend().get_texts()[3].set_text("Familiar img + Novel non-img")

# offset is the mean of the cluster means.
offset = ldameans.mean().values

# add biplot and save
plot_biplot(ax, lda, 2, offset, 8)
plt.tight_layout()
plt.savefig("paper_figures_v5/TDM_2DLDAplot_biplot.pdf")


# %% Target data manipulation results: full LDA projection vector (Fig. 6F)

lengths = np.linalg.norm(lda.scalings_[:, :2], axis=1)
norm_length = np.linalg.norm(lengths)
main_labels = np.array([r"$\rightarrow$"] * len(lengths))
top_labels = lda.scalings_[:, 0] / norm_length
bottom_labels = lda.scalings_[:, 1] / norm_length

fig, ax = plt.subplots(figsize=(6, 3))
ax, txt = sf.plot_heatmap_with_anno(
    ax,
    lengths / norm_length,
    alt_txt=main_labels,
    cmap="viridis",
    normalize=False,
    toptxt=top_labels,
    bottomtxt=bottom_labels,
    sigfig_sub=2,
)

# rotate the text accordingly with the angles of the LDA vectors.
rot_angles = np.arctan2(lda.scalings_[:, 1], lda.scalings_[:, 0]) * 180 / np.pi
# insert p to v element (19)
rot_angles = np.insert(rot_angles, 19, 0)
# make it 7 x 4, transpose, and linearize it again.
rot_angles = rot_angles.reshape(7, 4).T.flatten()
for i, txt in enumerate(txt):
    if i == 25:  # p to v
        # delete text
        txt.set_text("")
    txt.set_rotation(rot_angles[i])
    # also, make the text a bit larger
    txt.set_fontsize(18)

plt.tight_layout()

plt.savefig("paper_figures_v5/TDM_2DLDAvectorarrows.pdf")

# %% Gain analysis, function preparation


def simulate(samp, stim_ratio=1.0):
    session = samp.session
    figax = sf.plot_original_traces(session)
    solution = sf.get_simulation_results(
        session, samp[sf.sim_params], stim_ratio=stim_ratio
    )
    sf.fr_traces_one(solution, simulation=True, just_traces=True, figaxs=figax)


def mix_with_ratio(base, mod, ratio):
    return base * (1 - ratio) + mod * ratio


def artificial_stim(baseline, amplitude, duration=250):
    """Square pulse stimulus with 250 ms width and 750 ms period."""
    stim = np.zeros(6750)
    onsets = np.arange(0, 6750, 750)
    for onset in onsets:
        stim[onset : onset + duration] = 1
    stim = stim * amplitude + baseline
    return stim


def stim_amplitude_mod(session, ratio, onset=35, dur=500):
    stim = ssn.load_predefined_stimulus(session)
    onsets = onset + 750 * np.setdiff1d(np.arange(9), [4])
    on_period, off_period = ctd.on_off_periods(onset, dur)
    no_stim = stim.copy()
    for onset in onsets:
        no_stim[onset : onset + dur] = np.interp(
            np.arange(dur), [0, dur], [stim[onset], stim[onset + dur]]
        )

    mix_stim = mix_with_ratio(no_stim, stim, ratio)
    # safty layer not to have subzero values.
    mix_stim[mix_stim < 0] = 0
    return mix_stim


def solution_to_peak_curves(
    solution, plot=False, mean_res=False, stimtype="session", savename=None, range=None
):
    """
    arguments:
    """
    session = solution.session
    if stimtype == "session":
        if range is None:
            ratios = np.linspace(0, 5, 51)
            range = ratios
        else:
            ratios = range
        stims = [stim_amplitude_mod(session, ratio) for ratio in ratios]
        stims = np.array(stims)
    elif stimtype == "artificial":
        if range is None:
            amps = np.linspace(0, 25, 51)
            range = amps
        else:
            amps = range
        # amps = np.linspace(0, 10, 11)
        baseline = 2.7  # Hz
        duration = 250  # ms
        stims = [artificial_stim(baseline, amp, duration) for amp in amps]
        stims = np.array(stims)
    else:
        raise ValueError("stimtype must be either session or artificial.")
    if plot:
        figax = plt.subplots(1, 1, figsize=(3, 4))
        # colors = sns.color_palette("cool", len(stims))
        colors = plt.cm.cool(np.linspace(0, 1, len(stims)))
    peaks = []
    for i, stim in enumerate(stims):
        # do simulation with the stim.
        traces = ssn.sim_to_result_orig(stim, **solution[sf.sim_params])
        if plot:
            figax = sf.fr_traces_one(
                traces,
                simulation=True,
                just_traces=True,
                stim=i == 0,
                figaxs=figax,
                color=colors[i]
                # traces,
                # simulation=True,
                # stim=i == 0,
            )
            # plt.xlim(1500, 2000)
            # sf.add_shading(figax[1], [1500, 2250], [250, 250], alpha=0.05)
            plt.xlim(1400, 3000)
            if savename is not None:
                plt.savefig(savename + f"_{range[i]:.1f}.png")
        # evaluate the peak amplitude for each population.
        if mean_res:
            # on_period, off_period = ctd.on_off_periods(70, 300)
            on_period, off_period = ctd.on_off_periods(0, 300)
            on_period += 6750
            peaks.append(traces["output"][on_period].mean(axis=0))
            continue
        else:
            peaks.append(traces["output"][6750:9750].max(axis=0))
    peaks = np.array(peaks)
    if plot:
        plt.xlim(sf.range_to_show)
    return peaks


# %% Gain analysis, stimulus shape (Fig. 7 A)

df_sel = df_fig_gen4.query("selected == True").copy()


baseline = 2.7
amplitudes = [0.3, 1.3, 2.3, 3.3, 4.3]
plt.figure(figsize=(3, 3))


# define colors with winter colormap, reverse the order
# colors = plt.cm.winter(np.linspace(0, 1, len(amplitudes)))[::-1]
# I need a little more dramatic colors
colors = plt.cm.cool(np.linspace(0, 1, len(amplitudes)))

for i in range(len(amplitudes)):
    stim = artificial_stim(baseline, amplitudes[i])
    plt.plot(stim, color=colors[i], label=f"{amplitudes[i]:.1f} Hz")
    plt.gca().set_xticks([1500, 2250, 3000])
    plt.gca().set_xticklabels([0, 750, 1500])
    plt.xlim([1400, 2700])
    plt.ylim([0, 8])
    plt.xlabel("Time (ms)")
    plt.ylabel("Input from L4 (Hz)")
    if i == 0:
        ssn.add_flash(plt.gca(), start=1500, end=2700)

sf.nobox(ax=plt.gca())
plt.tight_layout()
# plt.legend()
# plt.savefig("paper_figures_v2/artificial_stims.png", dpi=300)
plt.savefig("paper_figures_v5/artificial_stims.pdf")


# %% Gain analysis, example traces (Fig. 7 B, C)


def plot_artificial_response(session):
    df_sel_i = df_sel[df_sel.session == session]
    # sample = df_sel_i.sample(1)
    # take a sample that has median value of the loss function.
    ind = np.where(
        df_sel_i["loss"] == df_sel_i["loss"].quantile(0.5, interpolation="nearest")
    )[0]
    sample = df_sel_i.iloc[ind]

    solution_to_peak_curves(
        sample.iloc[0], plot=True, stimtype="artificial", range=amplitudes
    )

    # manipulate the xy ticks.
    # xtick positions are 1500, 2250, 3000, and I want to call them, 0, 750, 1500.
    # ytick positions are 0, -30, -60, -90, and I want to call them, 0, 0, 0, 0.
    # also draw a horizontal line at each of the yticks.
    plt.gca().set_xticks([1500, 2250, 3000])
    plt.gca().set_xticklabels([0, 750, 1500])
    plt.gca().set_yticks([0, -10, -20, -30, -40, -50, -60, -70, -80, -90])
    plt.gca().set_yticklabels([0, 20, 10, 0, 20, 10, 0, 20, 10, 0])
    plt.gca().set_ylim([-95, 10])
    plt.gca().set_xlim([1400, 2700])
    plt.gca().set_xlabel("Time (ms)")
    # plt.gca().set_ylabel("Response (Hz)")
    names = {"G": "Familiar", "H": "Novel"}
    plt.gca().set_title(f"{names[session]} solution")
    plt.gca().axhline(0, linestyle="--", color="k")
    plt.gca().axhline(-30, linestyle="--", color="k")
    plt.gca().axhline(-60, linestyle="--", color="k")
    plt.gca().axhline(-90, linestyle="--", color="k")
    plt.tight_layout()
    sf.nobox(ax=plt.gca())
    plt.savefig(f"paper_figures_v5/artificial_stims_response_{session}.png", dpi=300)
    plt.savefig(f"paper_figures_v5/artificial_stims_response_{session}.pdf")


plot_artificial_response("G")
plot_artificial_response("H")


# %% Gain analysis, Response curves (Fig. 7 D, E)

# if the data are not generated, genetrate them.
if not os.path.exists("figure_data/peaks_all_artificial_mean_res2.pkl"):
    peaks_all_art2 = df_sel.parallel_apply(
        solution_to_peak_curves,
        axis=1,
        mean_res=True,
        stimtype="artificial",
        range=np.linspace(0, 10, 101),
    )
    peaks_all_art2.to_pickle("figure_data/peaks_all_artificial_mean_res2.pkl")


# load the data
mode = "artificial2"
filenames = {
    "session": "peaks_all_mean_res.pkl",
    "artificial": "peaks_all_artificial_mean_res.pkl",
    "artificial2": "peaks_all_artificial_mean_res2.pkl",
}
peaks_all = pd.read_pickle("figure_data/" + filenames[mode])
peaks_all_np = peaks_all.to_numpy()
peaks_stack = np.stack(peaks_all_np, axis=2)
#


Gind = np.array(df_sel.session == "G")
Hind = np.array(df_sel.session == "H")

nm = np.nanmedian(peaks_stack[:, :, Gind], axis=2)
nm = np.nanmedian(peaks_stack[:, :, Hind], axis=2)

ratios_modes = {
    "session": np.linspace(0, 5, 51),
    "artificial": np.linspace(0, 25, 51),
    "artificial2": np.linspace(0, 10, 101),
}
ratios = ratios_modes[mode]
pop_names = ["Exc", "PV", "SST", "VIP"]

gstack = peaks_stack[:, :, Gind]
g_low = np.nanpercentile(gstack, 5, axis=2)
g_med = np.nanpercentile(gstack, 50, axis=2)
g_high = np.nanpercentile(gstack, 95, axis=2)
# plot them all together.
figax = plt.subplots(1, 2, figsize=(7, 3), sharey=True, sharex=True)
ax = figax[1]
# # plot the confidence inteval with shading
for i in range(g_low.shape[1]):
    ax[0].fill_between(ratios, g_low[:, i], g_high[:, i], alpha=0.4)
    ax[0].plot(ratios, g_med[:, i], label=pop_names[i], linewidth=2.5)
ax[0].set_xlabel("Stimulus amplitude (Hz)")
ax[1].set_xlabel("Stimulus amplitude (Hz)")
ax[0].set_ylabel("Response (Hz)")

# do the same for H

h_low = np.nanpercentile(peaks_stack[:, :, Hind], 5, axis=2)
h_med = np.nanpercentile(peaks_stack[:, :, Hind], 50, axis=2)
h_high = np.nanpercentile(peaks_stack[:, :, Hind], 95, axis=2)
# instead, plot the upper and lower bound with shaded area.
for i in range(h_low.shape[1]):
    ax[1].fill_between(ratios, h_low[:, i], h_high[:, i], alpha=0.3)
    ax[1].plot(ratios, h_med[:, i], label=pop_names[i], linewidth=2.5)

ax[0].set_ylim([0, 20])
ax[0].set_xlim([-0.1, 5.9])
sf.nobox(ax[0])
sf.nobox(ax[1])
ax[0].legend(loc="upper left")


inds = [4, 6]  # for artificial
interval = np.diff(ratios[inds])
gslope = float(np.diff(g_med[inds, 0]) / interval)
hslope = float(np.diff(h_med[inds, 0]) / interval)
print(f"Gain slope: {gslope:.2f} (familiar) {hslope:.2f} (novel)")

# for both figures, plot dashed vertical lines at 2.3 (ghcolor 0) and 3.3 hz (ghcolor 1) with alpha 0.5
ax[0].axvline(2.3, alpha=0.5, color=sf.ghcolor[0], linestyle="--")
ax[0].axvline(3.3, alpha=0.5, color=sf.ghcolor[1], linestyle="--")

ax[1].axvline(2.3, alpha=0.5, color=sf.ghcolor[0], linestyle="--")
ax[1].axvline(3.3, alpha=0.5, color=sf.ghcolor[1], linestyle="--")
figax[0].tight_layout()
figax[0].savefig(f"paper_figures_v5/figure_gain_{mode}_quartile.pdf")


# %% Gain analysis, Excitatory neurons gain curve (Fig. 7 F)

incr = np.diff(ratios).mean()
g_deriv = np.diff(peaks_stack[:, 0, Gind], axis=0) / incr
h_deriv = np.diff(peaks_stack[:, 0, Hind], axis=0) / incr
# get the percentiles
g_low = np.nanpercentile(g_deriv, 5, axis=1)
g_med = np.nanpercentile(g_deriv, 50, axis=1)
g_high = np.nanpercentile(g_deriv, 95, axis=1)
h_low = np.nanpercentile(h_deriv, 5, axis=1)
h_med = np.nanpercentile(h_deriv, 50, axis=1)
h_high = np.nanpercentile(h_deriv, 95, axis=1)


fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
ax.fill_between(ratios[1:], g_low, g_high, alpha=0.3, color=sf.ghcolor[0])
ax.fill_between(ratios[1:], h_low, h_high, alpha=0.3, color=sf.ghcolor[1])
ax.plot(ratios[1:], g_med, label="Familiar", linewidth=2.0, color=sf.ghcolor[0])
ax.plot(ratios[1:], h_med, label="Novel", linewidth=2.0, color=sf.ghcolor[1])
# plot zero line
ax.plot([-1, 10], [0, 0], color="k", linestyle="--")
ax.set_xlim([-0.1, 5.9])
ax.set_ylim([-1.4, 2.6])
# add labels and legend
ax.set_xlabel("Stimulus amplitude (Hz)")
ax.set_ylabel("Derivative of Exc response")
ax.legend()
sf.nobox(ax)
fig.tight_layout()
fig.savefig(f"paper_figures_v5/figure_gain_{mode}_derivative_error.pdf")

# %%
