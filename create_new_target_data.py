# %%
# making new target data for a simple sensitivity analysis.

import SSN_model_funcs as ssn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# import matplotlib.patches as patches
import seaborn as sns
from importlib import reload
from scipy.signal import butter, lfilter
from scipy.ndimage import gaussian_filter1d


# %%


# %% plotting it
# plt.xlim([2250, 2350])
# plt.xlim([2050 + 250, 2650 + 250])


def gaussian_filter(data, sigma=20):
    return gaussian_filter1d(data, sigma, axis=0)


def on_off_periods(stim_period_onset, stim_period_duration):
    # This assumes the duration to be 6750 ms.
    duration = 6750
    onsets = stim_period_onset + 750 * np.setdiff1d(np.arange(9), [4])
    offsets = onsets + stim_period_duration
    on_period = np.concatenate(
        [np.arange(onset, offset) for onset, offset in zip(onsets, offsets)]
    )
    off_period = np.setdiff1d(np.arange(0, duration), on_period)
    return on_period, off_period


def make_on_off_diffs(fit_data, stim_period_onset, stim_period_duration):
    on_period, off_period = on_off_periods(stim_period_onset, stim_period_duration)

    # define the difference between G and H for image response and non-image period
    on_diff = np.zeros_like(fit_data["G"]["data"])
    on_diff[on_period, :] = gaussian_filter(
        fit_data["H"]["data"][on_period, :] - fit_data["G"]["data"][on_period, :]
    )

    off_diff = np.zeros_like(fit_data["G"]["data"])
    off_diff[off_period, :] = gaussian_filter(
        fit_data["H"]["data"][off_period, :] - fit_data["G"]["data"][off_period, :]
    )
    return on_diff, off_diff


def make_on_off_errs(fit_data, session, stim_period_onset, stim_period_duration):
    on_period, off_period = on_off_periods(stim_period_onset, stim_period_duration)

    on_err = np.zeros_like(fit_data["G"]["err"])
    on_err[on_period, :] = fit_data[session]["err"][on_period, :]
    off_err = np.zeros_like(fit_data["G"]["err"])
    off_err[off_period, :] = fit_data[session]["err"][off_period, :]
    return on_err, off_err


def plot_diffs(orig, mod):
    ssn.plot_results_ct(
        orig, mod, None, range(6750), 0, 6750, legend=("Original", "Modified")
    )


def plot_mod_data(fracs, orig, diff):
    fig, ax = plt.subplots(2, 2, figsize=(15, 6))
    titles = ["RS", "FS", "SST", "VIP"]

    cool = matplotlib.cm.get_cmap("cool")
    colors = cool(np.linspace(0, 1, len(fracs)))

    for j, frac in enumerate(fracs):
        mod = orig + frac * diff
        for i in range(4):
            ax[i // 2, i % 2].plot(mod[:, i], color=colors[j])

    for i in range(4):
        ax[i // 2, i % 2].plot(orig[:, i], "k")
        ax[i // 2, i % 2].set_title(titles[i])
        ssn.add_flash(ax[i // 2, i % 2])
        ax[i // 2, i % 2].set_xlim([0, 6750])

    ax[0, 1].legend(
        [f"Modified: {frac}" for frac in fracs] + ["Original"], loc="upper right"
    )
    plt.tight_layout()
    return fig, ax


# To make the diffs, we need the original error, the other error, mixing fraction.
# and the equation will be sqrt((original * (1 - frac))^2 + (other * frac)^2)
def make_targets(fracs, orig, orig_err_base, orig_err_mod, other_err_mod, diff):
    targets = []
    for frac in fracs:
        mod = orig + frac * diff
        mod_err = orig_err_base + np.sqrt(
            (orig_err_mod * (1 - frac)) ** 2 + (other_err_mod * frac) ** 2
        )
        mods = {"data": mod, "err": mod_err}
        targets.append(mods)
    return targets


def make_target_fracs(
    fracs, base_session, stim_period_onset, stim_period_duration, alternate=False
):
    # stim_period_onset = 70
    # stim_period_duration = 375 - stim_period_onset
    fit_data = ssn.prepare_fit_data()
    on_diff, off_diff = make_on_off_diffs(
        fit_data, stim_period_onset, stim_period_duration
    )
    on_err_g, off_err_g = make_on_off_errs(
        fit_data, "G", stim_period_onset, stim_period_duration
    )
    on_err_h, off_err_h = make_on_off_errs(
        fit_data, "H", stim_period_onset, stim_period_duration
    )
    # make a bunch of diffs
    if base_session == "G":
        if alternate:
            targets = make_targets(
                fracs, fit_data["G"]["data"], on_err_g, off_err_g, off_err_h, off_diff
            )
        else:
            targets = make_targets(
                fracs, fit_data["G"]["data"], off_err_g, on_err_g, on_err_h, on_diff
            )
    elif base_session == "H":
        targets = make_targets(
            fracs, fit_data["H"]["data"], on_err_h, off_err_h, off_err_g, -off_diff
        )
    else:  # complete interpolation, based on G
        orig = fit_data["G"]["data"]
        orig_err = fit_data["G"]["err"]
        other_err = fit_data["H"]["err"]
        diff = fit_data["H"]["data"] - fit_data["G"]["data"]
        targets = make_targets(fracs, orig, 0, orig_err, other_err, diff)

    return targets


def make_interp_stim(
    fracs, base_session, stim_period_onset, stim_period_duration, alternate=False
):
    # generate the E4 stim with interpolation
    # alternate changes the on/off periods. (use off for G-based)
    on_period, off_period = on_off_periods(stim_period_onset, stim_period_duration)
    gstim = ssn.load_predefined_stimulus("G")
    hstim = ssn.load_predefined_stimulus("H")
    diff_all = hstim - gstim

    if base_session == "G":
        base = gstim
        diff = np.zeros_like(gstim)
        if alternate:
            diff[off_period] = gaussian_filter(hstim[off_period] - gstim[off_period])
        else:
            diff[on_period] = gaussian_filter(hstim[on_period] - gstim[on_period])
    elif base_session == "H":
        base = hstim
        diff = np.zeros_like(hstim)
        diff[off_period] = gaussian_filter(gstim[off_period] - hstim[off_period])
    else:
        base = gstim
        diff = diff_all

    # based on these, make interpolation for each fraction
    stims = []
    for frac in fracs:
        stims.append(base + frac * diff)
    return stims


def prepare_interp_data_for_fit(fracs, base_session, alternate=False):
    # prepare all information (data, err, stim) for one fit
    stim_period_onset = 70
    stim_period_duration = 375 - stim_period_onset
    data = make_target_fracs(
        fracs,
        base_session,
        stim_period_onset,
        stim_period_duration,
        alternate=alternate,
    )
    stims = make_interp_stim(
        fracs,
        base_session,
        stim_period_onset,
        stim_period_duration,
        alternate=alternate,
    )

    for i in range(len(fracs)):
        subzero = data[i]["data"] < 0
        if np.any(subzero):
            data[i]["data"][subzero] = 0
            # print('subzero data is set to 0')
    return data, stims

stim_period_onset = 70
stim_period_duration = 375 - stim_period_onset

# %%
if __name__ == "__main__":
    # %% prepare the data
    fit_data = ssn.prepare_fit_data()
    # %% make variants of the stimuli
    on_diff, off_diff = make_on_off_diffs(
        fit_data, stim_period_onset, stim_period_duration
    )
    on_err_g, off_err_g = make_on_off_errs(
        fit_data, "G", stim_period_onset, stim_period_duration
    )
    on_err_h, off_err_h = make_on_off_errs(
        fit_data, "H", stim_period_onset, stim_period_duration
    )
    # %% make a bunch of diffs
    fracs = [-0.50, -0.25, 0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
    gbase_targets = make_targets(
        fracs, fit_data["G"]["data"], off_err_g, on_err_g, on_err_h, on_diff
    )
    hbase_targets = make_targets(
        fracs, fit_data["H"]["data"], on_err_h, off_err_h, off_err_g, off_diff
    )

    plt.plot(gbase_targets[7]["data"])
    plt.plot(gbase_targets[7]["err"])

    # %%

    # fracs = [0.125, 0.25, 0.5, 1, 1.3, 2]
    fracs = [-0.25, 0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
    orig = fit_data["G"]["data"]
    fig, ax = plot_mod_data(fracs, orig, on_diff)
    plt.suptitle("Base: G, adding image response of H")
    plt.tight_layout()

    orig = fit_data["H"]["data"]
    fig, ax = plot_mod_data(fracs, orig, -off_diff)
    plt.suptitle("Base: H, adding non-image response of G")
    plt.tight_layout()

    orig = fit_data["H"]["data"]
    fig, ax = plot_mod_data(fracs, orig, -off_diff)
    plt.suptitle("Base: H, adding non-image response of G")
    plt.tight_layout()

    orig = fit_data["G"]["data"]
    fig, ax = plot_mod_data(fracs, orig, off_diff)
    plt.suptitle("Base: G, adding non-image response of H")
    plt.tight_layout()

    # %% testing the interpolation of the stims
    stims = make_interp_stim(fracs, "G", stim_period_onset, stim_period_duration)
    plt.plot(stims[7])

    # %%
    data, stim = prepare_interp_data_for_fit(fracs, "H")
    # data[3]["err"]
    plt.plot(data[6]["data"])

    # %% prepare base G and adding H's non-image response
    data, stim = prepare_interp_data_for_fit(fracs, "G", alternate=True)
    plt.plot(data[7]["data"])
    plt.plot(data[7]["err"])
    plt.plot(stim[5])


# %%
