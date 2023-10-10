# a collection of functions necessary for running the SSN model
from numba import njit, jit
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from cycler import cycler
import model_data
import iminuit
import iminuit.cost
from scipy.stats import loguniform
import pickle
from scipy import integrate
import scipy


new_scheme = True  # apply new scheme of the fitting.
# The new scheme uses mean of the influence matrix (instead of median) and error from the data
# as well as the influence matrix error for cost function


# sys.path.append("..")  # Adds upper directory to python modules path.
# import src.utils as utils

# some parameters are set here
# These parameters are fixed in the current framework, but can be ported into functions
# to make them variables
default_start = 750 * 9  # fit starting time in ms
default_end = 750 * 18  # fit ending time in ms
exponents = np.array([2.0, 2.0, 2.0, 2.0])  # nonlinearity in the SSN model
dt = 0.001  # simulation time resolution


if new_scheme:
    influence_matrix = np.transpose(model_data.l23_infl_matrix_mean)  # new scheme
    influence_matrix_fracerr = np.transpose(model_data.l23_infl_matrix_fracerr)

    # influence matrix for the stimulation
    stim_infl = model_data.e4_l23_infl_matrix_mean
    stim_infl_fracerr = model_data.e4_l23_infl_matrix_fracerr
else:
    influence_matrix = np.transpose(
        model_data.influence_matrix[0]
    )  # 0: L2/3, old scheme


time_index = np.array(range(default_end))

init_state = np.array([3.06699372, 5.34753103, 3.7317455, 7.75074964])  # familiar
# init_state = np.array([3.26140733, 5.33929936, 4.1193868, 3.02922993])  # novel


# For visualization
line_cols = {  # copied form project_colors in util
    "Sst-IRES-Cre": (158 / 255, 218 / 255, 229 / 255),
    "sst": (158 / 255, 218 / 255, 229 / 255),
    "Slc17a7-IRES2-Cre": (255 / 255, 152 / 255, 150 / 255),
    "slc": (255 / 255, 152 / 255, 150 / 255),
    "Vip-IRES-Cre": (197 / 255, 176 / 255, 213 / 255),
    "vip": (197 / 255, 176 / 255, 213 / 255),
}
color_list = [line_cols["slc"], "mediumaquamarine", line_cols["sst"], line_cols["vip"]]
mpl.rcParams["axes.prop_cycle"] = cycler(color=color_list)


# some utility variables


@njit(parallel=False)
def relu(array):
    return array * (array > 0)


@njit(parallel=False)
def step(state, ext_input, mat, scale, exponents, decay_constants, dt):
    """Execute One step of the SSN model"""
    input = relu(np.dot(mat, state) * scale + ext_input * dt * 1000) ** exponents
    dr = (-state + input) / decay_constants * dt
    # dx = (np.matmul(mat, state) * scale + ext_input) * dt
    # print(dx)
    return relu(state + dr)


def normalize_trace(trace, time_range=None):
    """calculate z-score of a single trace"""
    if time_range is None:
        mean = np.mean(trace)
        std = np.std(trace)
    else:
        mean = np.mean(trace[time_range])
        std = np.std(trace[time_range])

    if std == 0:
        std = 1  # avoid errors
    return (trace - mean) / std


def normalize_traces(traces, time_range=None):
    """calculate z-score of each trace"""
    if time_range is None:
        mean = np.mean(traces, axis=0)
        std = np.std(traces, axis=0)
    else:
        mean = np.mean(traces[time_range, :], axis=0)
        std = np.std(traces[time_range, :], axis=0)

    std[std == 0] = 1  # avoid errors.
    return (traces - mean) / std


@njit(parallel=False)
def simulate(state, ext_inputs, *args):
    # state is initial state
    # ext_inputs is profile of external inputs. (n_step, 4) matrix
    n_steps = ext_inputs.shape[0]
    results = np.zeros((n_steps, 4))
    for t in range(n_steps):
        state = step(state, ext_inputs[t, :], *args)
        results[t, :] = state
    return results


# This works, but is very slow. (~800 ms per simulation, compared to 8 ms for the original)
# so let's use the original one...
def simulate_scipy(state, ext_inputs, *args):
    # instead of calling steps, let's do scipy integration.
    # args: mat, scale, exponents, decay_constants, dt
    n_steps = ext_inputs.shape[0]
    mat, scale, exponents, decay_constants, dt = args

    # ext_inputs is not usable as it is, so need to be converted to a function
    t = np.linspace(0, dt * (n_steps - 1), n_steps)
    ext_inputs_func = scipy.interpolate.interp1d(t, ext_inputs, axis=0)

    # t = np.linspace(0, dt * (n_steps - 1), n_steps)
    def step_scipy(t, state):
        input = (
            relu(np.dot(mat, state) * scale + ext_inputs_func(t) * dt * 1000)
            ** exponents
        )
        dr_dt = (-state + input) / decay_constants
        return dr_dt

    result = integrate.solve_ivp(
        step_scipy,
        (0, dt * (n_steps - 1)),
        state,
        method="RK45",
        t_eval=np.linspace(0, dt * (n_steps - 1), n_steps),
    )
    return result


def exp_decay(tau, dt, steps):
    t = np.linspace(0, dt * (steps - 1), steps)
    return np.exp(-t / tau)


def make_stimulation(
    baseline_input, e_amplitude, pv_amplitude, decay_ratio, offset_ratio, dt
):
    stim_steps = int(0.250 / dt)  # could be a problem if this is not true integer.
    blank_steps = int(0.500 / dt)  # ms
    total_steps = stim_steps + blank_steps
    stimulation = np.tile(np.expand_dims(baseline_input, axis=0), (total_steps, 1))

    decay_func_on = exp_decay(decay_ratio, dt, stim_steps)
    off_amp = max(offset_ratio, np.exp(-0.25 / decay_ratio))
    decay_func_off = off_amp * exp_decay(decay_ratio, dt, blank_steps)
    decay_func = np.concatenate((decay_func_on, decay_func_off))

    e_stim = e_amplitude * decay_func
    pv_stim = pv_amplitude * decay_func

    stimulation[:, 0] += e_stim
    stimulation[:, 1] += pv_stim

    return stimulation


def make_stimulation_simple(baseline_input, e_amplitude, pv_amplitude, decay_ratio, dt):
    # simpler version of the stimulus that does not care about offset stimulation
    # use it first to reduce the number of parameters
    steps = int(0.750 / dt)
    stimulation = np.tile(np.expand_dims(baseline_input, axis=0), (steps, 1))
    decay_func = exp_decay(decay_ratio, dt, steps)
    e_stim = e_amplitude * decay_func
    pv_stim = pv_amplitude * decay_func

    stimulation[:, 0] += e_stim
    stimulation[:, 1] += pv_stim
    return stimulation


def make_omission_stim(
    baseline_input, e_amplitude, pv_amplitude, decay_ratio, offset_ratio, dt
):
    # this is continuation of the previous one. starting point would be adjusted.
    # steps = int(0.750 / dt)
    # stimulation = np.tile(np.expand_dims(baseline_input, axis=0), (steps, 1))
    # decay_func = exp_decay(decay_ratio, dt, steps)
    # # deduct some ratio

    # stimulation[:, 0] += e_stim
    # stimulation[:, 1] += pv_stim

    # initial value depends on the relative strength between the offset stim.

    off_amp = max(offset_ratio, np.exp(-0.25 / decay_ratio))
    stimulation = make_stimulation_simple(
        baseline_input,
        # TODO: fix this!
        e_amplitude * np.exp(-0.5 / decay_ratio) * off_amp,
        pv_amplitude * np.exp(-0.5 / decay_ratio) * off_amp,
        decay_ratio,
        dt,
    )
    return stimulation


def make_stimulation_series(n_rep, omit, *args):
    # n_rep: number of repetitions
    # omit: list of omission trials
    ext_stims = []
    for i in range(n_rep):
        if np.isin(i, omit):  # omission trial
            stim = make_omission_stim(*args)
        else:
            stim = make_stimulation(*args)
        ext_stims.append(stim)
    return np.concatenate(ext_stims)


# @njit
def make_datadriven_stimulation(
    predefined_stim,
    conn_scale,
    input_e,
    input_p,
    input_s,
    input_v,
    stim_e,
    stim_p,
    stim_s,
    stim_v,
):
    """
    input for the SST and VIP are constant
    E and PV (RS and FS) can receive predefined_stim with a factor defined in parameters
    currently, the duration is fixed to the predefined_stim
    1/21/2022: Changed to inputs to 4 population, using the influence matrix from E4
    to each population.
    """

    # append 9 cycles at the beginning
    cycle3 = predefined_stim[: 750 * 3]
    appendum = np.concatenate([cycle3] * 3)
    stimdata_appended = np.concatenate([appendum, predefined_stim]) * conn_scale

    duration = len(stimdata_appended)
    stim = np.zeros((duration, 4))
    # inputargs = ["input_e", "input_p", "input_s", "input_v"]
    # for i, kw in enumerate(inputargs):
    #     stim[:, i] = kwargs[kw]
    stim[:, 0] = input_e
    stim[:, 1] = input_p
    stim[:, 2] = input_s
    stim[:, 3] = input_v

    # TODO: write this right
    stim[:, 0] += stimdata_appended * stim_infl[0] * stim_e
    stim[:, 1] += stimdata_appended * stim_infl[1] * stim_p
    stim[:, 2] += stimdata_appended * stim_infl[2] * stim_s
    stim[:, 3] += stimdata_appended * stim_infl[3] * stim_v

    return stim


def load_predefined_stimulus(type, double_omission=False, stim_ratio=1.0):
    d = model_data.l4_excitatory_activity[type]["mean"]
    # trim the data
    d_trim = d[0 : (default_end - default_start)]

    # append 9 cycles at the beginning
    # cycle3 = d_trim[: 750 * 3]
    # appendum = np.concatenate([cycle3] * 3)
    # d_final = np.concatenate([appendum, d_trim])
    if double_omission:
        d_trim = np.concatenate(
            [d_trim[:3750], np.array([d_trim[3750]] * 750), d_trim[3750:-750]]
        )

    return d_trim


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

        data_reformatted = data[:, 0 : (default_end - default_start)]
        data_reformatted = np.transpose(data_reformatted)
        data_err_reformatted = data_err[:, 0 : (default_end - default_start)]
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

        data_reformatted = data[:, 0 : (default_end - default_start)]
        data_reformatted = np.transpose(data_reformatted)
        data_array[type] = np.concatenate(
            (np.zeros_like(data_reformatted), data_reformatted)
        )
    return data_array


def plot_results_ct(
    result,
    data,
    ext_stims,
    t_index,
    start_time=default_start,
    end_time=default_end,
    figax=None,
    flash=True,
    color=None,
    legend=["Fit", "Data"],
):
    if figax is None:
        fig, ax = plt.subplots(2, 2, figsize=(15, 6))
    else:
        fig, ax = figax

    types = ["RS", "FS", "SST", "VIP"]
    for i, t in enumerate(types):
        ax[i // 2, i % 2].plot(result[:, i], color=color)
        if data is not None:
            ax[i // 2, i % 2].plot(t_index, data[:, i])
        ax[i // 2, i % 2].set_ylim(0, 25)
        ax[i // 2, i % 2].set_title(t)
        if data is not None:
            ax[i // 2, i % 2].legend(legend)
        ax[i // 2, i % 2].set_xlim(start_time, end_time)
        if flash:
            add_flash(ax[i // 2, i % 2])
    return fig, ax


def plot_results_one(
    result,
    data,
    ext_stims,
    t_index,
    start_time=default_start,
    end_time=default_end,
    type="RS",
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    types = ["RS", "FS", "SST", "VIP"]
    ind = np.where(np.array(types) == type)[0]
    ax.plot(result[:, ind])
    ax.plot(t_index, data[:, ind])
    ax.set_ylim(0, 25)
    ax.set_title(type)
    ax.legend(["Fit", "Data"])
    ax.set_xlim(start_time, end_time)
    add_flash(ax)
    return fig


def plot_results(
    result, data, ext_stims, t_index, start_time=default_start, end_time=default_end
):
    fig, ax = plt.subplots(2, 2, figsize=(15, 6))
    # ax[0, 0].plot(result[:, [0, 2, 3, 1]])
    ax[0, 0].plot(result)
    ax[0, 0].set_title("Model activity (Hz)")
    # ax[0, 0].legend(["E", "SST", "VIP", "PV"])
    ax[0, 0].legend(["E", "PV", "SST", "VIP"])
    ax[0, 0].set_ylim(0, 25)
    ax[0, 0].set_xlim(start_time, end_time)

    ax[0, 1].plot(t_index, data[:, :])
    ax[0, 1].set_title("Target data (Hz)")
    ax[0, 1].legend(["E", "PV", "SST", "VIP"])
    ax[0, 1].set_ylim(0, 25)
    ax[0, 1].set_xlim(start_time, end_time)

    # ax[1, 0].plot(t_index, data[:, :] - result[t_index, :][:, [0, 2, 3]])
    ax[1, 0].plot(t_index, data[:, :] - result[t_index, :])
    ax[1, 0].set_title("Residual (data - model)")
    ax[1, 0].legend(["E", "PV", "SST", "VIP"])
    ax[1, 0].set_ylim(-4, 4)
    ax[1, 0].set_xlim(start_time, end_time)

    # ax[1, 1].plot(ext_stims[:, [0, 2, 3, 1]])
    ax[1, 1].plot(ext_stims)
    ax[1, 1].set_title("External Input (xA (unit must be determined))")
    ax[1, 1].legend(["E", "PV", "SST", "VIP"])
    ax[1, 1].set_xlim(start_time, end_time)

    duration = result.shape[0]
    flash_times = np.arange(0, duration, 750)
    alpha = 0.10

    for axa in ax:
        for a in axa:
            # indicate stimulation period
            for flash_start in flash_times:
                # this is ad-hoc...
                if flash_start == 9750:
                    continue
                a.axvspan(
                    flash_start,
                    flash_start + 0.25 * 1000,
                    color="blue",
                    alpha=alpha,
                    zorder=-np.inf,
                )

            # indicate evaluation period
            a.axvline(x=start_time, color="gray", linestyle="--")
            a.axvline(x=end_time, color="gray", linestyle="--")
    return


def add_flash(ax, start=0, end=13500, color="blue", alpha=0.05):
    flash_times = np.arange(start, end, 750)
    # alpha = 0.05
    for flash_start in flash_times:
        # this is ad-hoc...
        if flash_start in (9750, 3000):
            continue
        ax.axvspan(
            flash_start,
            flash_start + 0.25 * 1000,
            color=color,
            alpha=alpha,
            zorder=-np.inf,
        )


# Functions from here would be


def get_normalization_scale(data, keys):
    return {k: {"mean": np.mean(data[k]), "std": np.std(data[k])} for k in keys}


def apply_normalization(data, norm_scale):
    result = copy.deepcopy(data)
    for k in norm_scale.keys():
        result[k] = (data[k] - norm_scale[k]["mean"]) / norm_scale[k]["std"]
    result["meta"]["normalization"] = "z-score"
    return result


def rescale_data(data, fr_range):
    result = copy.deepcopy(data)
    for k in fr_range.keys():
        offset = np.mean(fr_range[k])
        width = np.diff(fr_range[k]) / 2
        result[k] = (data[k] * width) + offset
    result["meta"]["normalization"] = "Mock FR (Hz)"
    return result


def cat_data(data1, data2):
    """concatenate two composite data, keeping most properties from data1."""
    data = copy.deepcopy(data1)
    label1 = data1["meta"]["composite_labels"]
    label2 = data2["meta"]["composite_labels"]
    data["time"] = np.concatenate([data1["time"], data2["time"] + 0.75 * len(label1)])
    data["meta"]["composite_labels"] = label1 + label2

    # for other things, if it contains Cre, they are data. so concatenate
    for k in data.keys():
        if "Cre" in k:
            data[k] = np.concatenate([data1[k], data2[k]])
    return data


def data_as_array(data, types):
    result = []
    for k in types:
        if "Cre" in k:
            result.append(data[k])
    return (data["time"], np.transpose(np.array(result)))


def pick_params(session_type, **kwargs):
    # pick the right parameters, and ignore the other session.
    if session_type == "familiar":
        pick = "fam"
        ignore = "nov"
    elif session_type == "novel":
        pick = "nov"
        ignore = "fam"
    else:
        raise (
            f"unknown session type {session_type}. Please pick from familiar or novel."
        )

    picked_params = {}
    for k in kwargs.keys():
        if k[-3:] == pick:
            # store without the suffix
            picked_params[k[:-4]] = kwargs[k]
        elif k[-3:] != ignore:
            # make sure you are not ignoring important one
            picked_params[k] = kwargs[k]

    return picked_params


def sim_to_combined_result(
    predefined_stim,  # unused. necessary for minuit...
    conn_scale,
    tau_e,
    tau_p,
    tau_s,
    tau_v,
    input_e_fam,
    input_p_fam,
    input_s_fam,
    input_v_fam,
    input_e_nov,
    input_p_nov,
    input_s_nov,
    input_v_nov,
    stim_e,
    stim_p,
    stim_s,
    stim_v,
    e_to_e,
    e_to_p,
    e_to_s,
    e_to_v,
    p_to_e,
    p_to_p,
    p_to_s,
    p_to_v,
    s_to_e,
    s_to_p,
    s_to_s,
    s_to_v_fam,
    s_to_v_nov,
    v_to_e,
    v_to_p,
    v_to_s,
    v_to_v,
):
    all_args = locals()
    param_set_fam = pick_params("familiar", **all_args)
    param_set_nov = pick_params("novel", **all_args)
    result_fam = sim_to_result(**param_set_fam)
    result_nov = sim_to_result(**param_set_nov)
    return np.concatenate((result_fam, result_nov), axis=1)


def sim_to_result(
    predefined_stim,  # unused. necessary for minuit...
    conn_scale,
    tau_e,
    tau_p,
    tau_s,
    tau_v,
    input_e,
    input_p,
    input_s,
    input_v,
    stim_e,
    stim_p,
    stim_s,
    stim_v,
    e_to_e,
    e_to_p,
    e_to_s,
    e_to_v,
    p_to_e,
    p_to_p,
    p_to_s,
    p_to_v,
    s_to_e,
    s_to_p,
    s_to_s,
    s_to_v,
    v_to_e,
    v_to_p,
    v_to_s,
    v_to_v,
):
    result = sim_to_result_orig(**locals())
    output = reformat_result(result["output"], time_index)
    return output


# admittedly, these functions are not optimal. They use global variables and redundant.
# I will need a solution for this.
def sim_to_result_orig(
    predefined_stim,  # unused. necessary for minuit...
    conn_scale,
    tau_e,
    tau_p,
    tau_s,
    tau_v,
    input_e,
    input_p,
    input_s,
    input_v,
    stim_e,
    stim_p,
    stim_s,
    stim_v,
    e_to_e,
    e_to_p,
    e_to_s,
    e_to_v,
    p_to_e,
    p_to_p,
    p_to_s,
    p_to_v,
    s_to_e,
    s_to_p,
    s_to_s,
    s_to_v,
    v_to_e,
    v_to_p,
    v_to_s,
    v_to_v,
    double_omission=False,
):
    """main function that calculates the simulation result."""
    baseline_input = np.array([input_e, input_p, input_s, input_v])

    if predefined_stim is None:
        stim_decay = stim_s  # these were renamed to make v4 simulation
        stim_offset = stim_v
        stimulation_params = (
            baseline_input,
            stim_e,
            stim_p,
            stim_decay,
            stim_offset,
            dt,
        )
        if double_omission:
            ext_stims = make_stimulation_series(18, [13, 14], *stimulation_params)
        else:
            ext_stims = make_stimulation_series(18, [13], *stimulation_params)
    else:
        # TODO: expand this for double omission and some sanity checks
        stimulation_params = (
            input_e,
            input_p,
            input_s,
            input_v,
            stim_e,
            stim_p,
            stim_s,
            stim_v,
        )
        ext_stims = make_datadriven_stimulation(
            predefined_stim, conn_scale, *stimulation_params
        )

    tau_neurons = np.array([tau_e, tau_p, tau_s, tau_v])
    # fmt: off
    infl_mat = np.array([[e_to_e, p_to_e, s_to_e, v_to_e],
                         [e_to_p, p_to_p, s_to_p, v_to_p],
                         [e_to_s, p_to_s, s_to_s, v_to_s],
                         [e_to_v, p_to_v, s_to_v, v_to_v]])
    # fmt: on
    infl_copy = np.copy(influence_matrix)
    sim_params = (infl_mat * infl_copy, conn_scale, exponents, tau_neurons, dt)
    result = simulate(init_state, ext_stims, *sim_params)
    return dict(output=result, stims=ext_stims)


def reformat_result(result, t_index, start_time=default_start, end_time=default_end):
    """chop the simulation so that it can be compared with the data"""
    use_index = (t_index >= start_time) & (t_index < end_time)
    result_slice = result[t_index[use_index], :]
    return result_slice  # result_slice[:, [0, 2, 3]]


def reformat_data(data, t_index, start_time=default_start, end_time=default_end):
    use_index = (t_index >= start_time) & (t_index < end_time)
    data_slice = data[use_index, :]
    return data_slice


def form_minuit(data, predefined_stim=None, data_y_err=None, penalty_coef=1.0):
    if predefined_stim is None:
        predefined_stim = np.zeros_like(data)
    if data_y_err is None:
        data_y_err = np.ones_like(data)

    least_squares = iminuit.cost.LeastSquares(
        predefined_stim, data, data_y_err, sim_to_result
    )

    init_param_set = dict(
        conn_scale=30.0,
        tau_e=0.05,
        tau_p=0.05,
        tau_s=0.05,
        tau_v=0.05,
        input_e=1.0,
        input_p=1.0,
        input_s=1.0,
        input_v=1.5,
        stim_e=5.0,
        stim_p=2.0,
        stim_s=1.0,
        stim_v=1.0,
    )

    # fmt: off
    infl_names = [
        ["e_to_e", "p_to_e", "s_to_e", "v_to_e"],
        ["e_to_p", "p_to_p", "s_to_p", "v_to_p"],
        ["e_to_s", "p_to_s", "s_to_s", "v_to_s"],
        ["e_to_v", "p_to_v", "s_to_v", "v_to_v"],
    ]
    if new_scheme:
        infl_names.append(["stim_e", "stim_p", "stim_s", "stim_v"])
    # fmt: on
    infl_dict = {}
    for i in range(len(infl_names)):
        for j in range(4):
            # infl_dict[infl_names[i][j]] = influence_matrix[i][j]
            infl_dict[infl_names[i][j]] = 1.0

    if new_scheme:
        ncon = iminuit.cost.NormalConstraint(
            list(infl_dict.keys()),
            list(infl_dict.values()),
            (list(influence_matrix_fracerr.flatten()) + stim_infl_fracerr)
            / np.sqrt(penalty_coef),
        )
        cost = least_squares + ncon
    else:
        cost = least_squares

    init_param_set.update(infl_dict)

    m = iminuit.Minuit(cost, **init_param_set)
    if new_scheme:
        m.limits["conn_scale"] = (1.0, 100.0)
        m.limits["stim_s"] = (0.1, 10.0)  # finite range for v4, 1 for v3
        m.limits["stim_v"] = (0.1, 10.0)
    else:
        m.limits["conn_scale"] = (30.0, 30.0)
        m.limits["stim_s"] = (0.01, 1.0)  # this is decay
        m.limits["stim_v"] = (0.0, 1.0)  # this is onset/offset ratio

    m.limits["tau_e"] = (0.01, 1.0)
    m.limits["tau_p"] = (0.01, 1.0)
    m.limits["tau_s"] = (0.01, 1.0)
    m.limits["tau_v"] = (0.01, 1.0)
    m.limits["input_e"] = (0.0, 10.0)
    m.limits["input_p"] = (0.0, 10.0)
    m.limits["input_s"] = (0.0, 10.0)
    m.limits["input_v"] = (0.0, 10.0)
    m.limits["stim_e"] = (0.1, 10.0)
    m.limits["stim_p"] = (0.1, 10.0)
    for i in range(len(infl_names)):
        for j in range(4):
            # l1 = 0.66 * influence_matrix[i][j]
            # l2 = 1.5 * influence_matrix[i][j]
            # lims = (min(l1, l2), max(l1, l2))  # remedy for negative values
            # m.limits[infl_names[i][j]] = lims
            m.limits[infl_names[i][j]] = (0.1, 10.0)
    m.limits["p_to_v"] = (1.0, 1.0)  # disable because it's 0
    return m


def lin_bounded_rn(low, high):
    return (high - low) * np.random.random() + low


def log_bounded_rn(low, high):
    return loguniform.rvs(low, high)


def randomize_fit(m):
    keys = m.values.to_dict().keys()
    for k in keys:
        if not m.fixed[k]:
            if m.limits[k][0] == 0.0:
                newval = lin_bounded_rn(*m.limits[k])
            else:
                newval = log_bounded_rn(*m.limits[k])
            m.values[k] = newval
    return m


def test_double_omission(params, type):
    """Double omission response test for the new simulation scheme."""
    pre_stim = load_predefined_stimulus(type)
    r = sim_to_result_orig(pre_stim, **params, double_omission=True)
    before_rates = r["output"][8899:8999, :]
    after_rates = r["output"][13399:13499, :]
    diffs = np.sqrt((((before_rates - after_rates) / before_rates) ** 2).mean())
    return diffs
