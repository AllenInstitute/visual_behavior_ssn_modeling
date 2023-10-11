# %% Demo of model run

import SSN_model_funcs as ssn
import support_funcs as sf
import pandas as pd

# stimulus is a 1D array with arbitrary length.
# In this program, the length of the stimulus determines the length of the simulation.
stimulus = ssn.load_predefined_stimulus("G")
print(f"stimulus length is {len(stimulus)}.")


# for this simulation, I'll pick up parameters from one of the rows in
# df_gen4_used_subset.feather
df = pd.read_feather("figure_data/df_gen4_used_subset.feather")

# pick the first row with selected == True
selected = df.query("selected == True").iloc[0]

# simulation parameters are defined in support_funcs.py
params = selected[sf.sim_params]

# show all of the params.
print(params)

# Run the simulation with these parameters.
# the result is (len(stimulus), 4) array, representing 4 cell types.
result = ssn.sim_to_result_orig(stimulus, **params)
# if you want to benchmark
# %timeit ssn.sim_to_result_orig(stimulus, **params)

# plot the results of the activity
t_index = range(0, len(stimulus) * 2)
# plot results per cell types
fig, axs = ssn.plot_results_ct(result["output"], None, None, t_index)

# save the figure
fig.savefig("demo_result.png")
