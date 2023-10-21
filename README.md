# visual_behavior_ssn_modeling

Code repository for the SSN modeling work on Visual Behavior Neuropixels data. This
repository is made for a reference purpose, and not inteded to be run on a local machine.
To completely reproduce what has been done, it requires intense computation on a cluster
machine.

The data necessary for figure reproduction is included, and one should be able to reproduce all of the figure panels using the attached notebook.

## Requirements for using the code

Written in requirements.txt.

## Figure reproduction

This repository contains all the necessary data for reproducing the figures. The analysis is also largely done in the figure generation script, paper_figures_v5.py. For most, this file will be sufficient to understand what were done for the analysis. Other scripts are used to generate these data. The steps to generate the final data are detailed in the last section (Generating data files).

## Running simulation

model_run_demo.py script is a demo of running the model with a single set of parameters. This will be instructive to reuse the code for running the model.

## Generating data files

### SSN_cluster_fit_*.py

Function: Perform computation (designed to be run on a cluster computer)

Reads: Basic data for simulation

Writes: fit_results\*/result\*.pkl

Notes: File structure (homedir variable) needs to be adjusted to run properly. Single
argument can give a random number seed that differentiates each run.

### cluster_fit_summarize3.py, cluster_fit_summarize5.py, cluster_fit_summarize6.py

Function: Pack indivisual output from the cluster jobs to one file

Reads: fit_results\*/result\_\*.pkl

Writes: fit_result?.feather

### cluster_fit_analysis4_granular.py

Function: Summarize the raw output of the simulations

Reads: fit_result4.feather

Writes: fit4_all_cost_resp.hdf

### cluster_fit_analysis5_cost.py

Function: Summarize the raw output of the simulations for target data manipulation.

Reads: fit_result5.feather, fit_results5_r2.feather, fit_result6.feather

Writes: fit5_all_cost_resp.h5

### make_figure_data.py

Function: Summarize the output of the simulations, and save in a compact format for
figure generation.

Reads: fit4_all_cost_resp.hdf, fit5_all_cost_resp.h5

Writes: figure_data/df_gen4.feather, figure_data/df_gen5.feather

### smaller_data.py

Function: Make the file size smaller so that they can be uploaded to GitHub

Reads: figure_data/df_gen4.feather, figure_data/df_gen5.feather

Writes: figure_data/df_gen4_used_subset.feather, figure_data/df_gen5_used_subset.feather

### paper_figures_v5.py

Reads: figure_data/df_gen4.feather, figure_data/df_gen5.feather
Writes: images for figure panels
