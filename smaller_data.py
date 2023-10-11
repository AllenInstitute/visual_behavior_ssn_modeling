# %% compress feather file further.

import pandas as pd

# read the file.
df = pd.read_feather("figure_data/df_gen5.feather")

# save with more agressive compression.
df.to_feather("figure_data/df_gen5.feather")
# even more
df.to_feather(
    "figure_data/df_gen5_stronger.feather", compression="zstd", compression_level=22
)
# no. it doesn't change much...

# let's filter out unnecessary data.
df_compact = df[df["data_type"] != "Unused"].copy()
df_compact.reset_index(inplace=True, drop=True)
df_compact.to_feather("figure_data/df_gen5_used_subset.feather")


# %% gen4
df = pd.read_feather("figure_data/df_gen4.feather")

df.cost_pass_loose.value_counts()
df2 = df.copy()
# delete df2 columns which name ends with '_cost'
for col in df2.keys():
    if col.endswith("_cost"):
        print(col)
        del df2[col]
    if col.endswith("_adj"):
        print(col)
        del df2[col]
    if col in ["index", "penalty", "seed", "iteration", "umap_x", "umap_y"]:
        print(col)
        del df2[col]


# df2 = df[df["cost_pass_loose"] == True].copy()
df2.reset_index(inplace=True, drop=True)
df2.to_feather("figure_data/df_gen4_used_subset.feather", compression="zstd")
