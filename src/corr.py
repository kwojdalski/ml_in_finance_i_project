# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
out12 = pickle.load(open("data/05_model_input/train_df_merged.pkl", "rb"))
out12 = out12.loc[:, ~out12.columns.duplicated()]
ret_cols = [
    col for col in out12.columns if col.startswith("RET_") and col[4:].isdigit()
]
out12["INDUSTRY"].nunique()
out12["INDUSTRY_GROUP"].nunique()
out12["SECTOR"].nunique()
out12["SUB_INDUSTRY"].nunique()

# %%
# Filter data and calculate industry correlations
out13 = out12[ret_cols + ["INDUSTRY", "DATE"]]  # .query("DATE < 155")

# Calculate mean returns by date, industry and return period
industry_means = (
    out13.melt(id_vars=["DATE", "INDUSTRY"], value_vars=ret_cols)
    .pivot_table(
        values="value", index=["DATE", "variable"], columns="INDUSTRY", aggfunc="mean"
    )
    .reset_index()
    .sort_values(
        ["DATE", "variable"],
        key=lambda x: x.str.extract(r"(\d+)").astype(float).squeeze()
        if x.name == "variable"
        else x,
    )
)
corr_mat = industry_means.drop(["DATE", "variable"], axis=1).corr()

# Create a mask for correlations between 0.8 and 1 in absolute terms
# Create masks for high positive and negative correlations
# pos_mask = (corr_mat >= 0.8) & (corr_mat <= 1)
# Get the upper triangle of the correlation matrix (excluding diagonal)
upper_tri = np.triu(corr_mat, k=1)

# Convert to DataFrame for easier filtering
corr_df_filtered = pd.DataFrame(
    upper_tri, index=corr_mat.index, columns=corr_mat.columns
)

# Get pairs of highly correlated features
# Create a mask for correlations >= 0.8 in absolute value
mask = np.abs(corr_df_filtered) >= 0

# Get indices where mask is True
high_corr_indices = np.where(mask)

# Create DataFrame with the high correlations
high_corr_pairs = pd.DataFrame(
    {
        "Industry 1": corr_df_filtered.index[high_corr_indices[0]],
        "Industry 2": corr_df_filtered.columns[high_corr_indices[1]],
        "Correlation": corr_df_filtered.values[high_corr_indices],
    }
)

# Convert to DataFrame and sort by absolute correlation
high_correlations = pd.DataFrame(high_corr_pairs)
high_correlations = high_correlations.sort_values(
    "Correlation", key=abs, ascending=False
)
high_correlations.to_pickle("data/02_intermediate/high_correlations.pkl")


# %%
time_correlations = []

# Get unique dates
dates = pivoted_data["DATE"].unique()
pivoted_data = (
    pivoted_data.groupby(["DATE", "INDUSTRY", "variable"]).mean().reset_index()
)
# For each date and highly correlated pair, calculate correlation
for date in dates:
    date_data = pivoted_data[pivoted_data["DATE"] == date]

    for _, pair in high_correlations.iterrows():
        ind1 = pair["Industry 1"]
        ind2 = pair["Industry 2"]

        # Calculate correlation for this date and pair
        corr_ind1 = date_data[date_data["INDUSTRY"] == ind1]["value"].reset_index(
            drop=True
        )
        corr_ind2 = date_data[date_data["INDUSTRY"] == ind2]["value"].reset_index(
            drop=True
        )
        date_corr = corr_ind1.corr(corr_ind2)

        time_correlations.append(
            {
                "DATE": date,
                "Industry 1": ind1,
                "Industry 2": ind2,
                "Correlation": date_corr,
            }
        )

# Convert to DataFrame
time_corr_df = pd.DataFrame(time_correlations)

# Plot correlations over time for top 5 most correlated pairs


plt.figure(figsize=(15, 8))
for _, pair in high_correlations.head().iterrows():
    pair_data = time_corr_df[
        (time_corr_df["Industry 1"] == pair["Industry 1"])
        & (time_corr_df["Industry 2"] == pair["Industry 2"])
    ]
    plt.plot(
        pair_data["DATE"],
        pair_data["Correlation"],
        label=f"{pair['Industry 1']} vs {pair['Industry 2']}",
    )

plt.title("Correlation Over Time for Most Correlated Industry Pairs")
plt.xlabel("Date")
plt.ylabel("Correlation")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# %%
