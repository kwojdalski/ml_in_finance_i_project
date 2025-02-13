# %%
t_df = out10["train_df_winsorized"]

ret_cols = [col for col in t_df.columns if col.startswith("RET_") and col[4:].isdigit()]
import pickle

out12 = pickle.load(open("data/05_model_input/train_df_merged.pkl", "rb"))
out12 = out12.loc[:, ~out12.columns.duplicated()]
# %%
# Filter data and calculate industry correlations
out13 = out12[ret_cols + ["INDUSTRY", "DATE"]].query("DATE < 155")

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
# Calculate autocorrelation for each industry
autocorr_results = []
industries = industry_means.columns[2:]  # Skip DATE and variable columns

for industry in industries:
    # Get the time series for this industry
    industry_data = industry_means.pivot(
        index="DATE", columns="variable", values=industry
    )

    # Calculate autocorrelation with 1-period lag
    autocorr = industry_data.apply(lambda x: x.autocorr(lag=1))

    # Get mean autocorrelation across all periods
    mean_autocorr = autocorr.mean()

    autocorr_results.append(
        {"Industry": industry, "Mean Autocorrelation": mean_autocorr}
    )

# Convert to DataFrame and sort by absolute autocorrelation
autocorr_df = pd.DataFrame(autocorr_results)
autocorr_df = autocorr_df.sort_values("Mean Autocorrelation", key=abs, ascending=False)

print("\nMost autocorrelated industries:")
print(autocorr_df.head())

industry_means

# %%
