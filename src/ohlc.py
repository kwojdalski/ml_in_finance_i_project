# %%
import logging

import pandas as pd

from src.qrt_stock_returns.utils import get_node_output, run_pipeline_node

# %%
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info("Starting XGBoost")

# %% [markdown]
# ### Loading Up Kedro Config
# Grab all our parameters from the config file
# * This has elements like our target variable and k-fold settings

# %%
out10 = get_node_output("handle_outliers_node")
out9 = get_node_output("transform_volret_features_node")


out10 = run_pipeline_node(
    "handle_outliers_node",
    {
        "train_df_transformed": out9["train_df_transformed"],
        "test_df_transformed": out9["test_df_transformed"],
        "params:outlier_threshold": 10,
        "params:outlier_method": "clip",
    },
)

# %%
# Example daily data (let's say this is the price for each day)

df = out10["train_df_winsorized"][
    [
        "RET_1",
        "RET_2",
        "RET_3",
        "RET_4",
        "RET_5",
        "RET_6",
        "RET_7",
        "RET_8",
        "RET_9",
        "RET_10",
        "RET_11",
        "RET_12",
        "RET_13",
        "RET_14",
        "RET_15",
        "RET_16",
        "RET_17",
        "RET_18",
        "RET_19",
        "RET_20",
    ]
]

df = df.transpose()
df = df + 1
df = df.cumprod()
df = df[::-1]


# %%
# Function to group daily data into OHLC-like structure
def create_ohlc_from_daily(df, group_size=4):
    ohlc_data = []

    for col in df.columns:
        # Step by group_size to avoid overlap
        for i in range(0, len(df) - group_size + 1, group_size):
            group = df[col].iloc[i : i + group_size]
            open_value = group.iloc[0]  # The first day in the group
            high_value = group.max()  # The maximum in the group
            low_value = group.min()  # The minimum in the group
            close_value = group.iloc[-1]  # The last day in the group

            # Append the new OHLC-like row to the result
            ohlc_data.append(
                {
                    "Open": open_value,
                    "High": high_value,
                    "Low": low_value,
                    "Close": close_value,
                    "Date_Start": df.index[i],
                    "Date_End": df.index[i + group_size - 1],
                    "Column": col,
                }
            )

    return pd.DataFrame(ohlc_data)


# %%
import talib

# Create the OHLC-like DataFrame
# Process in batches of 10k columns
ohlc_dfs = []
for i in range(0, df.shape[1], 10000):
    batch_df = create_ohlc_from_daily(df.iloc[:, i : i + 10000])
    print(i)
    ohlc_dfs.append(batch_df)
ohlc_df = pd.concat(ohlc_dfs, ignore_index=True)
# %%# Apply TA-Lib pattern recognition functions
pattern_functions = [
    talib.CDL2CROWS,
    talib.CDL3BLACKCROWS,
    talib.CDL3INSIDE,
    talib.CDL3LINESTRIKE,
    talib.CDL3OUTSIDE,
    talib.CDL3STARSINSOUTH,
    talib.CDL3WHITESOLDIERS,
    talib.CDLABANDONEDBABY,
    talib.CDLADVANCEBLOCK,
    talib.CDLBELTHOLD,
    talib.CDLBREAKAWAY,
    talib.CDLCLOSINGMARUBOZU,
    talib.CDLCONCEALBABYSWALL,
    talib.CDLDARKCLOUDCOVER,
    talib.CDLDOJI,
    talib.CDLDOJISTAR,
    talib.CDLDRAGONFLYDOJI,
    talib.CDLENGULFING,
    talib.CDLGRAVESTONEDOJI,
    talib.CDLHAMMER,
    talib.CDLHANGINGMAN,
    talib.CDLHARAMI,
    talib.CDLHARAMICROSS,
    talib.CDLHIGHWAVE,
    talib.CDLHIKKAKE,
    talib.CDLHOMINGPIGEON,
    talib.CDLIDENTICAL3CROWS,
    talib.CDLINNECK,
    talib.CDLINVERTEDHAMMER,
    talib.CDLKICKING,
    talib.CDLMARUBOZU,
    talib.CDLMATCHINGLOW,
    talib.CDLMORNINGDOJISTAR,
    talib.CDLMORNINGSTAR,
    talib.CDLONNECK,
    talib.CDLPIERCING,
    talib.CDLRICKSHAWMAN,
    talib.CDLRISEFALL3METHODS,
    talib.CDLSEPARATINGLINES,
    talib.CDLSHOOTINGSTAR,
    talib.CDLSHORTLINE,
    talib.CDLSPINNINGTOP,
    talib.CDLSTALLEDPATTERN,
    talib.CDLSTICKSANDWICH,
    talib.CDLTAKURI,
    talib.CDLTASUKIGAP,
    talib.CDLTHRUSTING,
    talib.CDLTRISTAR,
    talib.CDLUNIQUE3RIVER,
    talib.CDLUPSIDEGAP2CROWS,
    talib.CDLXSIDEGAP3METHODS,
]


# %% # Apply each pattern recognition function
pattern_results = pd.DataFrame()
for pattern_func in pattern_functions:
    pattern_name = pattern_func.__name__[3:]  # Remove 'CDL' prefix
    print(pattern_name)

    # Group by Column and apply pattern function to each group
    grouped_results = []
    for _, group in ohlc_df.groupby("Column"):
        result = pattern_func(
            group["Open"], group["High"], group["Low"], group["Close"]
        )
        grouped_results.append(result)

    # Concatenate results from all groups
    pattern_results[pattern_name] = pd.concat(grouped_results)
pattern_results.index = ohlc_df["Column"]
pattern_results["Date_Start"] = ohlc_df["Date_Start"]
pattern_results["Date_End"] = ohlc_df["Date_End"]
pattern_results.to_pickle("data/04_feature/pattern_results.pkl")

# %%
for col in pattern_results.columns:
    print(f"{col}: {pattern_results[col].unique()}")
# Filter out columns with less than 1 unique value
# Filter columns with more than 1 unique value
pattern_results = pattern_results[
    [col for col in pattern_results.columns if len(pattern_results[col].unique()) > 1]
]

# Select rows where values are non-zero and include 5 adjacent rows
non_zero_mask = (pattern_results.select_dtypes(include=["number"]) != 0).any(axis=1)
window_size = 5
expanded_mask = (
    pd.DataFrame(non_zero_mask)
    .rolling(window=window_size, center=True)
    .max()
    .fillna(False)
    .astype(bool)
)
pattern_results = pattern_results[expanded_mask[0]]

# %%
# Add pattern results to OHLC dataframe
# Normalize pattern values to -1, 0, 1
pattern_results = pattern_results.apply(
    lambda x: x.map(lambda y: -1 if y < 0 else (1 if y > 0 else 0))
)

pattern_results
