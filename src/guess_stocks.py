# %%
import logging
import os
import pickle
import time

import pandas as pd
import yfinance as yf

# %%
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info("Starting Guess Industry")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# %%
test_df = pickle.load(open("data/01_raw/test_df.pkl", "rb"))
# %%
date_dict = pd.read_pickle("data/01_raw/mapped_dates.pkl")
mapped_stocks = pd.read_pickle("data/01_raw/mapped_stocks.pkl")

test_df["DT"] = test_df["DATE"].map(date_dict)
test_df.head()
# %%
test_df["STOCK_NAME"] = test_df["STOCK"].map(mapped_stocks)
test_df.head()
# %%
# Reorder columns to put STOCK_NAME after STOCK and DT after DATE
cols = test_df.columns.tolist()
stock_idx = cols.index("STOCK")
date_idx = cols.index("DATE")

# Create new column order
new_cols = cols.copy()
new_cols.remove("STOCK_NAME")
new_cols.remove("DT")
new_cols.insert(stock_idx + 1, "STOCK_NAME")
new_cols.insert(date_idx + 1, "DT")

# Reorder dataframe columns
test_df = test_df[new_cols]
test_df.head()
test_df = test_df[~test_df["STOCK_NAME"].isna()]
test_df["DT_1BDAY_OFFSET"] = test_df["DT"] + pd.offsets.BDay(1)


# %%
def get_stock_prices(row):
    print("Processing", row["STOCK_NAME"].unique()[0])

    stock_name = row["STOCK_NAME"].unique()[0]
    start = min(row["DT"] - pd.Timedelta(days=1))
    end = max(row["DT"] + pd.Timedelta(days=2))
    try:
        if os.path.exists(f"./data/01_raw/yahoo_data/all/{stock_name}.pkl"):
            prices = pd.read_pickle(f"./data/01_raw/yahoo_data/all/{stock_name}.pkl")
        else:
            time.sleep(1)
            prices = yf.download(stock_name, start=start, end=end)["Close"].pct_change()
            pd.to_pickle(prices, f"./data/01_raw/yahoo_data/all/{stock_name}.pkl")
        prices.columns = ["RET"]
    except Exception as e:
        log.error(f"Error downloading {stock_name}: {e}")
        return None
    joined = pd.merge(
        row,
        prices,
        right_index=True,
        left_on=["DT_1BDAY_OFFSET"],
    )
    return joined


# %%
test_df_proc = (
    test_df.groupby("STOCK_NAME")
    .apply(  # [test_df["STOCK_NAME"] == "MSCI"]
        lambda x: get_stock_prices(x)
    )
    .droplevel(0)
)
# %%
test_df_proc["RET"].to_pickle("data/07_model_output/reverse_engineered_prices.pkl")
# %%
# yf.download("ZWS", start="2013-11-12", end="2013-11-16")["Close"].pct_change()
# %%

submission = pd.read_csv("./data/07_model_output/submission5.csv")
submission2 = pd.merge(
    submission[["ID", "pred"]],
    test_df_proc[["ID", "RET"]].assign(RET=lambda x: (x["RET"] > 0).astype(int)),
    on="ID",
    how="right",
)
submission2["RET"].unique()
submission2[submission2["ID"] == 526067]
# set(submission["ID"]) & set(test_df_proc["ID"])

test_df_proc["RET"]

# %%
submission.head()
# %%
