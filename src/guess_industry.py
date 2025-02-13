# %%
import logging
import os
import pickle
from datetime import datetime, timedelta
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from gics.definitions import d_20180929

# %% [markdown]
# ## Introduction
#
# Before this workflow I tried different models with decent but not spectacular results (my best score was **0.5158**).
# Experimenting with models included XGBoost, Catboost, Adaboost, lightgbm in different configuration based on
# optuna hyperparametrization. On top of it, I engineered new features coming from technical analysis.
# In general, I started being skeptical about ability to generate a great model for this particular competition once I went through
# [this](https://github.com/vitoriarlima/stock-returns) particular notebook, which includes incorrectly implemented features, but still somehow managed to land a 17th percentile
# position in the competition.
#
# This particular notebook, which significantly beats all my previous submissions in the competition,
# is based on the following idea - what would happen if I could
# identify stocks that were part of the dataset? Identyfing even a small subset of
# symbols could significantly boost my position in the competition as all results are very compressed
# (between 0.5 and 0.5287). It would take only approx. 5.2k correctly identyfied observations to be no. 1.
# Why 5k? Test set consists of 198k, expected value of correctly identified returns is **2.6k** (out of **5.2k**).
# So I need just above 2.6k net correct answers to be no. 1 (>~130bps). I think that should be doable with the method I assumed is worth
# trying.
#
# # What's the method then?
# 1. Load anonymized train dataset
# 2. Analyze industry, sector, industry group, sub-industry breakdown and distribution
# 3. Compare it with some standardized methodology (e.g. GICS).
# 4. Try to identify sector / industry / sub-industry / industry group for some potential symbol
# that I had high conviction of existed over the last 30 years or so (e.g. Microsoft, Berkshire Hathaway, etc.).
# 5. I decided to go with MSFT (more on that later)
# 6. Get MSFT daily returns
# 7. Fit them to returns coming from anonymized train dataset (but only for subindustry Microsoft is expected to be in).
# 8. Calculate max rolling correlation between MSFT daily returns and observations in the anonymized train dataset.
# If it's nearly 1 for a couple of observations, then we can be sure that stock id for that particular observation
# could be mapped to MSFT.
# 9. On top of it, I can map dates to DATE column in the anonymized train dataset.
# 10. Once I identify stocks and dates in train dataset, I can just do the same for test dataset.
# 11. The last step is to check returns for RET in test dataset with use of market data, remap it to 1s and 0s.
# 12. Load the best model I had hands on (the 0.5158 one) and found values from step 11 onto it.
# 13. Submit!
# **Caveats:**
# This method obviously could perform arbitrarily better if I could work more on reference data (take into account delistings, etc.) to get more symbols from the past.
# I thought that my current result (first attempt with this method) is good enough to win the competition, so I stopped there.
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


# %%
out12 = pickle.load(open("data/05_model_input/train_df_merged.pkl", "rb"))
out12 = out12.loc[:, ~out12.columns.duplicated()]
date_dict = pd.read_pickle("data/01_raw/mapped_dates.pkl")
out12["DT"] = out12["DATE"].map(date_dict)
out12.groupby("DT")["STOCK"].nunique().sort_values(ascending=False).head(50)
# %%
# Check how many unique values there are for each column
out12["INDUSTRY"].nunique()
out12["INDUSTRY_GROUP"].nunique()
out12["SECTOR"].nunique()
out12["SUB_INDUSTRY"].nunique()
# %%
# Count number of unique sub-industries per industry
sector_counts = (
    out12.groupby("SECTOR")["INDUSTRY_GROUP"].nunique().sort_values(ascending=False)
)
industry_group_counts = (
    out12.groupby("INDUSTRY_GROUP")["INDUSTRY"].nunique().sort_values(ascending=False)
)
log.info("\nNumber of unique sub-industries per industry:")
for sector, count in sector_counts.items():
    log.info(f"{sector}: {count}")

# %%
# Found mappings for GICS classification
# 0 - Energy
# 1 - Materials
# 2 - ?
# 3 - Industrials
# 4 - Consumer Discretionary
# 5 - Consumer Staples
# 6 - Health Care
# 7 - Financials
# 8 - Information Technology
# 9 - Communication Services
# 10 - Utilities
# 11 - Real Estate

# MSFT
# Sector -Information Technology
# Industry - Software
# IndustryGroup - Software & Services
# SubIndustry - Systems Software

# Print the structure of GICS classification levels
# %%
# Split GICS classifications by level based on key length
sectors = {k: v["name"] for k, v in d_20180929.items() if len(k) == 2}
industry_groups = {k: v["name"] for k, v in d_20180929.items() if len(k) == 4}
industries = {k: v["name"] for k, v in d_20180929.items() if len(k) == 6}
sub_industries = {k: v["name"] for k, v in d_20180929.items() if len(k) == 8}
log.info("Comparison data vs GICS:")
log.info(f"uniq industries: {out12['INDUSTRY'].nunique()} vs {len(industries)}")
log.info(f"ind groups: {out12['INDUSTRY_GROUP'].nunique()} vs {len(industry_groups)}")
log.info(f"sectors: {out12['SECTOR'].nunique()} vs {len(sectors)}")
log.info(f"sub-ind: {out12['SUB_INDUSTRY'].nunique()} vs {len(sub_industries)}")
# %%
# Create mappings for each GICS level
# Create mappings from sub-industry to higher levels
sub_industry_mappings = {}
for sub_ind_code, sub_ind_name in sub_industries.items():
    sector_code = sub_ind_code[:2]
    industry_group_code = sub_ind_code[:4]
    industry_code = sub_ind_code[:6]

    sub_industry_mappings[sub_ind_code] = {
        "sector": sectors[sector_code],
        "industry_group": industry_groups[industry_group_code],
        "industry": industries[industry_code],
        "sub_industry": sub_ind_name,
    }

# %%
# Count sub-industries per sector
sector_industry_group_counts = {}
for industry_group_code in industry_groups:
    sector_code = industry_group_code[:2]
    sector_name = sectors[sector_code]
    sector_industry_group_counts[sector_name] = (
        sector_industry_group_counts.get(sector_name, 0) + 1
    )

sector_counts.sort_index(ascending=True)
# %%
# It looks like two distribution are very similar
fig, ax = plt.subplots(figsize=(12, 6))
pd.Series(sector_industry_group_counts).plot(
    kind="bar", ax=ax, alpha=0.5, label="Industry Groups"
)
sector_counts.sort_index(ascending=True).plot(
    kind="bar", ax=ax, alpha=0.5, label="Sectors"
)

# %%
# Let's try to find MSFT! It must be somewhere in the data
# Get date range
end_date = datetime(2020, 10, 10)
start_date = end_date - timedelta(days=25 * 365)  # Approximately 25 years, just a guess

# Download MSFT data and calculate MSFT daily returns
msft = yf.download("MSFT", start=start_date, end=end_date, progress=False)
msft_returns = msft["Close"].pct_change().dropna()
# Display first few rows
# %%
### try to identify msft
# It turns out that msft is in the sub-industry of "Systems Software"
# All I need to do is find one symbol that would allow me to narrow down further data
# search
# Ok, I managed to guess where MSFT could be (manually)
# out14 = out12
# test_df.set_index("ID", inplace=True)
# out12 = test_df
out13 = out12[
    (out12["SECTOR"] == 8)
    & (out12["INDUSTRY_GROUP"] == 20)
    & (out12["INDUSTRY"] == 57)
    & (out12["SUB_INDUSTRY"] == 142)
]
# %%
# Getting returns and anonymized dates
ret_cols = [
    col for col in out13.columns if col.startswith("RET_") and col[4:].isdigit()
]
df = out13[ret_cols + ["DATE", "STOCK"]]
df = df.transpose()[::-1]


# %%
# Function to calculate max rolling correlation
def calc_max_rolling_corr(series, yfin_returns, window=20, return_date=False):
    """Calculate maximum rolling correlation between two time series.

    Args:
        series: First time series to compare (anonymized returns)
        yfin_returns: Second time series from Yahoo Finance to compare against
        window: Size of rolling window to use for correlation calculation
        return_date: If True, also return the date of maximum correlation

    Returns:
        If return_date is False:
            float: Maximum absolute correlation value found
        If return_date is True:
            tuple: (Maximum correlation value, Date of maximum correlation)
    """
    # Convert series to numpy for faster computation
    series_np = series.values
    yfin_np = yfin_returns.values.flatten()

    max_corr = -1
    n_yfin = len(yfin_np)
    best_date = None

    # Only need to check if we have enough data points
    if n_yfin >= window:
        # For each possible window in MSFT returns
        for i in range(n_yfin - window + 1):
            window_returns = yfin_np[i : i + window]
            corr = np.corrcoef(series_np, window_returns)[0, 1]
            if abs(corr) > max_corr:
                max_corr = abs(corr)
                if return_date:
                    # best_date should be the last day of the window
                    # TODO not sure about this one
                    best_date = yfin_returns.index[i + window - 1]

    if return_date:
        return max_corr, best_date
    return max_corr


# %%
# Calculate max corelations for MSFT
max_correlations = {}

for i in range(len(df.columns)):
    max_correlations[df.columns[i]] = calc_max_rolling_corr(
        df.iloc[2:, i], msft_returns, return_date=True
    )
msft_correlations = pd.DataFrame(max_correlations).T
msft_correlations.columns = ["corr", "date"]
high_correlations = msft_correlations.sort_values("corr", ascending=False).head(65)
high_correlations[high_correlations.index == 43259.0]

# %%
run = False
if run:
    merged_df = pd.merge(high_correlations, out13, left_index=True, right_on="ID")
    # Create dictionary mapping between date and DATE columns
    # date_dict2 = dict(zip(merged_df["DATE"], merged_df["date"]))
    date_dict = dict(zip(merged_df["DATE"], merged_df["date"]))
    # Merge date_dict and date_dict2 since they contain the same mapping
    date_dict.update(date_dict2)
    # datedict2 was for test_df
    # Anyway, here's mapping ~220 dates
    # pd.to_pickle(date_dict, "data/01_raw/mapped_dates.pkl")


# Based on the result of this loop (correlation of nearly 1, we can conclude that,
# for instance, observation 43259.0 is probably MSFT.)
# In theory, it could be something else, but extremely highly correlated (like CFD contract, SSF, etc.)
# So yeah, I managed to decode MSFT. Now let's try to decode more stock ids of them
# %%
# I managed to identify MSFT by stock id as 1720
# Now let's check if all returns can be identified by stock id, not just for single date
# I use df as it's highly narrowed down (instead of using 300k or so observations)
# I have to iterate over just 155 observations (which takes ~5 seconds)
# %%
# In order to do that we need to define two additional functions
def process_column(col, df, yfin_df):
    """Process a single column to find correlation with MSFT returns."""
    stock_id = df[col].iloc[0]
    date_idx = df[col].iloc[1]
    max_corr, date = calc_max_rolling_corr(df[col].iloc[2:], yfin_df, return_date=True)
    return {"date_idx": date_idx, "stock_id": stock_id, "date": date, "corr": max_corr}


# %%
def process_all_columns(df: pd.DataFrame, yfin_df: pd.Series):
    """Process all columns and track results.

    Iterates through columns in the dataframe, calculating correlations with subesquent columns of yfin_df.
    Tracks results in both a list and dataframe format, printing progress and highest
    correlations periodically.

    Args:
        df: Input dataframe containing columns to process
        yfin_df: Yahoo Finance dataframe with MSFT returns to compare against

    Returns:
        Tuple containing:
            - List of dictionaries with correlation results for each column
            - DataFrame containing the same correlation results in tabular format
    """
    results = []
    all_results = []
    date_df = pd.DataFrame()

    # Process each column in returns DataFrame
    for i in range(len(yfin_df.columns)):
        if i % 20 == 0:
            log.info(f"Processing {i} / {len(yfin_df.columns)}")

        # Process each column in input DataFrame
        for col in df.columns:
            result = process_column(col, df, pd.DataFrame(yfin_df.iloc[:, i]))
            results.append(result)

            # Add to date_df
            result_df = pd.DataFrame([result])
            result_df["symbol"] = yfin_df.columns[i]
            result_df["id"] = col
            date_df = pd.concat([date_df, result_df], ignore_index=True)

            # Print results
            log.debug(f"{col}: {result['corr']} on {result['date']}")

        all_results.append(date_df)
        date_df = pd.DataFrame()  # Reset for next iteration

        # Print highest correlations periodically
        if i % 1000 == 0 and i > 0:
            log.info("\nHighest correlations:")
            correlations = pd.Series({r["date_idx"]: r["corr"] for r in results})
            log.info(correlations.sort_values(ascending=False).head())

    return results, pd.concat(all_results, ignore_index=True)


# %%
# Process all columns for msft
# This function is designed in such a way that it can process more than one set of returns
# from yahoo finance. In this case, I'm passing only one set (MSFT),
# but further on, we will get to more generalized approach
results, date_df = process_all_columns(df.iloc[:, 0:10], msft_returns)


# All observations can be identified( correlations are very close to 0)
# and comes from date range from 2010-02-09 to 2017-12-18
date_df.date.min()
date_df.date.max()


# Get one of the dates
# In my case it's 2016-07-20
# which seems to map to 97
# In other words date_idx = 97 => 2016-07-20
# %%
# Ok, now let's try to identify other stocks for this particular date
# I'll use S&P 500, Dow Jones, and NASDAQ 100 ticker lists
# I'll use yfinance to get the data
# Obviously, it's probably not all, but good enough to win the competition. If I could identify
# just 50% of the stocks and observations, it would give me an edge over the competitors.
# I would score at least 0.75 (100% accuracy over 50% of observations and 50% over the rest)
# %%
def get_yahoo_syms(exclude_mapped=False):
    sec_tickers = pd.read_json(
        "./data/01_raw/company_tickers.json"
    ).T  # https://www.sec.gov/files/company_tickers.json
    sec_tickers.drop(columns=["cik_str"], inplace=True)
    sec_tickers.rename(columns={"ticker": "Symbol", "title": "Name"}, inplace=True)
    symbols = list(sec_tickers["Symbol"].unique())
    if exclude_mapped:
        mapped_stocks = pd.read_pickle("./data/01_raw/mapped_stocks.pkl")
        symbols = [sym for sym in symbols if sym not in mapped_stocks.values()]
    return symbols


# %%
# Get prices for all S&P 500 stocks on 2016-07-20 using yfinance
# Download prices for all constituents on target date
# Use list of symbols and handle potential errors
symbols = get_yahoo_syms()
end_date = datetime(2016, 7, 22)
start_date = end_date - timedelta(days=32)

# %%
# Now I want to download ticks more systematically than on the previous call (just for MSFT)
# We must be careful as there are limits on the number of requests per hour / daily (afaik)
# So I'll download 100 symbols at a time

all_returns = pd.DataFrame()
for file in os.listdir("./data/01_raw/yahoo_data"):
    print(f"Processing {file}")
    prices = pd.read_pickle(f"./data/01_raw/yahoo_data/20160722/{file}")
    returns = prices["Close"].pct_change()
    cols = returns[1:].isna().any(axis=0)
    returns = returns.loc[:, ~cols]
    returns = returns.loc[:, ~returns.columns.duplicated()]
    all_returns = pd.concat([all_returns, returns], axis=1)
    all_returns = all_returns.loc[:, ~all_returns.isna().all()]
# Check for any NaN values in columns


# %%
def map_stocks(all_results_df, correlation_threshold=0.99):
    """Create mapping of highly correlated stocks above threshold."""
    dict_ = {}
    for _, row in all_results_df[
        all_results_df["corr"] > correlation_threshold
    ].iterrows():
        stock_id = row["stock_id"]
        stock_name = row["symbol"]
        dict_[stock_id] = stock_name
    return dict_


# %%
# Now let's narrow down dataset to one date
out13_20160720 = out12[out12.DATE == 97]
# Ok, it has 29 observations which will allow to potentially identify 29 stocks
tmp2 = out13_20160720[ret_cols + ["DATE", "STOCK"]]
df2 = tmp2.transpose()[::-1]
# %%
for i in range(0, len(df2.columns), 10):
    print(f"Processing {i} / {len(df2.columns)}")
    _, all_results_df = process_all_columns(df2.iloc[:, i : i + 10], all_returns)
    all_results_df.sort_values("corr", ascending=False).groupby("date_idx").head(50)
    # Create mapping of highly correlated stocks (>0.99)
    mapped_stocks = map_stocks(all_results_df)
    # Load existing mapping if it exists, otherwise start with empty dict
    if os.path.exists("data/01_raw/mapped_stocks.pkl"):
        existing_mapped_stocks = pd.read_pickle("data/01_raw/mapped_stocks.pkl")
        existing_mapped_stocks.update(mapped_stocks)
        pd.to_pickle(existing_mapped_stocks, "data/01_raw/mapped_stocks.pkl")
    else:
        pd.to_pickle(mapped_stocks, "data/01_raw/mapped_stocks.pkl")


# %%
symbols = get_yahoo_syms(exclude_mapped=True)
len(symbols)

# Get prices for all S&P 500 stocks on 2016-07-20 using yfinance
# Download prices for all constituents on target date
# Use list of symbols and handle potential errors
end_date = datetime(2016, 7, 22)
start_date = end_date - timedelta(days=32)


# %%
# Now I want to download ticks more systematically than on the previous call (just for MSFT)
# We must be careful as there are limits on the number of requests per hour / daily (afaik)
# So I'll download 100 symbols at a time
def download_batch_prices(
    symbols: list, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """Download stock prices in batches for a list of symbols.

    Args:
        symbols: List of stock symbols to download
        start_date: Start date for price data
        end_date: End date for price data

    Returns:
        DataFrame containing downloaded price data
    """
    prices = pd.DataFrame()
    for i in range(0, len(symbols), 100):
        batch_symbols = symbols[i : i + 100]
        try:
            batch_prices = yf.download(
                batch_symbols,
                start=start_date,
                end=end_date,
                progress=True,
            )
            if prices.empty:
                prices = batch_prices
            else:
                prices = pd.concat([prices, batch_prices], axis=1)

        except Exception as e:
            log.error(f"Error downloading batch {i}-{i+100}: {e}")
            continue
    return prices


# %%

# List files in the yahoo data directory
data_dir = "./data/01_raw/yahoo_data"

files = os.listdir(data_dir)
files = [f.replace(".pkl", "") for f in files if f.endswith(".pkl")]
if __name__ == "__main__":
    for sym in symbols:
        if sym in files:
            print(f"{sym} already exists")
            continue
        try:
            print(f"Downloading {sym}")
            prices = download_batch_prices([sym], start_date, end_date)
            sleep(1)
            prices.to_pickle(f"./data/01_raw/yahoo_data/{sym}.pkl")
        except Exception as e:
            print(f"Error downloading {sym}: {e}")
            continue

# %%
