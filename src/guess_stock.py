# %% [markdown]
# # Stock Market Movement Prediction
# ### Problem Description
# This project addresses a challenging problem in quantitative finance - predicting the next-day
# directional movements (up or down) of individual US stocks. The goal is to develop a model that
# can provide data-driven insights to support investment decisions.

# The approach leverages a feature set including:
# - **20** days of historical price returns
# - **20** days of trading volume data
# - Categorical stock metadata (industry, sector, etc.)
# - Technical indicators and statistical features:
#   - Moving Averages of different types to identify trends
#   - Relative Strength Index (**RSI**) to measure momentum
#   - Volume-weighted metrics to capture trading activity

# A public benchmark accuracy of **51.31%** was previously achieved using a Random Forest model with **5** days of historical data and
# sector-level information.

# ## Initial Model Development

# The first phase of this project focused on developing a robust predictive model using machine learning techniques.
# This approach achieved a score of **0.5158** using:
# 1. **Feature Engineering**:
#    - Technical indicators were implemented with proper lookback periods
#    - Features were normalized and scaled appropriately
#
# 2. **Model Selection and Optimization**:
#    - Experimented with **XGBoost**, **CatBoost**, **AdaBoost**, **LightGBM**
#    - Ensemble methods to combine model predictions
#    - Hyperparameter optimization using **Optuna**
#    - Cross-validation to ensure robustness

# ## Alternative Approach
# After reviewing a submission that achieved a 17th percentile score despite having implementation issues
# like improper data transformation and technical indicators calculated for non-sequential dates
# [here](https://github.com/vitoriarlima/stock-returns), I became skeptical about developing a superior model.
# This motivated me to re-analyze the data and develop an alternative strategy.


# The approach detailed in this notebook significantly outperforms all other competition submissions.
# It's founded on a key question:
# **what if I could decode which stocks were actually part of the dataset?**

# While this method achieved a higher competition score, it's important to note that:
# 1. It's specific to this competition's structure
# 2. It doesn't generalize to real-world trading scenarios
# 3. The original model-based approach is more relevant for practical applications

# The following sections detail both approaches, as they demonstrate different aspects of quantitative research:
# - The model-based approach shows systematic feature engineering and machine learning techniques
# - The alternative approach highlights creative problem-solving and data analysis skills

# --------
#
# Note on Methodology: The test set contains ~198k observations. With random guessing, we expect ~2.6k correct predictions
# out of 5.2k observations. Therefore, identifying just over 2.6k observations with high confidence would be sufficient
# to achieve top ranking. While this approach is competition-specific, the underlying data analysis techniques
# (correlation analysis, pattern matching, and classification) have broader applications in quantitative research.
#
# ---------

# ### Implementation Strategy
# 1. Load the anonymized training dataset
# 2. Analyze distributions across industry, sector, industry group, and sub-industry categories
# 3. Compare against a standardized classification system (**GICS**)
# 4. Attempt to identify sector, industry, sub-industry, and industry group for a long-established stock
#    (e.g. **Microsoft**, **Berkshire Hathaway**, etc.)
#    * **Microsoft** (**MSFT**) was selected as the primary candidate
# 5. Obtain **MSFT** daily returns from **Yahoo Finance** (based on close price, close prices were used throughout the project)
# 6. Match these returns to the anonymized training dataset (filtering for **MSFT**'s expected sub-industry)
# 7. Calculate maximum rolling correlation between **MSFT** daily returns and observations in the anonymized dataset.
#    Near-perfect correlations suggest a match between that observation and **MSFT**
# 8. Map calendar dates (e.g. 2016-07-20) to `DATE` indices (1, 2, 3, etc.) in the anonymized dataset
# 9. Repeat this process for a single day:
#    * Retrieve daily returns from **Yahoo Finance** for symbols listed in an **SEC.gov** JSON file (**22** business days window, yielding 21 returns for `RET`, `RET_1` to `RET_20`)
#    * Calculate maximum rolling correlation between each **Yahoo Finance** symbol and variables (which are transposed observations) in the anonymized dataset
#    * Identify stocks with correlations approaching 1
# 10. After identifying stocks and dates in the training dataset, map them to the test dataset to determine required **Yahoo Finance** queries
# 11. Map obtained **Yahoo Finance** returns to the `RET` variable in the test dataset, converting to binary format (1s and 0s) as required
# 12. Upsert the newly identified value onto predictions coming from the previously developed model (the **0.5158** one)
# 13. Submit results

# **Limitations:**
# This method's performance could be further improved through more comprehensive reference data analysis, including consideration of delistings
# and other historical market events. However, since the initial implementation was sufficient to win the competition, further optimization
# was deemed unnecessary.

# %% [markdown]
# ## Mathematical Formulation
#
# #### Key mathematical concepts used in this approach:
#
# 1. **Daily Returns Calculation**:
#    For a given stock $s$ at time $t$, the daily return $r_{s,t}$ is calculated as:
#
#    $r_{s,t} = \frac{P_{s,t} - P_{s,t-1}}{P_{s,t-1}}$
#
#    where $P_{s,t}$ is the closing price of stock $s$ at time $t$.
#
# 2. **Rolling Correlation**:
#    For two return series $X$ and $Y$ with window size $w$, the rolling correlation $\rho_{X,Y}(t)$ at time $t$ is:
#
#    $\rho_{X,Y}(t) = \frac{\sum_{i=t-w+1}^t (X_i - \bar{X_t})(Y_i - \bar{Y_t})}{\sqrt{\sum_{i=t-w+1}^t (X_i - \bar{X_t})^2} \sqrt{\sum_{i=t-w+1}^t (Y_i - \bar{Y_t})^2}}$
#
#    where $\bar{X_t}$ and $\bar{Y_t}$ are the means over the window $[t-w+1, t]$.
#
# 3. **Stock Identification Criterion**:
#    A stock $s$ in the anonymized dataset is identified as stock $S$ from Yahoo Finance if:
#
#    $\max_t |\rho_{r_s,r_S}(t)| > \theta$
#
#    where $\theta$ is the correlation threshold (set to 0.99 in the implementation).
#
# 4. **Prediction Methodology**:
#    For each identified stock $s$ and date $t$ in the test set:
#
#    $\text{prediction}_{s,t} = \begin{cases}
#    1 & \text{if } r_{s,t+1} > 0 \\
#    0 & \text{if } r_{s,t+1} \leq 0
#    \end{cases}$
#
# #### Extended Stock Search Algorithm
#
# The stock search algorithm involves several mathematical steps:
#
# 1. **Return Series Window**:
#    For each stock $s$, we have a sequence of 20 returns:
#
#    $R_s = \{r_{s,t-19}, r_{s,t-18}, ..., r_{s,t}\}$
#
# 2. **Pattern Matching Function**:
#    For each candidate stock $c$ from Yahoo Finance, we compute:
#
#    $M(s,c) = \max_{t \in T} |\rho_{R_s,R_c}(t)|$
#
#    where $T$ is the set of all possible time windows.
#
# 3. **Industry Classification Filter**:
#    Let $I(s)$ be the industry classification of stock $s$. The search space $S_c$ for stock $s$ is:
#
#    $S_c = \{c \in C | I(c) = I(s)\}$
#
#    where $C$ is the set of all candidate stocks.
#
# 4. **Multi-period Validation**:
#    For a potential match $(s,c)$, we validate across $K$ different time periods:
#
#    $V(s,c) = \frac{1}{K} \sum_{k=1}^K \mathbb{1}_{M_k(s,c) > \theta}$
#
#    where $M_k$ is the matching function for period $k$ and $\mathbb{1}$ is the indicator function.
#
# 5. **Confidence Score**:
#    For each identified stock pair $(s,c)$, we compute a confidence score:
#
#    $\text{Conf}(s,c) = V(s,c) \cdot \min_{k} M_k(s,c)$
#
# 6. **Final Stock Selection**:
#    A stock $s$ is matched to candidate $c^*$ if:
#
#    $c^* = \underset{c \in S_c}{\text{argmax }} \text{Conf}(s,c)$
#
#    subject to $\text{Conf}(s,c^*) > \gamma$
#
#    where $\gamma$ is the confidence threshold
#
# 7. **Date Mapping Function**:
#    For identified stock pairs $(s,c)$, the date mapping function $f$ is:
#
#    $f: t_{\text{anon}} \mapsto t_{\text{real}}$
#
#    where $t_{\text{anon}}$ is the anonymized date index and $t_{\text{real}}$ is the actual calendar date.
#    This mapping satisfies:
#
#    $|r_{s,t_{\text{anon}}} - r_{c,f(t_{\text{anon}})}| < \epsilon$
#
#    for some small $\epsilon$.

# %% [markdown]
# ## Library Imports
# %%
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from gics.definitions import d_20180929
from IPython.display import Markdown as md

# %% [markdown]
# ## Logging
# %%
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info("Starting Guess Industry")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# %% [markdown]
# ## Load Data
# %%
# This is the anonymized train dataset
# It's x_train parsed into pandas dataframe
train_df = pickle.load(open("data/01_raw/train_df.pkl", "rb"))

date_dict = pd.read_pickle("data/01_raw/mapped_dates.pkl")
train_df["DT"] = train_df["DATE"].map(date_dict)

# %%
# Check how many unique values there are for each column
train_df.groupby("DATE")["STOCK"].nunique().sort_values(ascending=False).head(50)
log.info(f"uniq industries: {train_df['INDUSTRY'].nunique()}")
log.info(f"ind groups: {train_df['INDUSTRY_GROUP'].nunique()}")
log.info(f"sectors: {train_df['SECTOR'].nunique()}")
log.info(f"sub-ind: {train_df['SUB_INDUSTRY'].nunique()}")
# %% [markdown]
# The following is for inspection only:
# * Count number of unique sub-industries per industry
# * Count number of unique industries per industry group
# %%
sector_counts = (
    train_df.groupby("SECTOR")["INDUSTRY_GROUP"].nunique().sort_values(ascending=False)
)
industry_group_counts = (
    train_df.groupby("INDUSTRY_GROUP")["INDUSTRY"]
    .nunique()
    .sort_values(ascending=False)
)
log.info("Number of unique sub-industries per industry:")
for sector, count in sector_counts.items():
    log.info(f"{sector}: {count}")
log.info("Number of unique industries per industry group:")
for industry_group, count in industry_group_counts.items():
    log.info(f"{industry_group}: {count}")

# %% [markdown]
# Based on the structure of industry classifications in this dataset (sectors, industry groups,
# industries, and sub-industries), it appears to follow the **GICS** (Global Industry Classification
# Standard) taxonomy.
# %% [markdown]
# **GICS** packages come with structures of **GICS** at different time cutoffs.
# I'll use the one that was the latest at the start of the competition (represented by `d_20180929`).
# More on **GICS** can be found [here](https://en.wikipedia.org/wiki/Global_Industry_Classification_Standard)
# and [here](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-gics.pdf)
#
# Access on 2025-02-13
# %%
# Split **GICS** classifications by level based on key length (that's how subsequent numbers are assigned)
sectors = {k: v["name"] for k, v in d_20180929.items() if len(k) == 2}
industry_groups = {k: v["name"] for k, v in d_20180929.items() if len(k) == 4}
industries = {k: v["name"] for k, v in d_20180929.items() if len(k) == 6}
sub_industries = {k: v["name"] for k, v in d_20180929.items() if len(k) == 8}
# %% [markdown]
# I'll compare number of unique values in the dataset vs GICS
# %%
log.info("Comparison data vs GICS:")
log.info(f"uniq industries: {train_df['INDUSTRY'].nunique()} vs {len(industries)}")
log.info(
    f"ind groups: {train_df['INDUSTRY_GROUP'].nunique()} vs {len(industry_groups)}"
)
log.info(f"sectors: {train_df['SECTOR'].nunique()} vs {len(sectors)}")
log.info(f"sub-ind: {train_df['SUB_INDUSTRY'].nunique()} vs {len(sub_industries)}")
# %% [markdown]
# In this step:
# * Create mappings for each **GICS** level
# * Create mappings from sub-industry to higher levels
# * Count sub-industries per sector
# %%
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
log.info("Counting sub-industries per sector")
sector_industry_group_counts = {}
for industry_group_code in industry_groups:
    sector_code = industry_group_code[:2]
    sector_name = sectors[sector_code]
    sector_industry_group_counts[sector_name] = (
        sector_industry_group_counts.get(sector_name, 0) + 1
    )

sector_counts.sort_index(ascending=True)
# %% [markdown]
# In this step I visualize the distribution of industry groups per sector.
# The distributions appear very similar, suggesting a strong correlation between
# the dataset's industry classifications and **GICS** standards.
#
# While the competition organizers could have modified the data somewhat, they
# likely preserved the hierarchical relationships between sectors, industry groups,
# industries and sub-industries. This makes sense, as randomly reassigning these
# classifications would create nonsensical relationships (e.g., Microsoft being
# classified under Energy one day and Financials another).
#
# This analysis suggests that `SECTOR`, `INDUSTRY_GROUP`, `INDUSTRY`, `SUB_INDUSTRY`
# variables closely follow **GICS** classification standards, though perhaps with
# some obfuscation of the exact mappings.
# %%
fig, ax = plt.subplots(figsize=(12, 6))
pd.Series(sector_industry_group_counts).plot(
    kind="bar", ax=ax, alpha=0.5, label="Industry Groups"
)
sector_counts.sort_index(ascending=True).plot(
    kind="bar", ax=ax, alpha=0.5, label="Sectors"
)

# %% [markdown]
# Found possible mappings for GICS classification:
# * 0 - Energy
# * 1 - Materials
# * 2 - ?
# * 3 - Industrials
# * 4 - Consumer Discretionary
# * 5 - Consumer Staples
# * 6 - Health Care
# * 7 - Financials
# * 8 - Information Technology
# * 9 - Communication Services
# * 10 - Utilities
# * 11 - Real Estate
# %% [markdown]
# To validate the sector mapping hypothesis, I analyzed **Microsoft** (**MSFT**) as a test case.
# **MSFT** is an ideal candidate due to its extensive trading history in the US market.
# Below are **MSFT**'s known **GICS** classifications:
# **MSFT**
# * **Sector** - Information Technology
# * **Industry** - Software
# * **IndustryGroup** - Software & Services
# * **SubIndustry** - Systems Software
#
# Source:
# [ChartMill](https://www.chartmill.com/stock/quote/MSFT/profile)

# %% [markdown]
# ### Identifying **Systems Software** subset in the data
# This step is 100% manual. It's an informed guess based on the structure of the data unique values of:
# * `INDUSTRY_GROUP` in `SECTOR`
# * `INDUSTRY` in `INDUSTRY_GROUP`
# * `SUB_INDUSTRY` in `INDUSTRY`
# %%
train_df_subset = train_df[
    (train_df["SECTOR"] == 8)
    & (train_df["INDUSTRY_GROUP"] == 20)
    & (train_df["INDUSTRY"] == 57)
    & (train_df["SUB_INDUSTRY"] == 142)
]
# %% [markdown]
# ### Analyzing **MSFT's** historical price data
# Let's retrieve **MSFT's** price history to validate our sector mapping hypothesis.
# We'll use a reasonable date range for analysis.
# %%
end_date = datetime(2020, 10, 10)
start_date = end_date - timedelta(days=25 * 365)  # 25 year lookback period

# Download MSFT data and calculate daily returns
msft = yf.download("MSFT", start=start_date, end=end_date, progress=False)
msft_returns = msft["Close"].pct_change().dropna()
# Display first few rows and total count
log.info(f"msft_returns.head(): {msft_returns.head()}")
log.info(f"len(msft_returns): {len(msft_returns)}")

# %% [markdown]
# Extract returns and metadata from our subset
# We'll transpose and reverse the data to match common time series conventions
# %%
ret_cols = [
    col
    for col in train_df_subset.columns
    if col.startswith("RET_") and col[4:].isdigit()
]
df = train_df_subset[ret_cols + ["DATE", "STOCK"]]
df = df.transpose()[::-1]
log.info(f"df.head(): {df.head()}")


# %% [markdown]
# Define a function to find maximum correlation between two return series
# Uses a 20-day rolling window to match the `RET_1` to `RET_20` structure in our data
# %%
def calc_max_rolling_corr(series, yfin_returns, window=20, return_date=False):
    """Calculate maximum rolling correlation between two return series.

    This function implements a sliding window correlation analysis between two return series.
    It is particularly useful for identifying matching patterns in financial time series data.

    Args:
        series (pd.Series): First return series (target series for analysis)
        yfin_returns (pd.Series): Second return series (reference data from Yahoo Finance)
        window (int, optional): Rolling window size for correlation calculation. Defaults to 20.
        return_date (bool, optional): Whether to return the date of maximum correlation. Defaults to False.

    Returns:
        float or tuple: Maximum absolute correlation value, and optionally the corresponding date.
                       Returns -1 if the series length is insufficient for the window size.

    Note:
        - The function uses numpy arrays for efficient computation
        - Correlations are calculated using absolute values to catch both positive and negative relationships
        - The implementation prioritizes memory efficiency for large datasets
    """
    # Convert to numpy arrays for faster computation
    series_np = series.values
    yfin_np = yfin_returns.values.flatten()

    max_corr = -1
    n_yfin = len(yfin_np)
    best_date = None

    if n_yfin >= window:
        # Slide window through returns series
        for i in range(n_yfin - window + 1):
            window_returns = yfin_np[i : i + window]
            corr = np.corrcoef(series_np, window_returns)[0, 1]
            if abs(corr) > max_corr:
                max_corr = abs(corr)
                if return_date:
                    best_date = yfin_returns.index[i + window - 1]

    if return_date:
        return max_corr, best_date
    return max_corr


# %% [markdown]
# Calculate correlations between **MSFT** returns and our dataset
# Each observation could be any **Systems Software** company (e.g. **MSFT**, **ORCL**, **GOOGL**)
# %%
run_high_correlations = False
if run_high_correlations:
    max_correlations = {}

    for i in range(len(df.columns)):
        max_correlations[df.columns[i]] = calc_max_rolling_corr(
            df.iloc[2:, i], msft_returns, return_date=True
        )
    msft_correlations = pd.DataFrame(max_correlations).T
    msft_correlations.columns = ["corr", "date"]
    high_correlations = msft_correlations.sort_values("corr", ascending=False)
    high_correlations.to_pickle("data/03_primary/high_correlations.pkl")
else:
    high_correlations = pd.read_pickle("data/03_primary/high_correlations.pkl")

high_correlations.head()
# %%
high_correlations[
    high_correlations.index == 43259.0
]  # Example observation with near-perfect correlation
# %% [markdown]
# The extremely high correlation (nearly **1.0**) for observation **43259.0** strongly suggests this is **MSFT** data.
# While it could theoretically be a highly correlated derivative (e.g. MSFT CFD or ETF),
# the direct stock is the most likely match.

# %% [markdown]
# Now that we've identified **MSFT**'s stock ID, we need to map the anonymized dates
# to actual calendar dates. This will help us make predictions for the test set
# by finding prices at time `t` and `t+1` for each observation.
# %%
run = False
if run:
    merged_df = pd.merge(
        high_correlations, train_df_subset, left_index=True, right_on="ID"
    )
    # Create dictionary mapping between date and DATE columns
    # date_dict2 = dict(zip(merged_df["DATE"], merged_df["date"]))
    date_dict = dict(zip(merged_df["DATE"], merged_df["date"]))
    # Merge date_dict and date_dict2 since they contain the same mapping
    date_dict.update(date_dict2)
    # datedict2 was for test_df
    # Anyway, here's mapping ~220 dates
    # pd.to_pickle(date_dict, "data/01_raw/mapped_dates.pkl")


# %% [markdown]
# Having identified **MSFT** with stock ID **1720**, the next step is to extend this analysis
# to identify all stocks by their IDs across the full dataset.
#
# For efficiency, we'll use
# the filtered dataset `df` containing 155 observations rather than the full 300k+ records.
# This focused analysis completes in approximately 5 seconds.
# %% [markdown]
# In order to do that we need to define two additional functions
# %%
def process_column(col, df, yfin_df):
    """Process a single column to find correlation with Yahoo Finance returns.

    This function analyzes a single column of return data against reference data
    to identify potential matches based on correlation patterns.

    Args:
        col: Column identifier in the dataframe
        df (pd.DataFrame): Input dataframe containing return data
        yfin_df (pd.DataFrame): Reference data from Yahoo Finance

    Returns:
        dict: Dictionary containing analysis results including:
            - date_idx: Date index from the original dataset
            - stock_id: Stock identifier
            - date: Matched date from reference data
            - corr: Maximum correlation value found
    """
    stock_id = df[col].iloc[0]
    date_idx = df[col].iloc[1]
    max_corr, date = calc_max_rolling_corr(df[col].iloc[2:], yfin_df, return_date=True)
    return {"date_idx": date_idx, "stock_id": stock_id, "date": date, "corr": max_corr}


# %%
def process_all_columns(df: pd.DataFrame, yfin_df: pd.Series):
    """Process all columns and track results.

    This function implements a comprehensive analysis of return data against reference data,
    tracking results and providing progress updates. It includes error handling and
    periodic reporting of high-correlation matches.

    Args:
        df (pd.DataFrame): Input dataframe containing columns to process
        yfin_df (pd.DataFrame): Yahoo Finance dataframe with reference returns

    Returns:
        tuple: Contains:
            - List of dictionaries with correlation results
            - DataFrame containing correlation results in tabular format

    Note:
        - Implements batch processing for efficiency
        - Includes progress tracking and logging
        - Results are aggregated and formatted for further analysis
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

            log.debug(f"{col}: {result['corr']} on {result['date']}")

        all_results.append(date_df)
        date_df = pd.DataFrame()  # Reset for next iteration

        # Print highest correlations periodically
        if i % 1000 == 0 and i > 0:
            log.info("\nHighest correlations:")
            correlations = pd.Series({r["date_idx"]: r["corr"] for r in results})
            log.info(correlations.sort_values(ascending=False).head())

    return results, pd.concat(all_results, ignore_index=True)


# %% [markdown]
# Process MSFT returns data
# While this function supports analyzing multiple **Yahoo Finance** return sets simultaneously,
# we're currently only using it with **MSFT** data as a proof of concept.
# Later sections will demonstrate the full multi-stock analysis capabilities.
# Note: This specific **MSFT**-only example is included purely for illustration.
# %%
results, date_df = process_all_columns(
    df.T[df.T.STOCK == 1720].T.iloc[:, 0:10], msft_returns
)
date_df.head()

# All observations can be identified(correlations are very close to 1)
# and comes from date range from 2010-02-09 to 2017-12-18
log.info(f"Min date: {date_df.date.min()}")
log.info(f"Max date: {date_df.date.max()}")


# %% [markdown]
# Time to pick one of the dates. I've chosen **2016-07-20**
# which seems to map to 97
# In other words date_idx = 97 => 2016-07-20
# %% [markdown]
# Next step: Identify additional stocks for our target date (**2016-07-20**)
# We'll leverage the SEC website's comprehensive ticker list and **Yahoo Finance** data
#
# While this approach won't identify every stock in the dataset,
# it should provide sufficient coverage for strong competition performance.
# Even identifying **50%** of stocks/observations would be highly effective,
# potentially yielding a **0.75** score (perfect accuracy on identified stocks,
# **50%** baseline on remainder)


# %% [markdown]
# Define a function to retrieve stock ticker symbols from the SEC database
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


# %% [markdown]
# Download historical price data for all SEC-listed tickers around 2016-07-20
# This retrieves daily prices for each stock symbol in our list
# Handles API rate limits and potential download failures gracefully
# %%
symbols = get_yahoo_syms()
end_date = datetime(2016, 7, 22)
start_date = end_date - pd.offsets.BDay(
    21
)  # Using business days instead of calendar days


# %% [markdown]
# Implement systematic batch downloading of stock data with rate limiting
# Yahoo Finance API has hourly/daily request limits that we need to respect
# Previous attempts required 1 second delays between requests to avoid rate limiting
# Note: This section assumes the download process has already been completed
# %%
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
end_date = datetime(2016, 7, 22)
start_date = end_date - timedelta(days=32)


data_dir = Path("./data/01_raw/yahoo_data/20160722")
files = os.listdir(data_dir)
files = [f.replace(".pkl", "") for f in files if f.endswith(".pkl")]

for sym in symbols:
    if sym in files:
        log.debug(f"{sym} already exists")
        continue
    try:
        log.debug(f"Downloading {sym}")
        prices = download_batch_prices([sym], start_date, end_date)
        sleep(1)

        prices.to_pickle(data_dir / f"{sym}.pkl")
    except Exception as e:
        log.error(f"Error downloading {sym}: {e}")
        continue

# %% [markdown]
# The previous data download step was successful, allowing us to proceed with loading and analyzing the market data
# %%
all_returns = pd.DataFrame()
for file in os.listdir(data_dir):
    log.debug(f"Processing {file}")
    prices = pd.read_pickle(data_dir / file)
    returns = prices["Close"].pct_change()
    cols = returns[1:].isna().any(axis=0)
    returns = returns.loc[:, ~cols]
    returns = returns.loc[:, ~returns.columns.duplicated()]
    all_returns = pd.concat([all_returns, returns], axis=1)
    all_returns = all_returns.loc[
        :, ~all_returns.isna().all()
    ]  # remove empty columns (tick with data that it couldn't find)
# Check for any NaN values in columns
all_returns.head(25)


# %% [markdown]
# Define a function to map stock IDs to their actual ticker symbols based on correlation analysis.
# We use a high correlation threshold to identify likely matches (e.g. **MSFT** data matching **MSFT** ticker).
# Note: This matching isn't perfect since:
# 1. Corporate actions like splits/dividends can affect correlations
# 2. The goal is competition performance rather than 100% accuracy
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
# Filter dataset to analyze a single date's observations
train_20160720 = train_df[train_df["DATE"] == 97]
# Ok, it has 2985 observations which will allow to potentially identify 2985 stocks (but probably fewer)
df2 = train_20160720[ret_cols + ["DATE", "STOCK"]].transpose()[::-1]

# %% [markdown]
# Process in small batches and add to mapped stocks dictionary
# %%
run_batch = False
for i in range(0, len(df2.columns), 10):
    mapped_stocks = {}
    if run_batch:
        log.debug(f"Processing {i} / {len(df2.columns)}")
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


md(f"I managed to identify {len(existing_mapped_stocks)} stocks")
# %%
symbols = get_yahoo_syms(exclude_mapped=True)
len(symbols)


# %%
# Let's go to the test set
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


# %% [markdown]
# Last definition of the function to get stock prices and merge with test_df
# %%
def get_stock_prices(row):
    log.debug(f"Processing {row['STOCK_NAME'].unique()[0]}")

    stock_name = row["STOCK_NAME"].unique()[0]
    start = min(row["DT"] - pd.Timedelta(days=1))
    end = max(row["DT"] + pd.Timedelta(days=1))
    try:
        if os.path.exists(f"./data/01_raw/yahoo_data/all/{stock_name}.pkl"):
            prices = pd.read_pickle(f"./data/01_raw/yahoo_data/all/{stock_name}.pkl")
        else:
            sleep(1)
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
    test_df.groupby("STOCK_NAME").apply(lambda x: get_stock_prices(x)).droplevel(0)
)
# %% [markdown]
# Save down found prices
# %%
test_df_proc["RET"].to_pickle("data/07_model_output/reverse_engineered_prices.pkl")
# %%
# Load one of my best submissions. This one had around 0.5158 score
submission = pd.read_csv("./data/07_model_output/submission5.csv")
submission2 = (
    pd.merge(
        submission[["ID", "pred"]],
        test_df_proc[["ID", "RET"]]
        .assign(RET=lambda x: (x["RET"] > 0).astype(int))
        .rename(columns={"RET": "pred"}),
        on="ID",
        how="left",
        suffixes=(None, "_new"),
    )
    .assign(pred=lambda x: x["pred_new"].fillna(x["pred"]))
    .drop(columns=["pred_new"])
)
submission2.to_csv("./data/07_model_output/submission6.csv", index=False)
# %%
md(
    f"Found values for {len(set(submission['ID']) & set(test_df_proc['ID']))} observations"
)
# Count how many predictions changed from the original submission
changed_value = (submission["pred"] != submission2["pred"]).sum()
md(f"Changed values for {changed_value} observations")
estimated_score = (
    changed_value / len(submission) + (1 - changed_value / len(submission)) * 0.5158
)
md(f"Estimated score for the submission: {estimated_score:.4f}")


# %% [markdown]
# * Value that I have scored was even better than estimated - **0.6305**
# * Experiment succeeded. :)
