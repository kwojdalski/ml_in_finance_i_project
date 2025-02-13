# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% _kg_hide-input=true papermill={"duration": 2.604319, "end_time": "2023-11-27T23:37:27.607457", "exception": false, "start_time": "2023-11-27T23:37:25.003138", "status": "completed"}
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import joblib


import sys
import os

# %% [markdown] papermill={"duration": 0.008167, "end_time": "2023-11-27T23:37:27.626262", "exception": false, "start_time": "2023-11-27T23:37:27.618095", "status": "completed"}
# # De-anonymizing stock_id
#
# This notebook attempts to match stock_id used in Optiver - Trading at the Close dataset to the actual NASDAQ symbols. The idea is to:
# * Find the tick size expressed in wap returns
# * Convert that to price in USD
# * Compare to actual NASDAQ data (close price)
# * Find the stock_id<->symbol and date_id<->date mapping that minimizes the difference between the inferred price and NASDAQ closing price time series. 
#
#
# Not every stock ID is a good match, and date ID is likely off by a day or two, so use this your own risk.

# %% _kg_hide-input=true papermill={"duration": 0.039556, "end_time": "2023-11-27T23:37:27.674529", "exception": false, "start_time": "2023-11-27T23:37:27.634973", "status": "completed"}
tldr = {166: 'KHC', 121: 'KDP', 105: 'MDLZ', 151: 'CSCO', 170: 'HOLX', 0: 'MNST', 65: 'EXC', 109: 'CSX', 123: 'GILD', 198: 'CMCSA', 131: 'INCY', 21: 'SSNC', 148: 'XEL', 38: 'ATVI', 30: 'LNT', 63: 'LKQ', 24: 'AKAM', 130: 'SBUX', 120: 'FOXA', 195: 'AEP', 53: 'EBAY', 81: 'SGEN', 47: 'TXRH', 154: 'DBX', 160: 'PEP', 55: 'FAST', 37: 'ADP', 90: 'FFIV', 186: 'CTSH', 187: 'EA', 76: 'HAS', 117: 'AGNC', 134: 'VTRS', 3: 'HON', 165: 'HST', 97: 'NBIX', 145: 'CG', 25: 'EXPD', 68: 'PAYX', 52: 'CINF', 112: 'LSTR', 181: 'JBHT', 28: 'FANG', 43: 'JKHY', 12: 'AMGN', 149: 'VRSK', 144: 'PCAR', 192: 'PTC', 153: 'HTZ', 175: 'GOOGL', 189: 'CSGP', 116: 'CDW', 35: 'ROST', 46: 'TECH', 164: 'CPRT', 44: 'CME', 146: 'MIDD', 125: 'UTHR', 171: 'TROW', 73: 'GEN', 196: 'XRAY', 9: 'PFG', 199: 'BKR', 193: 'PYPL', 106: 'TRMB', 122: 'AAL', 4: 'MAR', 176: 'FTNT', 2: 'AXON', 167: 'CHRW', 91: 'DOCU', 128: 'WDC', 152: 'SBAC', 155: 'CHK', 26: 'HBAN', 132: 'TSCO', 119: 'MASI', 27: 'QRVO', 84: 'GOOGL', 23: 'SWKS', 110: 'TMUS', 182: 'UAL', 157: 'ADSK', 168: 'AMZN', 147: 'APA', 64: 'MKTX', 190: 'DXCM', 19: 'ALNY', 1: 'WING', 49: 'FITB', 194: 'PENN', 140: 'TXN', 133: 'ISRG', 177: 'SWAV', 32: 'NTAP', 22: 'ON', 77: 'VRTX', 104: 'WBA', 107: 'PODD', 59: 'Z', 72: 'ADI', 158: 'APLS', 169: 'MSFT', 94: 'PCTY', 66: 'LBRDK', 126: 'MU', 139: 'EXPE', 159: 'STLD', 137: 'TTWO', 78: 'HOOD', 114: 'LPLA', 141: 'AMAT', 15: 'ABNB', 60: 'CRWD', 183: 'MCHP', 10: 'NDAQ', 135: 'DKNG', 197: 'SPLK', 99: 'PARA', 56: 'ETSY', 13: 'TER', 62: 'RGEN', 80: 'TXG', 67: 'MRNA', 178: 'ZM', 39: 'CTAS', 173: 'FIVE', 184: 'DDOG', 162: 'ENPH', 16: 'ZBRA', 89: 'ENTG', 45: 'MSFT', 124: 'ASO', 42: 'SAIA', 115: 'ILMN', 50: 'MTCH', 98: 'JBLU', 103: 'ZS', 40: 'CZR', 108: 'SEDG', 179: 'META', 6: 'POOL', 100: 'MQ', 48: 'WDAY', 150: 'PANW', 74: 'ALGN', 113: 'LULU', 163: 'COST', 111: 'SPWR', 36: 'DLTR', 85: 'CAR', 79: 'WBD', 83: 'INTC', 75: 'CDNS', 57: 'IDXX', 180: 'GH', 93: 'ZION', 87: 'LSCC', 51: 'ROKU', 33: 'CROX', 58: 'ROP', 7: 'LRCX', 172: 'APP', 61: 'LYFT', 185: 'ODFL', 102: 'TEAM', 188: 'RUN', 17: 'KLAC', 88: 'NFLX', 95: 'AMD', 14: 'ADBE', 54: 'SNPS', 18: 'ZI', 129: 'CFLT', 136: 'LITE', 191: 'TSLA', 20: 'ULTA', 161: 'PTON', 5: 'OKTA', 71: 'EQIX', 34: 'REGN', 142: 'AVGO', 92: 'MSTR', 156: 'LCID', 41: 'NVDA', 69: 'SOFI', 138: 'SMCI', 174: 'AFRM', 11: 'COIN', 70: 'BYND', 96: 'MRVL', 118: 'FCNCA', 29: 'ORLY', 143: 'TLRY', 86: 'ONEW', 82: 'OPEN', 127: 'MDB', 101: 'FCNCA', 8: 'BKNG', 31: 'NVCR'}
print("tl;dr:")
print(tldr)

# %% papermill={"duration": 20.099647, "end_time": "2023-11-27T23:37:47.785815", "exception": false, "start_time": "2023-11-27T23:37:27.686168", "status": "completed"}
FILE = "/kaggle/input/optiver-trading-at-the-close/train.csv"
sample_data = pd.read_csv(FILE)

# %% [markdown] papermill={"duration": 0.007597, "end_time": "2023-11-27T23:37:47.801310", "exception": false, "start_time": "2023-11-27T23:37:47.793713", "status": "completed"}
# ## Inspect price return difference

# %% papermill={"duration": 0.075814, "end_time": "2023-11-27T23:37:47.885631", "exception": false, "start_time": "2023-11-27T23:37:47.809817", "status": "completed"}
sample_data.query("stock_id==1 and date_id==4").bid_price.diff().abs().value_counts()

# %% papermill={"duration": 0.796622, "end_time": "2023-11-27T23:37:48.690150", "exception": false, "start_time": "2023-11-27T23:37:47.893528", "status": "completed"}
sample_data["bid_price_diff"] = sample_data.groupby(["date_id", "stock_id"]).bid_price.diff().abs()
sample_data["ask_price_diff"] = sample_data.groupby(["date_id", "stock_id"]).ask_price.diff().abs()


# %% [markdown] papermill={"duration": 0.007625, "end_time": "2023-11-27T23:37:48.705655", "exception": false, "start_time": "2023-11-27T23:37:48.698030", "status": "completed"}
# Get the minimum price difference for a given day, and assume that this corresponds to the tick size.

# %% papermill={"duration": 1.267871, "end_time": "2023-11-27T23:37:49.981461", "exception": false, "start_time": "2023-11-27T23:37:48.713590", "status": "completed"}
tick_size_est_1 = sample_data[sample_data.bid_price_diff>0].groupby(["date_id", "stock_id"]).bid_price_diff.min().rename("tick_est_bid").to_frame()
tick_size_est_2 = sample_data[sample_data.ask_price_diff>0].groupby(["date_id", "stock_id"]).ask_price_diff.min().rename("tick_est_ask")
tick_size_est = tick_size_est_1.join(tick_size_est_2)
tick_size_est["tick_size_min"] = tick_size_est.min(1)
tick_size_est["price_est"] = 0.01 / tick_size_est.tick_size_min


# %% papermill={"duration": 0.051208, "end_time": "2023-11-27T23:37:50.041069", "exception": false, "start_time": "2023-11-27T23:37:49.989861", "status": "completed"}
wide_est = pd.pivot(tick_size_est.reset_index(), index="date_id", values="price_est", columns="stock_id").fillna(method="ffill")

# %% papermill={"duration": 0.353332, "end_time": "2023-11-27T23:37:50.402928", "exception": false, "start_time": "2023-11-27T23:37:50.049596", "status": "completed"}
import matplotlib.pyplot as plt

plt.plot(wide_est.index, wide_est[0])
plt.title("Stock 1 - Inferred Price over time")

plt.xlabel("date_id")
plt.ylabel("inferred price")

# %% [markdown] papermill={"duration": 0.008255, "end_time": "2023-11-27T23:37:50.419816", "exception": false, "start_time": "2023-11-27T23:37:50.411561", "status": "completed"}
# ## Load Stock data
#
# This is from https://www.kaggle.com/datasets/svaningelgem/nasdaq-daily-stock-prices/ by @svaningelgem.

# %% papermill={"duration": 66.041694, "end_time": "2023-11-27T23:38:56.469979", "exception": false, "start_time": "2023-11-27T23:37:50.428285", "status": "completed"}
import glob
all_stock_data = []
for s in glob.glob("/kaggle/input/nasdaq-daily-stock-prices/*.csv"):
    df = pd.read_csv(s, dtype={"ticker": str}).query("date>'2020-01-01' ") # and date<'2021-08-04' ")
    if len(df)>0:
        all_stock_data.append(df)
    
all_stock_data = pd.concat(all_stock_data)

all_stock_data = all_stock_data.dropna(subset=["ticker"])

# %% [markdown] papermill={"duration": 0.008583, "end_time": "2023-11-27T23:38:56.487891", "exception": false, "start_time": "2023-11-27T23:38:56.479308", "status": "completed"}
# ## Step 1: Find MSFT
#
# I decided to first find whether any of the stocks would correspond to MSFT. The reason I picked MSFT was that (1) it's a big part of NASDAQ 100 (2) its last split was on 2003-02-18, so if needed I can go back last 20 years without worrying about correcting for splits.
#
# I tried fancier ways of comparing time series, but at the end what worked best was to slide a window of size 481 in increments of 100 and pick the stock ID minimizing the Euclidean distance.

# %% _kg_hide-input=true papermill={"duration": 0.443852, "end_time": "2023-11-27T23:38:56.940443", "exception": false, "start_time": "2023-11-27T23:38:56.496591", "status": "completed"}
actual_stocks = pd.Series(all_stock_data.ticker.unique()).sort_values()
actual_stocks = ["MSFT"]

sid_to_name = {}

name_to_sid = {}

all_scores = []

lag = 0
for s in actual_stocks:
    distances = []
    stock_lag = []
    actual = all_stock_data.loc[all_stock_data.ticker==s]

    for lag in np.arange(0,1000,100):    
        lagged_s = actual.iloc[lag:]
        lagged_s = lagged_s.close.values[0:481]
        if len(lagged_s) < 481:
            continue

        for i in range(0,200):
            c = np.linalg.norm(lagged_s - wide_est[i].values, 2) / lagged_s.mean()
            c = c if c==c else np.inf
            distances.append(c)
            stock_lag.append((i,lag))

        idx = np.argmin(distances)
        sid, lag = stock_lag[idx]
        min_distance = min(distances)
        all_scores.append([s, sid, min_distance])

# print(min(distances), stock_lag[idx])
msft = stock_lag[idx][0]
approx_lag = stock_lag[idx][1]

print(f"Stock ID={msft}, Lag={approx_lag}")

# %% papermill={"duration": 0.395902, "end_time": "2023-11-27T23:38:57.345767", "exception": false, "start_time": "2023-11-27T23:38:56.949865", "status": "completed"}
plt.plot(wide_est[msft])
plt.plot(np.arange(0,481), actual.close.values[approx_lag:approx_lag+481])
plt.title("Compare inferred price to NASDAQ MSFT close data")
plt.legend([f"stock id={msft}", "NASDAQ close/MSFT"])

# %% [markdown] papermill={"duration": 0.009514, "end_time": "2023-11-27T23:38:57.365260", "exception": false, "start_time": "2023-11-27T23:38:57.355746", "status": "completed"}
# The date id lag value 400 appears to be a good visual match, but the code below tries to minimize the distance by checking nearby values as well.

# %% [markdown] papermill={"duration": 0.009468, "end_time": "2023-11-27T23:38:57.384510", "exception": false, "start_time": "2023-11-27T23:38:57.375042", "status": "completed"}
# # Find optimal date for MSFT

# %% _kg_hide-input=true papermill={"duration": 0.430353, "end_time": "2023-11-27T23:38:57.824554", "exception": false, "start_time": "2023-11-27T23:38:57.394201", "status": "completed"}
actual_stocks = pd.Series(all_stock_data.ticker.unique()).sort_values()
actual_stocks = ["MSFT"]

sid_to_name = {}
name_to_sid = {}
all_scores = []

lag = 0
for s in actual_stocks:
    distances = []
    stock_lag = []
    actual = all_stock_data.loc[all_stock_data.ticker==s]
    for lag in np.arange(300,500):            
        lagged_s = actual.iloc[lag:]
        lagged_s = lagged_s.close.values[0:481]
        if len(lagged_s) < 481:
            continue

        for i in [msft]:
            c = np.linalg.norm(lagged_s - wide_est[i].values[::1], 2) / lagged_s.mean()
            c = c if c==c else np.inf
            distances.append(c)
            stock_lag.append((i,lag))

        idx = np.argmin(distances)

        sid, lag = stock_lag[idx]
        min_distance = min(distances)
        all_scores.append([s, sid, min_distance])


idx = np.argmin(distances)
print(min(distances), stock_lag[idx])
best_lag = stock_lag[idx][1]
date0 = actual.iloc[best_lag].date
print(f"Date corresponding to date_id=0: {date0}")

# %% [markdown] papermill={"duration": 0.00947, "end_time": "2023-11-27T23:38:57.843839", "exception": false, "start_time": "2023-11-27T23:38:57.834369", "status": "completed"}
# ## Map date ID to date

# %% papermill={"duration": 0.023589, "end_time": "2023-11-27T23:38:57.877278", "exception": false, "start_time": "2023-11-27T23:38:57.853689", "status": "completed"}
import datetime

def align_dates(dates, d0):
    index = dates[dates==d0].index[0]
    date_to_date_id = {d:i-index for i,d in enumerate(sorted(dates))}
    date_id_to_date= {i-index:d for i,d in enumerate(sorted(dates))}
    return date_id_to_date, date_to_date_id

# date0_dt = datetime.date.fromisoformat(date0)
dates = actual.date[actual.date>=date0][0:481].reset_index(drop=True)
date_id_to_date, date_to_date_id = align_dates(dates, date0)

# %% papermill={"duration": 0.024993, "end_time": "2023-11-27T23:38:57.912659", "exception": false, "start_time": "2023-11-27T23:38:57.887666", "status": "completed"}
dates

# %% [markdown] papermill={"duration": 0.009832, "end_time": "2023-11-27T23:38:57.932772", "exception": false, "start_time": "2023-11-27T23:38:57.922940", "status": "completed"}
# ## Find the best match symbol for all stocks - check last N days only
#
# Now that we have a date/date_id mapping, we can find a match for the remaining stocks. I limit the search to the time between date_id=450 and date_id=481. This is to minimize the effects of splits.

# %% papermill={"duration": 834.498584, "end_time": "2023-11-27T23:52:52.441334", "exception": false, "start_time": "2023-11-27T23:38:57.942750", "status": "completed"}
import tqdm

REF_INDEX = 0
actual_stocks = pd.Series(all_stock_data.ticker.unique()).sort_values()

sid_to_name = {}
name_to_sid = {}
all_scores = []

min_date_id = 450
date1 = date_id_to_date[min_date_id]
max_date_id = 481
size = max_date_id - min_date_id

for s in tqdm.tqdm(actual_stocks):
    distances = []
    stock_lag = []
    
    size = max_date_id - min_date_id
    date1 = date_id_to_date[min_date_id]

    actual = all_stock_data.loc[all_stock_data.ticker==s]
    actual = actual[actual.date>=str(date1)]

    lagged_s = actual.close.values[0:size]        
    if len(lagged_s) < size:            
        continue

    for i in range(0,200):
        c = np.linalg.norm(lagged_s - wide_est[i].fillna(0).values[min_date_id:max_date_id], 2) / lagged_s.mean()
        c = c if c==c else np.inf
        all_scores.append([s, i, c]) 


# %% _kg_hide-input=true papermill={"duration": 0.646353, "end_time": "2023-11-27T23:52:53.308976", "exception": false, "start_time": "2023-11-27T23:52:52.662623", "status": "completed"}
score_df = pd.DataFrame(all_scores)
score_df.columns = ["ticker", "stock_id", "distance"]

ix=score_df.groupby("stock_id").distance.apply("idxmin")
score_df.loc[ix]

pd.options.display.max_rows = 200
best_df = score_df.loc[ix].sort_values(by="distance")
best_df.sample(20)

# %% papermill={"duration": 0.310781, "end_time": "2023-11-27T23:52:53.841085", "exception": false, "start_time": "2023-11-27T23:52:53.530304", "status": "completed"}
print("Duplicated:")
best_df[best_df.ticker.duplicated(False)]

# %% papermill={"duration": 0.233239, "end_time": "2023-11-27T23:52:54.299253", "exception": false, "start_time": "2023-11-27T23:52:54.066014", "status": "completed"}
tldr = best_df.set_index("stock_id").ticker.to_dict()
print(tldr)

# %% [markdown] papermill={"duration": 0.223059, "end_time": "2023-11-27T23:52:54.746139", "exception": false, "start_time": "2023-11-27T23:52:54.523080", "status": "completed"}
# Note that the matching is not exactly one-to-one. This could be due to different asset classes (such as GOOG vs GOOGL - the source dataset only contains GOOGL).

# %% [markdown] papermill={"duration": 0.22368, "end_time": "2023-11-27T23:52:55.193082", "exception": false, "start_time": "2023-11-27T23:52:54.969402", "status": "completed"}
# # Visual Comparison

# %% _kg_hide-input=true papermill={"duration": 0.833323, "end_time": "2023-11-27T23:52:56.248305", "exception": false, "start_time": "2023-11-27T23:52:55.414982", "status": "completed"}
all_stock_data["date_id"] = all_stock_data.date.map(date_to_date_id)
all_stock_data["stock_id"] =  all_stock_data.ticker.map(best_df.set_index("ticker").stock_id.to_dict())

# %% _kg_hide-input=true papermill={"duration": 0.450625, "end_time": "2023-11-27T23:52:56.921955", "exception": false, "start_time": "2023-11-27T23:52:56.471330", "status": "completed"}
all_stock_data_with_tick_est = all_stock_data.merge(tick_size_est, left_on=["date_id", "stock_id"], right_index=True)

# %% [markdown] papermill={"duration": 0.306251, "end_time": "2023-11-27T23:52:57.453326", "exception": false, "start_time": "2023-11-27T23:52:57.147075", "status": "completed"}
# NASDAQ data (blue) vs reverse engineered price (red). Some stocks don't match well, such as stock_id=8.

# %% _kg_hide-input=true papermill={"duration": 105.889994, "end_time": "2023-11-27T23:54:43.567024", "exception": false, "start_time": "2023-11-27T23:52:57.677030", "status": "completed"}
g = sns.FacetGrid(data=all_stock_data_with_tick_est.query("date_id>400"), col="stock_id", col_wrap=10, height=2, sharey=False)
g.map(plt.plot, "date_id", "close", color="b")
g.map(plt.plot, "date_id", "price_est", color="r")

# %% [markdown] papermill={"duration": 0.263785, "end_time": "2023-11-27T23:54:44.089022", "exception": false, "start_time": "2023-11-27T23:54:43.825237", "status": "completed"}
#

# %% [markdown] papermill={"duration": 0.265553, "end_time": "2023-11-27T23:54:44.618127", "exception": false, "start_time": "2023-11-27T23:54:44.352574", "status": "completed"}
# ## Credits / Further reading
#
# [@svaningelgem](https://www.kaggle.com/svaningelgem) for the NASDAQ dataset
#
# https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/457721 about making use of splits and bid/ask size fields.

# %% papermill={"duration": 0.265096, "end_time": "2023-11-27T23:54:45.225058", "exception": false, "start_time": "2023-11-27T23:54:44.959962", "status": "completed"}
