# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: QRT
#     language: python
#     name: python3
# ---

# Function to calculate RSI for each stock
def calculate_rsi(data, window=14):
    """
    Calculate RSI for each stock.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'RET_i' columns (i from 1 to 20).
    window: int, the lookback period for RSI calculation.

    Returns:
    pd.DataFrame with RSI values for each stock.
    """
    data = data.copy()
    
    # Calculate average gains and losses over the window period
    avg_gain = data.groupby(['STOCK', 'DATE'])[[f'RET_{day}' for day in range(1, window+1)]].mean().agg(lambda x: x[x > 0].mean(), axis=1)
    avg_loss = data.groupby(['STOCK', 'DATE'])[[f'RET_{day}' for day in range(1, window+1)]].mean().agg(lambda x: x[x < 0].mean(), axis=1).abs()

    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    # Convert RSI Series to DataFrame
    rsi_df = rsi.reset_index()
    rsi_df.rename(columns={0: 'RSI'}, inplace=True)

    # Join RSI values back to the original dataframe
    data = data.merge(rsi_df, on=['STOCK', 'DATE'], how='left')

    # Fill NaNs in RSI with the median RSI value for each stock
    data['RSI'] = data.groupby('STOCK')['RSI'].transform(lambda x: x.fillna(x.median()))

    # Add overbought and oversold signals
    data['overbought_rsi'] = np.where(data['RSI'] > 70, 1, 0)
    data['oversold_rsi'] = np.where(data['RSI'] < 30, 1, 0)

    return data

# Function to calculate RSI per sector
def calculate_rsi_per_sector(data, window=14):
    """
    Calculate RSI for each sector.

    Args:
    data: pd.DataFrame, containing 'SECTOR', 'DATE', and 'RET_i' columns (i from 1 to 20).
    window: int, the lookback period for RSI calculation.

    Returns:
    pd.DataFrame with RSI values for each sector.
    """
    data = data.copy()
    
    # Calculate average gains and losses over the window period
    avg_gain = data.groupby(['SECTOR', 'DATE'])[[f'RET_{day}' for day in range(1, window+1)]].mean().agg(lambda x: x[x > 0].mean(), axis=1)
    avg_loss = data.groupby(['SECTOR', 'DATE'])[[f'RET_{day}' for day in range(1, window+1)]].mean().agg(lambda x: x[x < 0].mean(), axis=1).abs()

    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi_sector = 100 - (100 / (1 + rs))

    # Convert RSI Series to DataFrame
    rsi_sector_df = rsi_sector.reset_index()
    rsi_sector_df.rename(columns={0: 'RSI_SECTOR'}, inplace=True)

    # Join RSI values back to the original dataframe
    data = data.merge(rsi_sector_df, on=['SECTOR', 'DATE'], how='left')

    # Add overbought and oversold signals
    data['overbought_rsi_sector'] = np.where(data['RSI_SECTOR'] > 70, 1, 0)
    data['oversold_rsi_sector'] = np.where(data['RSI_SECTOR'] < 30, 1, 0)

    return data

# Function to calculate ROC for past rows
def calculate_roc_past_rows(data, window=12):
    """
    Calculate ROC for each stock for the columns RET_1 to RET_5 over a rolling window of past rows.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'RET_i' columns (i from 1 to 5).
    window: int, the lookback period for ROC calculation.

    Returns:
    pd.DataFrame with ROC values for the columns RET_1 to RET_5 for each stock.
    """
    data = data.copy()
    new_features = []

    for i in [1, 5, 14]:
    # %% [markdown]
# ##### Imports
#

# %%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from itertools import combinations

# %%
_Xtrain = pd.read_csv("x_train.csv", index_col='ID')
_y = pd.read_csv("y_train.csv", index_col='ID')
train = pd.concat([_Xtrain, _y], axis=1)
test = pd.read_csv('x_test.csv', index_col='ID')

# %% [markdown]
# # Stock Return Prediction

# %% [markdown]
# ## Challenge Goals
#
# **Link:**
# Link to the Data Challenge by QRT [HERE](https://challengedata.ens.fr/participants/challenges/23/).
#
# **Context:**
# The proposed challenge aims at predicting the return of a stock in the US market using historical data over a recent period of 20 days. The one-day return of a stock \( j \) on day \( t \) with price \( P^t_j \) (adjusted from dividends and stock splits) is given by:
#
# $$
# R^t_j = \frac{P^t_j}{P^{t-1}_j} - 1
# $$
#
# In this challenge, we consider the residual stock return, which corresponds to the return of a stock without the market impact. Historical data are composed of residual stock returns and relative volumes, sampled each day during the 20 last business days (approximately one month). The relative volume \( V^t_j \) at time \( t \) of a stock \( j \) among the \( n \) stocks is defined by:
#
# $$
# V^t_j = \frac{V^t}{\text{median}\left(\{V^{t-1}, \ldots, V^{t-20}\}\right)}
# $$
#
# $$
# V^t_j = \overline{V^t_j} - \frac{1}{n} \sum_{i=1}^n \overline{V^t_i}
# $$
#
# where \( V^t_j \) is the volume at time \( t \) of a stock \( j \). We also give additional information about each stock such as its industry and sector.
#
# The metric considered is the accuracy of the predicted residual stock return sign.
#
# ## Data Description
#
# The dataset comprises 46 descriptive features (all float/int values):
#
# * **DATE:** An index of the date (the dates are randomized and anonymized so there is no continuity or link between any dates).
# * **STOCK:** An index of the stock.
# * **INDUSTRY:** An index of the stock industry domain (e.g., aeronautic, IT, oil company).
# * **INDUSTRY_GROUP:** An index of the group industry.
# * **SUB_INDUSTRY:** A lower-level index of the industry.
# * **SECTOR:** An index of the work sector.
# * **RET_1 to RET_20:** The historical residual returns among the last 20 days (i.e., RET_1 is the return of the previous day and so on).
# * **VOLUME_1 to VOLUME_20:** The historical relative volume traded among the last 20 days (i.e., VOLUME_1 is the relative volume of the previous day and so on).
#
# ### Target Variable
#
# * **RET:** The sign of the residual stock return at time \( t \) (binary).
#
# ### Feature Engineering
#
# To enhance the dataset and improve prediction accuracy, the following feature engineering techniques were applied:
#
# 1. **Volatility Measures:**
#     * Rolling Standard Deviation
#     * Volatility Skew
#     * Volatility of Volatility
#
# 2. **Technical Indicators:**
#     * Money Flow Index (MFI)
#     * Relative Strength Index (RSI)
#     * Accumulation/Distribution Line (ADL)
#     * Average True Range (ATR)
#     * Moving Average Convergence Divergence (MACD)
#
# These engineered features aim to capture various market dynamics and investor behaviors to better predict stock returns.
#

# %% [markdown]
# ## Structure of this code:
#
# - cleaning dataset
#     - addressing missing values
#     - deciding how to drop missing values depending on important features (RET_1 to RET_5)
#     - deciding how to fill missing values (because of outliers, median is favoured to mean)
# - EDA
#     - having a look at the distributions of the stocks' RET_i and VOLUME_i
#     - having a look at the distribution of the aggregated SECTOR and INDUSTRY's RET_i and VOLUME_i
# - Feature Engineering
#     - **Technical indicators:** to extract information from our small dataset, technical indicators have been coded. I picked typical technical indicators and hard coded them myself, typically these can be found in the [TA-Lib library](https://ta-lib.org/functions/) though 
#     - **Volatility Indicators:** Volatility (Std), Volatility of Volatility (std of std) per stock and per sector and per stock adjusted per sector.
#     - **Statistical Indicators:** to extract more information, I considered also factor investing. Factor investing has been already extensively covered in literature. The most pleasant read I have found recently about this is Chapter 7 of the new book of Giuseppe Paleologo [here](https://www.dropbox.com/scl/fo/ehfyv6ckypgj2fepyijxd/APOqG_Z9vGOL7be2IN3CgG0/EQI_20240729.pdf?rlkey=w0v26iyoxd0xj7b3lchdu5zhn&e=1&dl=0). Nonetheless, the way I went about it is straightforward. I have calculated Principal Components per SECTOR and INDUSTRY (aggregated per stock obviously, because that is where our time series is) whitened and not whitened. These didn't seem to have a positive impact in terms of explainability of our target variable 'RET' as much as the Technical Indicators, so you will find this at the end of the notebook.
#
# - Prediction Model
#     - The model is a Random Forest applied with stratified cross validation, to ensure proper assessment and avoid overfitting to any traning part of the dataset
#
# - Submission to the leaderboard
#
#
# ## Outcome
#
# As of today me writing this notebook: in the leaderboard, I am the 70th submission out of 399 submissions, I am in the top 17.3% percent of submissions. 
#
# ## Future Work
#
# Future ideas include:
# - think about how to leverage kurtosis and skewness of the distributions for statistically-driven feature engineering and indicators
# - read more extensively [Giuseppe's notes](https://www.dropbox.com/scl/fo/dcjs09n8o1n9who0vo4nl/AAPjHxg0j0CRJ5me1OKF7JE/NYU%20notes%20Giuseppe?rlkey=liz1nlorbnzzolzhyv88sp69u&e=1&dl=0) and get inspiration about more statistically and mathematically proven and robust methods to derive alpha 

# %% [markdown]
# # Adressing missing values

# %% [markdown]
# From the code provided for this data challenge, we know that the most significant variables are in days between 1 and 5,

# %%
to_drop = [f'RET_{day}' for day in range(6,21)]
to_drop += [f'VOLUME_{day}' for day in range(6,21)]
cleaned_train = train.drop(columns= to_drop)

# %% [markdown]
# ### Possible Reasons for NaN Values
#
# 1. **Calculation of Relative Volumes**:
#    - The relative volumes are computed using the median of the past 20 days' volumes. If any day within this 20-day window has a missing volume value, it will cause NaN values in the calculation for subsequent days. For example, if there is a missing value on day $D$, then the relative volumes for days $D$ to $D+19$ will be affected. The relative volume $\tilde{V}^t_j$ at time $t$ of a stock $j$ is calculated as:
#      $$
#      \tilde{V}^t_j = \frac{V^t_j}{\text{median}( \{ V^{t-1}_j, \ldots, V^{t-20}_j \} )}
#      $$
#      The adjusted relative volume $V^t_j$ is then given by:
#      $$
#      V^t_j = \tilde{V}^t_j - \frac{1}{n} \sum_{i=1}^{n} \tilde{V}^t_i
#      $$
#
# 2. **Market Closures**:
#    - Stock markets do not operate on weekends and public holidays. Consequently, there are no trading volumes or returns recorded on these days, resulting in NaN values for those dates. This absence of data could propagate into calculations requiring continuous data over several days, such as rolling averages or median calculations.
#
# 3. **Data Gaps**:
#    - The dataset might have gaps due to data collection issues or missing entries. These gaps can lead to NaN values when the algorithm attempts to calculate features based on missing data points.
#
# 4. **Data Randomization and Anonymization**:
#    - According to the data description, dates are randomized and anonymized. This process could potentially introduce NaN values if not handled carefully, especially when aligning data points chronologically for calculations.
#
#

# %%
(cleaned_train.isna().sum()/len(cleaned_train)*100)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
plt.subplots_adjust(hspace=0.5)

for i, categorie in enumerate(['INDUSTRY', 'INDUSTRY_GROUP', 'SECTOR', 'SUB_INDUSTRY', 'STOCK', 'DATE']):
    plt.subplot(3, 2, i + 1)
    plt.title(categorie)
    
    # Calculate the percentage of NaN values per sub-category
    nan_percentages = [
        (cleaned_train[cleaned_train[categorie] == sub_categorie].isna().sum(axis=1) > 0).sum() / 
        len(cleaned_train[cleaned_train[categorie] == sub_categorie]) * 100 
        for sub_categorie in cleaned_train[categorie].sort_values().unique()
    ]
    
    sns.barplot(x=cleaned_train[categorie].sort_values().unique(), y=nan_percentages)
    
    plt.xlabel('sub-categorie')
    plt.ylabel('%')
    plt.xticks(rotation=90)

plt.show()


# %% [markdown]
# **Analysis of NaN Values Distribution**
#
# The plots show the distribution of NaN values per sub-category in percentage. We can see a fairly even distribution of NaN values for the categorical variable `SECTOR`, while the amount of NaN values for the other categorical variables appears to be more equally spread out. 
#
# It might be insightful to check if there are rows that predominantly consist of NaN values (especially for the descriptive variables `RET` and `VOLUME`). If such rows exist, we can drop them in good faith as these columns do not provide value in understanding the underlying structure. 
#
# During this investigation, I noticed the following:
#
# - Given no observed returns, there is no volume observed. Therefore, we should only delete those observations where no return has been observed over the past 5 days. I chose a 5-day window since I am using the observations of the past 5 days in my prediction.
#

# %%
return_features = [f'RET_{day}' for day in range(1,6)]
return_to_drop = cleaned_train[(cleaned_train[return_features].isna().sum(axis=1)/(cleaned_train[return_features].shape[1]) >= 1)][return_features]
return_to_drop

# %% [markdown]
# There are 2,256 rows that do not provide any value to our dataset, as these rows contain only NaN values for both `RET` and `VOLUME`. Therefore, I will proceed to drop these rows.
#
# Dropping the corresponding rows:
#

# %%
cleaned_train.drop(index=return_to_drop.index, inplace=True)

# %%
(cleaned_train.isna().sum()/len(cleaned_train)*100)

# %% [markdown]
# We still have quite a few missing values, let's see what is the best way to bo about it to fill these:

# %%
num_cols= [f'RET_{i}' for i in range(1, 6)] + [f'VOLUME_{i}' for i in range(1, 6)]


# %%
cleaned_train[num_cols].describe()

# %% [markdown]
# **Outliers**
#
# Let's check for outliers. Since, from above, Volumes' stardard deviations are significantly larger than returns on average, I am concerned for **outliers**.

# %% [markdown]
# To properly visualize outliers, let's do boxplots.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Generate the lists using list comprehension
ret_columns = [f'RET_{i}' for i in range(1, 6)]
volume_columns = [f'VOLUME_{i}' for i in range(1, 6)]

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Boxplot for volume columns
sns.boxplot(data=cleaned_train[volume_columns], ax=axs[0])
axs[0].set_ylim((-2, 2))
axs[0].set_title('Volume Columns')

# Boxplot for ret columns
sns.boxplot(data=cleaned_train[ret_columns], ax=axs[1])
axs[1].set_ylim((-0.1, 0.1))
axs[1].set_title('RET Columns')

# Show the plots
plt.tight_layout()
plt.show()


# %% [markdown]


# %%
train['RET'].value_counts(normalize=True)*100


# %% [markdown]
# **Correlation**
#
# Let's check correlation between features.

# %%
features = [f'RET_{day}' for day in range(1,21)]
features += [f'VOLUME_{day}' for day in range(1,21)]
features += ['RET']

fig = plt.figure(figsize=(20,20))
plt.matshow(train[features].corr(), fignum=fig.number)
plt.xticks(range(train[features].shape[1]), train[features].columns, rotation=90, fontsize=14)
plt.yticks(range(train[features].shape[1]), train[features].columns, fontsize=14)
plt.colorbar()
plt.show()

# %% [markdown]
# **Filling the Missing Values**
#
# Based on the above, I wil fill the missing values with median and not mean.

# %%
ret_columns = [f'RET_{day}' for day in range(1,6)]
dropping = train[(train[ret_columns].isna().sum(axis=1)/(train[ret_columns].shape[1]) >= 1)][ret_columns]

train.drop(index=dropping.index, inplace=True) 

# %%
for column in [f'VOLUME_{day}' for day in range(1,21)]:
    train[column] = train[column].fillna(train[column].median())
    test[column] = test[column].fillna(test[column].median())

# %%
for column in [f'RET_{day}' for day in range(1,21)]:
    train[column] = train[column].fillna(train[column].median())
    test[column] = test[column].fillna(test[column].median())

# %%
test.shape

# %%
test

# %%
train

# %% [markdown]
# # EDA Exploratory Data Analysis

# %% [markdown]
# Let's look at stock return and volume distributions!  These can be quite telling of how our entire dataset looks like.

# %%
df = train.copy()


# Now proceed with your plotting code
import ipywidgets as widgets
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt

# Dropdown to select stock
stock_dropdown = widgets.Dropdown(
    options=df['STOCK'].unique(),
    description='Stock:',
    disabled=False,
)

# Function to update the plot
def update_plot(stock):
    stock_df = df[df['STOCK'] == stock]
    fig, axes = plt.subplots(2, 5, figsize=(18, 12))
    
    # Plotting RET_1 to RET_5
    for i in range(5):
        sns.histplot(stock_df[f'RET_{i+1}'], bins=50, kde=True, ax=axes[0, i])
        axes[0, i].set_title(f'Distribution of {stock} Returns (RET_{i+1})')
        axes[0, i].set_xlabel(f'RET_{i+1}')
        axes[0, i].set_ylabel('Frequency')
    
    # Plotting VOLUME_1 to VOLUME_5
    for i in range(5):
        sns.histplot(stock_df[f'VOLUME_{i+1}'], bins=50, kde=True, ax=axes[1, i])
        axes[1, i].set_title(f'Distribution of {stock} Volume (VOLUME_{i+1})')
        axes[1, i].set_xlabel(f'VOLUME_{i+1}')
        axes[1, i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Link the dropdown to the update function
interactive_plot = widgets.interactive(update_plot, stock=stock_dropdown)
display(interactive_plot)


# %% [markdown]
# **Log**
#
# Let's look at log returns and volumes

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display

# Load your dataset
df = train.copy()

# Function to update the plot
def update_plot(stock):
    stock_df = df[df['STOCK'] == stock]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    sns.histplot(stock_df['RET_1'], bins=50, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title(f'Distribution of {stock} Returns (RET_1)')
    axes[0, 0].set_xlabel('RET_1')
    axes[0, 0].set_ylabel('Frequency')

    sns.histplot(stock_df['VOLUME_1'], bins=50, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title(f'Distribution of {stock} Volume (VOLUME_1)')
    axes[0, 1].set_xlabel('VOLUME_1')
    axes[0, 1].set_ylabel('Frequency')

    sns.histplot(np.log(stock_df['RET_1'].replace(0, np.nan)).dropna(), bins=50, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title(f'Distribution of Log {stock} Returns (Log(RET_1))')
    axes[1, 0].set_xlabel('Log(RET_1)')
    axes[1, 0].set_ylabel('Frequency')

    sns.histplot(np.log(stock_df['VOLUME_1'].replace(0, np.nan)).dropna(), bins=50, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title(f'Distribution of Log {stock} Volume (Log(VOLUME_1))')
    axes[1, 1].set_xlabel('Log(VOLUME_1)')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Dropdown to select stock
stock_dropdown = widgets.Dropdown(
    options=df['STOCK'].unique(),
    description='Stock:',
    disabled=False,
)

# Link the dropdown to the update function
interactive_plot = widgets.interactive(update_plot, stock=stock_dropdown)
display(interactive_plot)


# %% [markdown]
# **Comment**
# In the code above we have a dropdown menu where we can select the stock and have look at the distributions of RET_1 to 5 and VOLUME_1 to 5.
#
# We cannot click 1 by 1 across 5k stocks, so let's have a macro view per SECTOR and INDUSTRY now.

# %% [markdown]
# **Stats EDA per SECTOR and INDUSTRY**

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Load your dataset
df = train.copy()

# Function to calculate skewness and kurtosis
def calc_skew_kurt(stock_df):
    skewness_ret = skew(stock_df['RET_1'])
    kurt_ret = kurtosis(stock_df['RET_1'])
    skewness_vol = skew(stock_df['VOLUME_1'])
    kurt_vol = kurtosis(stock_df['VOLUME_1'])
    return skewness_ret, kurt_ret, skewness_vol, kurt_vol

# Calculate skewness and kurtosis for each stock
stock_stats = df.groupby('STOCK').apply(lambda x: pd.Series({
    'Skewness_RET': calc_skew_kurt(x)[0],
    'Kurtosis_RET': calc_skew_kurt(x)[1],
    'Skewness_VOLUME': calc_skew_kurt(x)[2],
    'Kurtosis_VOLUME': calc_skew_kurt(x)[3]
})).reset_index()

# Calculate average skewness and kurtosis per sector and industry
sector_stats = df.groupby('SECTOR').apply(lambda x: pd.Series({
    'Average Skewness_RET': calc_skew_kurt(x)[0],
    'Average Kurtosis_RET': calc_skew_kurt(x)[1],
    'Average Skewness_VOLUME': calc_skew_kurt(x)[2],
    'Average Kurtosis_VOLUME': calc_skew_kurt(x)[3]
})).reset_index()

industry_stats = df.groupby('INDUSTRY').apply(lambda x: pd.Series({
    'Average Skewness_RET': calc_skew_kurt(x)[0],
    'Average Kurtosis_RET': calc_skew_kurt(x)[1],
    'Average Skewness_VOLUME': calc_skew_kurt(x)[2],
    'Average Kurtosis_VOLUME': calc_skew_kurt(x)[3]
})).reset_index()


# %%
# Plotting sector and industry statistics
def plot_sector_industry_stats():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.barplot(x='SECTOR', y='Average Skewness_RET', data=sector_stats, ax=axes[0, 0])
    axes[0, 0].set_title('Average Skewness_RET per Sector')
    axes[0, 0].tick_params(axis='x', rotation=90)

    sns.barplot(x='SECTOR', y='Average Kurtosis_RET', data=sector_stats, ax=axes[0, 1])
    axes[0, 1].set_title('Average Kurtosis_RET per Sector')
    axes[0, 1].tick_params(axis='x', rotation=90)

    sns.barplot(x='SECTOR', y='Average Skewness_VOLUME', data=sector_stats, ax=axes[1, 0])
    axes[1, 0].set_title('Average Skewness_VOL per Sector')
    axes[1, 0].tick_params(axis='x', rotation=90)

    sns.barplot(x='SECTOR', y='Average Kurtosis_VOLUME', data=sector_stats, ax=axes[1, 1])
    axes[1, 1].set_title('Average Kurtosis_VOL per Sector')
    axes[1, 1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.barplot(x='INDUSTRY', y='Average Skewness_RET', data=industry_stats, ax=axes[0, 0])
    axes[0, 0].set_title('Average Skewness_RET per Industry')
    axes[0, 0].tick_params(axis='x', rotation=90)

    sns.barplot(x='INDUSTRY', y='Average Kurtosis_RET', data=industry_stats, ax=axes[0, 1])
    axes[0, 1].set_title('Average Kurtosis_RET per Industry')
    axes[0, 1].tick_params(axis='x', rotation=90)

    sns.barplot(x='INDUSTRY', y='Average Skewness_VOLUME', data=industry_stats, ax=axes[1, 0])
    axes[1, 0].set_title('Average Skewness_VOL per Industry')
    axes[1, 0].tick_params(axis='x', rotation=90)

    sns.barplot(x='INDUSTRY', y='Average Kurtosis_VOLUME', data=industry_stats, ax=axes[1, 1])
    axes[1, 1].set_title('Average Kurtosis_VOL per Industry')
    axes[1, 1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

# Call the function to plot the sector and industry stats
plot_sector_industry_stats()

# %% [markdown]
# ### **Explanation of the Graphs**
#
# The graphs show the average skewness and kurtosis for `RET` (returns) and `VOLUME` (volume) across different sectors and industries. Skewness measures the asymmetry of the distribution of values, while kurtosis measures the "tailedness" or the propensity for extreme values. Here are the details:
#
# 1. **Top Row (Sector Level)**:
#    - **Average Skewness of RET per Sector**: Shows how the returns' distributions are skewed (left or right) for each sector.
#    - **Average Kurtosis of RET per Sector**: Shows the propensity for extreme return values in each sector.
#    - **Average Skewness of VOLUME per Sector**: Shows how the volumes' distributions are skewed (left or right) for each sector.
#    - **Average Kurtosis of VOLUME per Sector**: Shows the propensity for extreme volume values in each sector.
#
# 2. **Bottom Row (Industry Level)**:
#    - **Average Skewness of RET per Industry**: Shows how the returns' distributions are skewed for each industry.
#    - **Average Kurtosis of RET per Industry**: Shows the propensity for extreme return values in each industry.
#    - **Average Skewness of VOLUME per Industry**: Shows how the volumes' distributions are skewed for each industry.
#    - **Average Kurtosis of VOLUME per Industry**: Shows the propensity for extreme volume values in each industry.
#
# ### Mathematical Explanation
#
# 1. **Skewness**:
#    - **Formula**: $$ \text{Skewness} = \frac{E[(X - \mu)^3]}{\sigma^3} $$
#    - **Interpretation**: A measure of the asymmetry of the probability distribution of a real-valued random variable. Positive skewness indicates a distribution with an asymmetric tail extending toward more positive values, while negative skewness indicates a tail extending toward more negative values.
#
# 2. **Kurtosis**:
#    - **Formula**: $$ \text{Kurtosis} = \frac{E[(X - \mu)^4]}{\sigma^4} $$
#    - **Interpretation**: A measure of the "tailedness" of the probability distribution. High kurtosis indicates more outliers (heavier tails), while low kurtosis indicates fewer outliers (lighter tails).
#
# ### Comments on Distributions
#
# 1. **Sector Level**:
#    - **RET Skewness**: Sectors like 5 and 7 have high positive skewness, indicating returns are skewed toward positive values. Sector 11 shows negative skewness.
#    - **RET Kurtosis**: Sector 7 has extremely high kurtosis, indicating a high propensity for extreme return values.
#    - **VOLUME Skewness**: Sectors 5 and 7 also show high positive skewness in volume, indicating volume data is skewed toward higher values.
#    - **VOLUME Kurtosis**: Sector 6 has an extremely high kurtosis, suggesting many extreme volume values.
#
# 2. **Industry Level**:
#    - **RET Skewness**: Industries have varied skewness, with some showing positive and others negative skewness. This indicates diverse return distributions across industries.
#    - **RET Kurtosis**: Certain industries exhibit very high kurtosis, indicating the presence of extreme return values.
#    - **VOLUME Skewness**: Similar to returns, volume skewness varies widely across industries.
#    - **VOLUME Kurtosis**: Some industries show extremely high kurtosis, suggesting many extreme volume values.
#
# ### Comments on Sectors
#
# 1. **Sector 7**:
#    - **RET Skewness**: High positive skewness suggests that returns are more often positive but with some very high return days.
#    - **RET Kurtosis**: Extremely high kurtosis indicates the presence of very high returns on certain days, suggesting volatility.
#    - **VOLUME Skewness**: High positive skewness suggests that trading volumes are frequently high.
#    - **VOLUME Kurtosis**: High kurtosis indicates that there are days with very high trading volumes, indicating sporadic trading spikes.
#
# 2. **Sector 11**:
#    - **RET Skewness**: Negative skewness indicates that returns are more often negative.
#    - **RET Kurtosis**: Moderate kurtosis suggests fewer extreme return days compared to Sector 7.
#    - **VOLUME Skewness**: Relatively low skewness compared to other sectors.
#    - **VOLUME Kurtosis**: Low kurtosis indicates more stable trading volumes without significant spikes.
#
# ### Comments on Industries
#
# 1. **Industry with highest RET Kurtosis**:
#    - **Skewness**: The skewness can be either positive or negative, but the high kurtosis indicates a lot of extreme returns.
#    - **Kurtosis**: High kurtosis indicates that this industry has a lot of outlier returns, suggesting high volatility and risk.
#
# 2. **Industry with highest VOLUME Kurtosis**:
#    - **Skewness**: The skewness could be positive, indicating frequent high trading volumes.
#    - **Kurtosis**: Very high kurtosis indicates that this industry experiences significant spikes in trading volumes, possibly due to market events or news affecting this industry.
#
# In summary, sectors and industries exhibit different skewness and kurtosis characteristics in both returns and volumes, indicating varying degrees of asymmetry and tail behavior. Some sectors and industries are more prone to extreme values, which can indicate higher risk and volatility.
#
#

# %% [markdown]
# **Broader Look**

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Load your dataset
df = train.copy()

# Define a function to calculate required statistics
def calculate_statistics(data):
    stats = {}
    # RETURN
    stats['Mean_RET'] = np.mean(data['RET_1'])
    stats['Skewness_RET'] = skew(data['RET_1'])
    stats['Kurtosis_RET'] = kurtosis(data['RET_1'])
    stats['Std_RET'] = np.std(data['RET_1'])
    stats['Std_of_Std_RET'] = np.std(data['RET_1'].rolling(window=20).std().dropna())

    log_RET_1 = np.log(data['RET_1'].replace(0, np.nan)).dropna()
    log_squared_RET_1 = log_RET_1**2
    stats['Log_Mean_RET'] = np.mean(log_RET_1)
    stats['Log_Skewness_RET'] = skew(log_RET_1)
    stats['Log_Kurtosis_RET'] = kurtosis(log_RET_1)
    stats['Log_Std_RET'] = np.std(log_RET_1)
    stats['Log_Std_of_Std_RET'] = np.std(log_RET_1.rolling(window=20).std().dropna())

    stats['Log_Squared_Skewness_RET'] = skew(log_squared_RET_1)
    stats['Log_Squared_Kurtosis_RET'] = kurtosis(log_squared_RET_1)
    stats['Log_Squared_Std_RET'] = np.std(log_squared_RET_1)
    stats['Log_Squared_Std_of_Std_RET'] = np.std(log_squared_RET_1.rolling(window=20).std().dropna())



    # VOLUME
    stats['Mean_VOLUME'] = np.mean(data['VOLUME_1'])
    stats['Skewness_VOLUME'] = skew(data['VOLUME_1'])
    stats['Kurtosis_VOLUME'] = kurtosis(data['VOLUME_1'])
    stats['Std_VOLUME'] = np.std(data['VOLUME_1'])
    stats['Std_of_Std_VOLUME'] = np.std(data['VOLUME_1'].rolling(window=20).std().dropna())

    log_VOLUME_1 = np.log(data['VOLUME_1'].replace(0, np.nan)).dropna()
    log_squared_VOLUME_1 = log_VOLUME_1**2
    stats['Log_Mean_VOLUME'] = np.mean(log_VOLUME_1)
    stats['Log_Skewness_VOLUME'] = skew(log_VOLUME_1)
    stats['Log_Kurtosis_VOLUME'] = kurtosis(log_VOLUME_1)
    stats['Log_Std_VOLUME'] = np.std(log_VOLUME_1)
    stats['Log_Std_of_Std_VOLUME'] = np.std(log_VOLUME_1.rolling(window=20).std().dropna())
    stats['Log_Squared_Skewness_VOLUME'] = skew(log_squared_VOLUME_1)
    stats['Log_Squared_Kurtosis_VOLUME'] = kurtosis(log_squared_VOLUME_1)
    stats['Log_Squared_Std_VOLUME'] = np.std(log_squared_VOLUME_1)
    stats['Log_Squared_Std_of_Std_VOLUME'] = np.std(log_squared_VOLUME_1.rolling(window=20).std().dropna())

    return pd.Series(stats)

# Calculate statistics for each sector and industry
sector_stats = df.groupby('SECTOR').apply(calculate_statistics).reset_index()
industry_stats = df.groupby('INDUSTRY').apply(calculate_statistics).reset_index()


# %%
# Define a function to plot the statistics
def plot_sector_industry_stats():
    # Plot sector statistics
    fig, axes = plt.subplots(2, 5, figsize=(35, 10))

    sns.barplot(x='SECTOR', y='Mean_RET', data=sector_stats, ax=axes[0, 0])
    axes[0, 0].set_title('Mean_RET per Sector')
    axes[0, 0].tick_params(axis='x', rotation=90)

    sns.barplot(x='SECTOR', y='Skewness_RET', data=sector_stats, ax=axes[0, 1])
    axes[0, 1].set_title('Skewness_RET per Sector')
    axes[0, 1].tick_params(axis='x', rotation=90)

    sns.barplot(x='SECTOR', y='Kurtosis_RET', data=sector_stats, ax=axes[0, 2])
    axes[0, 2].set_title('Kurtosis_RET per Sector')
    axes[0, 2].tick_params(axis='x', rotation=90)

    sns.barplot(x='SECTOR', y='Std_RET', data=sector_stats, ax=axes[0, 3])
    axes[0, 3].set_title('Std_RET per Sector')
    axes[0, 3].tick_params(axis='x', rotation=90)

    sns.barplot(x='SECTOR', y='Std_of_Std_RET', data=sector_stats, ax=axes[0, 4])
    axes[0, 4].set_title('Std_of_Std_RET per Sector')
    axes[0, 4].tick_params(axis='x', rotation=90)

    sns.barplot(x='SECTOR', y='Log_Mean_RET', data=sector_stats, ax=axes[1, 0])
    axes[1, 0].set_title('Log_Mean_RET per Sector')
    axes[1, 0].tick_params(axis='x', rotation=90)

    sns.barplot(x='SECTOR', y='Log_Skewness_RET', data=sector_stats, ax=axes[1, 1])
    axes[1, 1].set_title('Log_Skewness_RET per Sector')
    axes[1, 1].tick_params(axis='x', rotation=90)

    sns.barplot(x='SECTOR', y='Log_Kurtosis_RET', data=sector_stats, ax=axes[1, 2])
    axes[1, 2].set_title('Log_Kurtosis_RET per Sector')
    axes[1, 2].tick_params(axis='x', rotation=90)

    sns.barplot(x='SECTOR', y='Log_Std_RET', data=sector_stats, ax=axes[1, 3])
    axes[1, 3].set_title('Log_Std_RET per Sector')
    axes[1, 3].tick_params(axis='x', rotation=90)

    sns.barplot(x='SECTOR', y='Log_Std_of_Std_RET', data=sector_stats, ax=axes[1, 4])
    axes[1, 4].set_title('Log_Std_of_Std_RET per Sector')
    axes[1, 4].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

    # Plot industry statistics
    fig, axes = plt.subplots(2, 5, figsize=(35, 10))

    sns.barplot(x='INDUSTRY', y='Mean_RET', data=industry_stats, ax=axes[0, 0])
    axes[0, 0].set_title('Mean_RET per Industry')
    axes[0, 0].tick_params(axis='x', rotation=90)

    sns.barplot(x='INDUSTRY', y='Skewness_RET', data=industry_stats, ax=axes[0, 1])
    axes[0, 1].set_title('Skewness_RET per Industry')
    axes[0, 1].tick_params(axis='x', rotation=90)

    sns.barplot(x='INDUSTRY', y='Kurtosis_RET', data=industry_stats, ax=axes[0, 2])
    axes[0, 2].set_title('Kurtosis_RET per Industry')
    axes[0, 2].tick_params(axis='x', rotation=90)

    sns.barplot(x='INDUSTRY', y='Std_RET', data=industry_stats, ax=axes[0, 3])
    axes[0, 3].set_title('Std_RET per Industry')
    axes[0, 3].tick_params(axis='x', rotation=90)

    sns.barplot(x='INDUSTRY', y='Std_of_Std_RET', data=industry_stats, ax=axes[0, 4])
    axes[0, 4].set_title('Std_of_Std_RET per Industry')
    axes[0, 4].tick_params(axis='x', rotation=90)

    sns.barplot(x='INDUSTRY', y='Log_Mean_RET', data=industry_stats, ax=axes[1, 0])
    axes[1, 0].set_title('Log_Mean_RET per Industry')
    axes[1, 0].tick_params(axis='x', rotation=90)

    sns.barplot(x='INDUSTRY', y='Log_Skewness_RET', data=industry_stats, ax=axes[1, 1])
    axes[1, 1].set_title('Log_Skewness_RET per Industry')
    axes[1, 1].tick_params(axis='x', rotation=90)

    sns.barplot(x='INDUSTRY', y='Log_Kurtosis_RET', data=industry_stats, ax=axes[1, 2])
    axes[1, 2].set_title('Log_Kurtosis_RET per Industry')
    axes[1, 2].tick_params(axis='x', rotation=90)

    sns.barplot(x='INDUSTRY', y='Log_Std_RET', data=industry_stats, ax=axes[1, 3])
    axes[1, 3].set_title('Log_Std_RET per Industry')
    axes[1, 3].tick_params(axis='x', rotation=90)

    sns.barplot(x='INDUSTRY', y='Log_Std_of_Std_RET', data=industry_stats, ax=axes[1, 4])
    axes[1, 4].set_title('Log_Std_of_Std_RET per Industry')
    axes[1, 4].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

# Call the function to plot the sector and industry stats
plot_sector_industry_stats()


# %% [markdown]
# # Feature Engineering
#
# The following provides a breakdown of all the feature engineering tasks performed on the dataset, categorized into return-related features and volume-related features. Each feature is listed with a brief description.
#
# #### Return-Related Features
#
# 1. **Relative Strength Index (RSI):**
#    - Measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
#
# 2. **Rate of Change (ROC):**
#    - Measures the percentage change between the most recent price and the price n periods ago.
#
# 3. **Momentum:**
#    - Measures the change in price over a specified period.
#
# 4. **Stochastic Oscillator (%K and %D):**
#    - Measures the current price relative to the price range over a specified period.
#
# 5. **Moving Average Convergence Divergence (MACD):**
#    - Measures the difference between the short-term and long-term exponential moving averages (EMA).
#
# 6. **Golden Cross**
#    - A Golden Cross is a bullish breakout pattern formed from a crossover involving a security's short-term moving average (such as the 10-day moving average) breaking above its long-term moving average (such as the 200-day moving average) or resistance level.
#
#
# 7. **Bollinger Bands:**
#    - Measures price volatility by placing bands around a moving average.
#
# 8. **Cumulative Return**
#    - Cumulative return is the total change in the price of an asset over a period, expressed as a percentage.
#
# 9. **Bollinger for Cum Rets**
#
# 10. **Money Flow Index (MFI)**
#
# 11. **Moving Averages:**
#    - **Exponential Moving Average (EMA):** Measures the average price over a specified period with more weight on recent prices.
#    - **Simple Moving Average (SMA):** Measures the average price over a specified period.
#
#
# 12. **Average True Range (ATR):**
#     - Measures the volatility of a security.
#
#
#
#
# #### Volume-Related Features
#
# 1. **Volume Deviation**
#
# 2. **Volume Spike Detection**
#
# 3. **Accumulation/Distribution Line (ADL):**
#    - Measures the cumulative flow of money into and out of a security.
#
# 1. **Relative Volume:**
#    - Measures trading volume relative to the average volume over a specified period.
#
# 2. **On-Balance Volume (OBV):**
#    - Measures cumulative volume flow that indicates buying and selling pressure.
#
#
#
#
# This breakdown provides a comprehensive overview of the feature engineering tasks, categorized into return-related and volume-related features.
#

# %% [markdown]
# **New Features List**

# %%
new_features = []

# %% [markdown]
# **1. Relative Strength Index (RSI)**
#
# The RSI is a momentum oscillator that measures the speed and change of price movements. It oscillates between 0 and 100. Traditionally, RSI is considered overbought when above 70 and oversold when below 30.
#
# **Formula:**
# $$ \text{RSI} = 100 - \frac{100}{1 + RS} $$
# $$ RS = \frac{\text{Average Gain}}{\text{Average Loss}} $$
#

# %%
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def calculate_rsi(data, window=14):
    """
    Calculate RSI for each stock.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'RET_i' columns (i from 1 to 20).
    window: int, the lookback period for RSI calculation.

    Returns:
    pd.DataFrame with RSI values for each stock.
    """
    data = data.copy()
    
    # Calculate average gains and losses over the window period
    avg_gain = data.groupby(['STOCK', 'DATE'])[[f'RET_{day}' for day in range(1, window+1)]].mean().agg(lambda x: x[x > 0].mean(), axis=1)
    avg_loss = data.groupby(['STOCK', 'DATE'])[[f'RET_{day}' for day in range(1, window+1)]].mean().agg(lambda x: x[x < 0].mean(), axis=1).abs()

    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    # Convert RSI Series to DataFrame
    rsi_df = rsi.reset_index()
    rsi_df.rename(columns={0: 'RSI'}, inplace=True)



    # Join RSI values back to the original dataframe
    data = data.merge(rsi_df, on=['STOCK', 'DATE'], how='left')

    # Fill NaNs in RSI with the median RSI value for each stock
    data['RSI'] = data.groupby('STOCK')['RSI'].transform(lambda x: x.fillna(x.median()))

    # Add overbought and oversold signals
    data['overbought_rsi'] = np.where(data['RSI'] > 70, 1, 0)
    data['oversold_rsi'] = np.where(data['RSI'] < 30, 1, 0)





    return data


# %%
new_features.append('RSI')
new_features.append('overbought_rsi')
new_features.append('oversold_rsi')

# %%
train = calculate_rsi(train)
test = calculate_rsi(test)


# %%
train[new_features]

# %%
test[new_features]

# %%
train[new_features].isna().sum()

# %%
new_features

# %% [markdown]
# **RSI per Sector**
#
# The RSI is also calculated for each sector by grouping the data by 'SECTOR' and 'DATE'.
#
# **Formula:**
# $$ \text{RSI} = 100 - \frac{100}{1 + RS} $$
# $$ RS = \frac{\text{Average Gain}}{\text{Average Loss}} $$
#

# %%
import pandas as pd
import numpy as np

def calculate_rsi_per_sector(data, window=14):
    """
    Calculate RSI for each sector.

    Args:
    data: pd.DataFrame, containing 'SECTOR', 'DATE', and 'RET_i' columns (i from 1 to 20).
    window: int, the lookback period for RSI calculation.

    Returns:
    pd.DataFrame with RSI values for each sector.
    """
    data = data.copy()
    
    # Calculate average gains and losses over the window period
    avg_gain = data.groupby(['SECTOR', 'DATE'])[[f'RET_{day}' for day in range(1, window+1)]].mean().agg(lambda x: x[x > 0].mean(), axis=1)
    avg_loss = data.groupby(['SECTOR', 'DATE'])[[f'RET_{day}' for day in range(1, window+1)]].mean().agg(lambda x: x[x < 0].mean(), axis=1).abs()

    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi_sector = 100 - (100 / (1 + rs))

    # Convert RSI Series to DataFrame
    rsi_sector_df = rsi_sector.reset_index()
    rsi_sector_df.rename(columns={0: 'RSI_SECTOR'}, inplace=True)

    # Join RSI values back to the original dataframe
    data = data.merge(rsi_sector_df, on=['SECTOR', 'DATE'], how='left')

    # Add overbought and oversold signals
    data['overbought_rsi_sector'] = np.where(data['RSI_SECTOR'] > 70, 1, 0)
    data['oversold_rsi_sector'] = np.where(data['RSI_SECTOR'] < 30, 1, 0)

    return data


# %%
new_features.append('RSI_SECTOR')
new_features.append('overbought_rsi_sector')
new_features.append('oversold_rsi_sector')

# %%
train = calculate_rsi_per_sector(train)
test = calculate_rsi_per_sector(test)

# %%
train[new_features].isna().sum()

# %%
train[new_features]

# %%
new_features

# %%

# %%
test.shape

# %% [markdown]
# **2. Rate of Change (ROC)**
#
# The ROC is a momentum oscillator that measures the percentage change in price between the current price and the price a certain number of periods ago.
#
# **Formula:**
# $$ \text{ROC} = \frac{\text{Current Price} - \text{Price n periods ago}}{\text{Price n periods ago}} \times 100 $$
#

# %%
import pandas as pd
import numpy as np

def calculate_roc_past_rows(data, window=12):
    """
    Calculate ROC for each stock for the columns RET_1 to RET_5 over a rolling window of past rows.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'RET_i' columns (i from 1 to 5).
    window: int, the lookback period for ROC calculation.

    Returns:
    pd.DataFrame with ROC values for the columns RET_1 to RET_5 for each stock.
    """
    data = data.copy()
    new_features = []

    for i in [1, 5, 14]:
        ret_col = f'RET_{i}'
        roc_col = f'ROC_{i}'
        new_features.append(f'ROC_{i}')
        
        # Calculate ROC over past rows for each stock and date
        data[roc_col] = data.groupby(['STOCK'])[ret_col].transform(lambda x: x.pct_change(periods=window) * 100)

        # Fill NaNs with the median of the ROC values for each stock
        data[roc_col] = data.groupby('STOCK')[roc_col].transform(lambda x: x.fillna(x.median()))

        # Fillna for those stocks which have only 1 observation
        data[roc_col] = data.groupby('SECTOR')[roc_col].transform(lambda x: x.fillna(x.median()))
    
    return data, new_features



# %%
train, f = calculate_roc_past_rows(train) 
test, _ = calculate_roc_past_rows(test) 

# %%
new_features.extend(f)


# %%
new_features

# %%
train[new_features].isna().sum()

# %%
train[new_features]

# %%
test.shape

# %% [markdown]
# **3. Momentum**
#
#
# In this implementation, we calculate the momentum for both `RET` and `VOLUME` over a specified window period. Momentum is calculated by taking the difference between the current value (`RET_1` or `VOLUME_1`) and the average value of the subsequent days within the window. This method provides an indication of how much the value has changed relative to the average over the window period.
#
#
# **Mathematical Expression**
#
# $$ \text{Momentum} = \text{Current Value} - \text{Average Value Over Window} $$
#
# Where:
# - $\text{Current Value}$ is the value on the first day of the window (`RET_1` or `VOLUME_1`).
# - $\text{Average Value Over Window}$ is the average of the values from the second day to the end of the window period.
#
# **Implementation Steps**
#
# 1. **Define the window size:** Set the window size for the calculation.
# 2. **Calculate the rolling mean:** Compute the rolling mean of the target columns (`RET` or `VOLUME`) from the second day to the end of the window period.
# 3. **Calculate the current mean:** Compute the mean of the first day's value (`RET_1` or `VOLUME_1`).
# 4. **Align the data:** Align the current mean and the rolling mean to ensure they match for the subtraction.
# 5. **Calculate momentum:** Subtract the rolling mean from the current mean to get the momentum.
# 6. **Rename and join:** Rename the resulting column to indicate momentum and join it back to the original dataset.
#
# This approach ensures that a single momentum value encapsulating the past 20 days is calculated for both `RET` and `VOLUME` for each stock.
#

# %%
import pandas as pd
import numpy as np

def calculate_momentum(data, window=10):
    """
    Calculate Momentum for both RET and VOLUME for each stock encapsulating the past 20 days.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', 'RET_i' columns (i from 1 to 20), and 'VOLUME_i' columns (i from 1 to 20).
    window: int, the lookback period for Momentum calculation (default is 10).

    Returns:
    pd.DataFrame with Momentum values for RET and VOLUME for each stock and a list of new feature names.
    """
    data = data.copy()
    new_features = []

    for target in ['RET', 'VOLUME']:
        window_size = window
        momentum_col_name = f'{target}_{window_size}_day_momentum'
        new_features.append(momentum_col_name)
        
        # Calculate the rolling mean of target columns and mean of the first day
        rolling_mean_target = data.groupby(by=['STOCK', 'DATE'])[[f'{target}_{day}' for day in range(2, window_size+1)]].mean()
        target_1_mean = data.groupby(by=['STOCK', 'DATE'])[[f'{target}_1']].mean()

        # Align the data for subtraction
        current_value_aligned, rolling_mean_value_aligned = target_1_mean.align(rolling_mean_target, axis=0, level='STOCK')
        momentum_value = current_value_aligned.sub(rolling_mean_value_aligned.mean(axis=1), axis=0)

        # Rename the column to indicate momentum
        momentum_value.rename(columns={f'{target}_1': momentum_col_name}, inplace=True)

        # Join the momentum back to the original data
        placeholder = data.join(momentum_value, on=['STOCK', 'DATE'], how='left')
        data[momentum_col_name] = placeholder[momentum_col_name]

    return data, new_features




# %%
# Example usage:
train, f = calculate_momentum(train, window=10)
test, _ = calculate_momentum(test, window=10)


# %%
new_features.extend(f)


# %%
new_features

# %%
train[f].isna().sum()

# %%
train[f]

# %%
test.shape

# %% [markdown]
# **Momentum per Sector**
#
# Momentum is also calculated for each sector by grouping the data by 'SECTOR' and 'DATE'.
#
# **Formula:**
# $$ \text{Momentum} = \text{Current Price} - \text{Price n periods ago} $$
#
#
# **Momentum Calculation**
#
# In this implementation, we calculate the momentum for both `RET` and `VOLUME` over a specified window period, grouped by `SECTOR` and `DATE`. Momentum is calculated by taking the difference between the current value (`RET_1` or `VOLUME_1`) and the average value of the subsequent days within the window. This method provides an indication of how much the value has changed relative to the average over the window period.
#
# **Mathematical Expression**
#
# $$ \text{Momentum} = \text{Current Value} - \text{Average Value Over Window} $$
#
# Where:
# - $\text{Current Value}$ is the value on the first day of the window (`RET_1` or `VOLUME_1`).
# - $\text{Average Value Over Window}$ is the average of the values from the second day to the end of the window period.
#
# **Implementation Steps**
#
# 1. **Define the window size:** Set the window size for the calculation.
# 2. **Calculate the rolling mean:** Compute the rolling mean of the target columns (`RET` or `VOLUME`) from the second day to the end of the window period.
# 3. **Calculate the current mean:** Compute the mean of the first day's value (`RET_1` or `VOLUME_1`).
# 4. **Align the data:** Align the current mean and the rolling mean to ensure they match for the subtraction.
# 5. **Calculate momentum:** Subtract the rolling mean from the current mean to get the momentum.
# 6. **Rename and join:** Rename the resulting column to indicate momentum and join it back to the original dataset.
#
# This approach ensures that a single momentum value encapsulating the past 20 days is calculated for both `RET` and `VOLUME` for each sector.
#
#

# %%
import pandas as pd
import numpy as np

def calculate_momentum_sector(data, window=10):
    """
    Calculate Momentum for both RET and VOLUME for each sector encapsulating the past 20 days.

    Args:
    data: pd.DataFrame, containing 'SECTOR', 'DATE', 'RET_i' columns (i from 1 to 20), and 'VOLUME_i' columns (i from 1 to 20).
    window: int, the lookback period for Momentum calculation (default is 10).

    Returns:
    pd.DataFrame with Momentum values for RET and VOLUME for each sector and a list of new feature names.
    """
    data = data.copy()
    new_features = []

    for target in ['RET', 'VOLUME']:
        window_size = window
        momentum_col_name = f'{target}_{window_size}_day_momentum_sector'
        new_features.append(momentum_col_name)
        
        # Calculate the rolling mean of target columns and mean of the first day
        rolling_mean_target = data.groupby(by=['SECTOR', 'DATE'])[[f'{target}_{day}' for day in range(2, window_size+1)]].mean()
        target_1_mean = data.groupby(by=['SECTOR', 'DATE'])[[f'{target}_1']].mean()

        # Align the data for subtraction
        current_value_aligned, rolling_mean_value_aligned = target_1_mean.align(rolling_mean_target, axis=0, level='SECTOR')
        momentum_value = current_value_aligned.sub(rolling_mean_value_aligned.mean(axis=1), axis=0)

        # Rename the column to indicate momentum
        momentum_value.rename(columns={f'{target}_1': momentum_col_name}, inplace=True)

        # Join the momentum back to the original data
        placeholder = data.join(momentum_value, on=['SECTOR', 'DATE'], how='left')
        data[momentum_col_name] = placeholder[momentum_col_name]

    return data, new_features



# %%
# Example usage:
train, f = calculate_momentum_sector(train, window=10)
test, _ = calculate_momentum_sector(test, window=10)


# %%
new_features.extend(f)

# %%
new_features

# %%
train[f].isna().sum()

# %%
train[f]

# %%
test.shape

# %% [markdown]
# **4. Stochastic Oscillator**
#
# The Stochastic Oscillator is a momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time.
#
# **Formula:**
# $$ \%K = \frac{(C - L14)}{(H14 - L14)} \times 100 $$
# $$ \%D = \text{SMA of } \%K $$
#
#
# **Stochastic Oscillator Calculation**
#
# In this implementation, we calculate the Stochastic Oscillator using `RET_1` to `RET_3` instead of `High`, `Low`, and `Close` columns. The Stochastic Oscillator is a momentum indicator that compares a particular closing price of a security to a range of its prices over a certain period of time.
#
# **Mathematical Expressions**
#
# $$ \%K = \frac{\text{Close} - \text{L14}}{\text{H14} - \text{L14}} \times 100 $$
# $$ \%D = \text{SMA}(\%K, \text{smooth\_window}) $$
#
# Where:
# - $\text{Close}$ is the current return (e.g., `RET_1`).
# - $\text{L14}$ is the lowest return over the lookback period.
# - $\text{H14}$ is the highest return over the lookback period.
# - $\%K$ is the Stochastic Oscillator value.
# - $\%D$ is the smoothed value of \(\%K\).
#
# ### Implementation Steps
#
# 1. **Calculate Lowest and Highest Returns:** Compute the lowest and highest returns over the lookback period (`RET_1`, `RET_2`, `RET_3`).
# 2. **Calculate %K:** Calculate the Stochastic Oscillator value using the formula.
# 3. **Calculate %D:** Calculate the smoothed value of %K using a simple moving average (SMA).
# 4. **Return New Features:** Return the new feature names along with the modified DataFrame.
#
# This approach ensures that the Stochastic Oscillator values are calculated for each stock using the returns data.
#

# %%
import pandas as pd
import numpy as np

def calculate_stochastic_oscillator(data, window=14, smooth_window=3):
    """
    Calculate Stochastic Oscillator for each stock using RET_1, RET_3, RET_5, RET_10, and RET_20.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'RET_i' columns (i from 1 to 20).
    window: int, the lookback period for Stochastic Oscillator calculation.
    smooth_window: int, the smoothing window for %D calculation.

    Returns:
    pd.DataFrame with %K and %D values for each stock and a list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in [1, 5, 14]:
        ret_col = f'RET_{i}'
        low_col = f'L14_{i}'
        high_col = f'H14_{i}'
        k_col = f'%K_{i}'
        d_col = f'%D_{i}'
        # new_features.extend([low_col, high_col, k_col, d_col])
        new_features.extend([k_col, d_col])

        # Calculate the lowest low and highest high over the lookback period
        data[low_col] = data.groupby('STOCK')[ret_col].transform(lambda x: x.rolling(window, min_periods=1).min())
        data[high_col] = data.groupby('STOCK')[ret_col].transform(lambda x: x.rolling(window, min_periods=1).max())

        # Calculate %K
        data[k_col] = ((data[ret_col] - data[low_col]) / (data[high_col] - data[low_col])) * 100
        
        # Calculate %D (SMA of %K)
        data[d_col] = data.groupby('STOCK')[k_col].transform(lambda x: x.rolling(smooth_window, min_periods=1).mean())

        # Fill NaNs with the median for each stock group
        data[[k_col, d_col]] = data.groupby('STOCK')[[k_col, d_col]].transform(lambda x: x.fillna(x.median()))


        # Fill NaNs with the median for each sector group
        data[[k_col, d_col]] = data.groupby('SECTOR')[[k_col, d_col]].transform(lambda x: x.fillna(x.median()))

    return data, new_features



# %%
train, f = calculate_stochastic_oscillator(train)
test, _ = calculate_stochastic_oscillator(test)


# %%
new_features.extend(f)

# %%
train[f].isna().sum()

# %%
train[f]

# %%
test.shape

# %% [markdown]
# **5. MACD and MACD Divergence**
#
# The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.
#
# **Formula:**
#
# $$ \text{MACD} = \text{EMA}_{12} - \text{EMA}_{26} $$
# $$ \text{Signal Line} = \text{EMA}_{9} \text{ of MACD} $$
# $$ \text{MACD Divergence} = \text{MACD} - \text{Signal Line} $$
#
#
#
# **MACD Calculation**
#
# In this implementation, we calculate the MACD for `RET_1` to `RET_5` instead of `Close` prices. The MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.
#
# **Mathematical Expressions**
#
# 1. **Short-term EMA:**
# $ \text{EMA}_{12} = \text{EMA}_{short}(\text{RET}) $
#
# 2. **Long-term EMA:**
# $ \text{EMA}_{26} = \text{EMA}_{long}(\text{RET}) $
#
# 3. **MACD:**
# $ \text{MACD} = \text{EMA}_{12} - \text{EMA}_{26} $
#
# 4. **Signal Line:**
# $ \text{Signal Line} = \text{EMA}_{signal}(\text{MACD}) $
#
# 5. **MACD Divergence:**
# $ \text{MACD Divergence} = \text{MACD} - \text{Signal Line} $
#
# **Implementation Steps**
#
# 1. **Calculate Short-term EMA:** Compute the short-term EMA (`EMA_12_i`) for each `RET_i`.
# 2. **Calculate Long-term EMA:** Compute the long-term EMA (`EMA_26_i`) for each `RET_i`.
# 3. **Calculate MACD:** Calculate the MACD (`MACD_i`) as the difference between the short-term EMA and the long-term EMA.
# 4. **Calculate Signal Line:** Compute the signal line (`Signal_Line_i`) as the EMA of the MACD.
# 5. **Calculate MACD Divergence:** Calculate the MACD divergence (`MACD_Divergence_i`) as the difference between the MACD and the signal line.
# 6. **Return New Features:** Return the new feature names along with the modified DataFrame.
#
# This approach ensures that the MACD values are calculated for each stock using the returns data.
#
#

# %%
import pandas as pd
import numpy as np

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """
    Calculate MACD and MACD Divergence for each stock for RET_1 to RET_5.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'RET_i' columns (i from 1 to 5).
    short_window: int, the short window period for EMA calculation.
    long_window: int, the long window period for EMA calculation.
    signal_window: int, the signal line period for EMA calculation.

    Returns:
    pd.DataFrame with MACD, Signal Line, and MACD Divergence for each stock for RET_1 to RET_5.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in [1, 5, 14]:
        ret_col = f'RET_{i}'
        ema_12_col = f'EMA_12_{i}'
        ema_26_col = f'EMA_26_{i}'
        macd_col = f'MACD_{i}'
        signal_line_col = f'Signal_Line_{i}'
        macd_divergence_col = f'MACD_Divergence_{i}'

        # new_features.extend([ema_12_col, ema_26_col, macd_col, signal_line_col, macd_divergence_col])
        new_features.extend([macd_col, signal_line_col, macd_divergence_col])

        # Calculate short-term EMA
        data[ema_12_col] = data.groupby('STOCK')[ret_col].transform(lambda x: x.ewm(span=short_window, adjust=False).mean())
        
        # Calculate long-term EMA
        data[ema_26_col] = data.groupby('STOCK')[ret_col].transform(lambda x: x.ewm(span=long_window, adjust=False).mean())
        
        # Calculate MACD
        data[macd_col] = data[ema_12_col] - data[ema_26_col]
        
        # Calculate Signal Line
        data[signal_line_col] = data.groupby('STOCK')[macd_col].transform(lambda x: x.ewm(span=signal_window, adjust=False).mean())
        
        # Calculate MACD Divergence
        data[macd_divergence_col] = data[macd_col] - data[signal_line_col]
    
    return data, new_features



# %%
train, f = calculate_macd(train)
test, _ = calculate_macd(test)

# %%
new_features.extend(f)

# %%
train[f].isna().sum()

# %%
train[f]


# %% [markdown]
# **6. Golden Cross**
#
# A Golden Cross is a bullish breakout pattern formed from a crossover involving a security's short-term moving average (such as the 10-day moving average) breaking above its long-term moving average (such as the 200-day moving average) or resistance level.
#

# %%
def calculate_golden_cross(data, short_window=10, long_window=200):
    """
    Calculate Golden Cross for each stock.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'Close' columns.
    short_window: int, the short window period for moving average calculation.
    long_window: int, the long window period for moving average calculation.

    Returns:
    pd.DataFrame with Golden Cross binary variable for each stock.
    """
    data = data.copy()
    
    # Calculate short-term SMA
    data[f'SMA_{short_window}_1'] = data.groupby(['STOCK'])['RET_1'].transform(lambda x: x.rolling(window=short_window, min_periods=1).mean())
    
    # Calculate long-term SMA
    data[f'SMA_{long_window}_1'] = data.groupby(['STOCK'])['RET_1'].transform(lambda x: x.rolling(window=long_window, min_periods=1).mean())
    
    # Determine Golden Cross
    data['Golden_Cross_1'] = np.where(data[f'SMA_{short_window}_1'] > data[f'SMA_{long_window}_1'], 1, 0)

    # features = [f'SMA_{short_window}_1', f'SMA_{long_window}_1', 'Golden_Cross_1']
    features = ['Golden_Cross_1']

    return data, features



# %%
train, f = calculate_golden_cross(train)
test, _ = calculate_golden_cross(test)

# %%
new_features.extend(f)

# %%
train[f].isna().sum().sum()

# %%
train[f]

# %%
train['Golden_Cross_1'].sum()

# %% [markdown]
# **7. Bollinger Bands**
#
# Bollinger Bands consist of a middle band being a simple moving average (SMA) and an upper and lower band at a specified number of standard deviations above and below the middle band.
#
# **Formula:**
# $$ \text{Middle Band} = \text{SMA}(n) $$
# $$ \text{Upper Band} = \text{Middle Band} + (K \times \text{Standard Deviation}) $$
# $$ \text{Lower Band} = \text{Middle Band} - (K \times \text{Standard Deviation}) $$
#

# %% [markdown]
# **Bollinger Bands Calculation**
#
# In this implementation, we calculate the Bollinger Bands for `RET_1` to `RET_5`. Bollinger Bands are volatility bands placed above and below a moving average. Volatility is based on the standard deviation, which changes as volatility increases and decreases.
#
# **Mathematical Expressions**
#
# 1. **Simple Moving Average (SMA):**
# $$ \text{SMA} = \frac{\sum \text{RET}}{n} $$
#
# 2. **Standard Deviation (STD):**
# $$ \text{STD} = \sqrt{\frac{\sum (\text{RET} - \text{SMA})^2}{n}} $$
#
# 3. **Upper Band:**
# $$ \text{Upper Band} = \text{SMA} + (\text{num\_std\_dev} \times \text{STD}) $$
#
# 4. **Lower Band:**
# $$ \text{Lower Band} = \text{SMA} - (\text{num\_std\_dev} \times \text{STD}) $$
#
# 5. **Band Distance:**
# $$ \text{Band Distance} = \text{Upper Band} - \text{Lower Band} $$
#
# ### Implementation Steps
#
# 1. **Calculate SMA:** Compute the simple moving average (`SMA_i`) for each `RET_i`.
# 2. **Calculate Standard Deviation:** Compute the standard deviation (`STD_i`) for each `RET_i`.
# 3. **Calculate Bollinger Bands:** Calculate the upper (`Upper_Band_i`) and lower (`Lower_Band_i`) bands using the SMA and STD.
# 4. **Calculate Band Distance:** Compute the distance between the upper and lower bands (`Band_Distance_i`).
# 5. **Return New Features:** Return the new feature names along with the modified DataFrame.
#
# This approach ensures that the Bollinger Bands are calculated for each stock using the returns data.
#

# %%
import pandas as pd
import numpy as np

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    """
    Calculate Bollinger Bands for each stock for RET_1 to RET_5.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'RET_i' columns (i from 1 to 5).
    window: int, the lookback period for SMA calculation.
    num_std_dev: int, the number of standard deviations for band calculation.

    Returns:
    pd.DataFrame with Bollinger Bands and distance between bands for each stock for RET_1 to RET_5.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in [1, 5, 14]:
        ret_col = f'RET_{i}'
        sma_col = f'SMA_{i}'
        std_col = f'STD_{i}'
        upper_band_col = f'Upper_Band_{i}'
        lower_band_col = f'Lower_Band_{i}'
        band_distance_col = f'Band_Distance_{i}'

        # new_features.extend([sma_col, std_col, upper_band_col, lower_band_col, band_distance_col])
        new_features.extend([upper_band_col, lower_band_col, band_distance_col])

        # Calculate SMA
        data[sma_col] = data.groupby('STOCK')[ret_col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        
        # Calculate Standard Deviation
        data[std_col] = data.groupby('STOCK')[ret_col].transform(lambda x: x.rolling(window, min_periods=1).std())
        data = data.fillna(data.median())

        # Calculate Bollinger Bands
        data[upper_band_col] = data[sma_col] + (num_std_dev * data[std_col])
        data[lower_band_col] = data[sma_col] - (num_std_dev * data[std_col])
        
        # Calculate distance between bands
        data[band_distance_col] = data[upper_band_col] - data[lower_band_col]
    
    return data, new_features



# %%
train, f = calculate_bollinger_bands(train)
test, _ = calculate_bollinger_bands(test)

# %%
new_features.extend(f)

# %%
f

# %%
train.isna().sum().sum()

# %%
train[f]

# %% [markdown]
# **8. Cumulative Return**
#
# Cumulative return is the total change in the price of an asset over a period, expressed as a percentage.
#

# %%
import pandas as pd

# Function to calculate cumulative returns
def calculate_cumulative_returns(df):
    feat_ = []
    for day in range(1, 21):
        cum_return_col = f'CUM_RET_{day}'
        feat_.append(cum_return_col)
        df[cum_return_col] = df.groupby('STOCK')[f'RET_{day}'].transform(lambda x: (1 + x).cumprod() - 1)
    return df, feat_

# Apply cumulative returns calculation to train and test datasets
train, cum_return_col = calculate_cumulative_returns(train)
test, _ = calculate_cumulative_returns(test)

print("Cumulative returns calculated.")

# %%
# new_features.extend(cum_return_col)

# Not adding the cumulative returns

# %%
train[cum_return_col].isna().sum().sum()

# %%
train[cum_return_col]

# %% [markdown]
# **9. Bollinger Bands for Cum Ret**

# %%
import pandas as pd
import numpy as np

def calculate_bollinger_bands_cum_ret(data, window=20, num_std_dev=2):
    """
    Calculate Bollinger Bands for each stock for CUM_RET_1 to CUM_RET_5.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'CUM_RET_i' columns (i from 1 to 5).
    window: int, the lookback period for SMA calculation.
    num_std_dev: int, the number of standard deviations for band calculation.

    Returns:
    pd.DataFrame with Bollinger Bands and distance between bands for each stock for CUM_RET_1 to CUM_RET_5.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in [1, 5, 14]:
        cum_ret_col = f'CUM_RET_{i}'
        sma_col = f'SMA_CUM_{i}'
        std_col = f'STD_CUM_{i}'
        upper_band_col = f'Upper_Band_CUM_{i}'
        lower_band_col = f'Lower_Band_CUM_{i}'
        band_distance_col = f'Band_Distance_CUM_{i}'

        # new_features.extend([sma_col, std_col, upper_band_col, lower_band_col, band_distance_col])
        new_features.extend([band_distance_col])

        # Calculate SMA
        data[sma_col] = data.groupby('STOCK')[cum_ret_col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        
        # Calculate Standard Deviation
        data[std_col] = data.groupby('STOCK')[cum_ret_col].transform(lambda x: x.rolling(window, min_periods=1).std())

        # Fill NaNs with the median for each stock group
        data[[sma_col, std_col]] = data.groupby('STOCK')[[sma_col, std_col]].transform(lambda x: x.fillna(x.median()))

        # Calculate Bollinger Bands
        data[upper_band_col] = data[sma_col] + (num_std_dev * data[std_col])
        data[lower_band_col] = data[sma_col] - (num_std_dev * data[std_col])
        
        # Calculate distance between bands
        data[band_distance_col] = data[upper_band_col] - data[lower_band_col]


        # Fill NaNs with the median for each stock group
        data[[std_col, upper_band_col,lower_band_col , band_distance_col]] = data.groupby('STOCK')[[std_col, upper_band_col,lower_band_col, band_distance_col ]].transform(lambda x: x.fillna(x.median()))


        # Fill NaNs with the median for each sector group
        data[[std_col, upper_band_col,lower_band_col , band_distance_col]] = data.groupby('SECTOR')[[std_col, upper_band_col,lower_band_col , band_distance_col]].transform(lambda x: x.fillna(x.median()))
    
    return data, new_features

# Example usage:
# df_with_bbands_cum, new_features_bbands_cum = calculate_bollinger_bands_cum_ret(df)


# %%
train, f = calculate_bollinger_bands_cum_ret(train)
test, _ = calculate_bollinger_bands_cum_ret(test)

# %%
new_features.extend(f)

# %%
train[f].isna().sum()

# %%
train[f]

# %% [markdown]
# **10. Money Flow Index (MFI)**
#
# The Money Flow Index (MFI) is a momentum indicator that uses both price and volume data to measure buying and selling pressure.
#
# **Formula:**
# $$ \text{Typical Price} = \frac{\text{High} + \text{Low} + \text{Close}}{3} $$
# $$ \text{Raw Money Flow} = \text{Typical Price} \times \text{Volume} $$
# $$ \text{Money Flow Ratio} = \frac{\text{Positive Money Flow}}{\text{Negative Money Flow}} $$
# $$ \text{MFI} = 100 - \frac{100}{1 + \text{Money Flow Ratio}} $$
#

# %% [markdown]
# **Money Flow Index (MFI) Calculation**
#
# **We do not have prices, but rather the daily returns!  Will use this as proxy!**
#
# In this implementation, we adapt the calculation of the Money Flow Index (MFI) to use `RET_1`, `RET_10`, and `RET_20` as proxies for typical price changes. The MFI is a momentum indicator that measures the inflow and outflow of money into a security over a specific period.
#
# **Mathematical Expressions**
#
# 1. **Raw Money Flow:**
# $$ \text{Raw Money Flow}_i = \text{RET}_i \times \text{VOLUME}_i $$
#
# 2. **Positive and Negative Money Flow:**
# $$ \text{Positive Money Flow}_i = \begin{cases} 
# \text{Raw Money Flow}_i & \text{if} \ \text{RET}_i > 0 \\
# 0 & \text{otherwise}
# \end{cases} $$
# $$ \text{Negative Money Flow}_i = \begin{cases} 
# \text{Raw Money Flow}_i & \text{if} \ \text{RET}_i < 0 \\
# 0 & \text{otherwise}
# \end{cases} $$
#
# 3. **Money Flow Ratio:**
# $$ \text{Money Flow Ratio}_i = \frac{\sum \text{Positive Money Flow}_i}{\sum \text{Negative Money Flow}_i} $$
#
# 4. **Money Flow Index (MFI):**
# $$ \text{MFI}_i = 100 - \frac{100}{1 + \text{Money Flow Ratio}_i} $$
#
# **Implementation Steps**
#
# 1. **Calculate Raw Money Flow:** Compute the raw money flow using `RET_i` and `VOLUME_i`.
# 2. **Calculate Positive and Negative Money Flow:** Determine the positive and negative money flows based on the changes in `RET_i`.
# 3. **Calculate Money Flow Ratio:** Compute the money flow ratio using the rolling sums of positive and negative money flows.
# 4. **Calculate MFI:** Calculate the Money Flow Index (MFI) using the money flow ratio.
# 5. **Add Overbought and Oversold Signals:** Determine overbought and oversold conditions based on the MFI values.
# 6. **Return New Features:** Return the new feature names along with the modified DataFrame.
#
# This approach ensures that the MFI is calculated for each stock using `RET_1`, `RET_10`, and `RET_20` as proxies for price changes and volume.
#

# %%
import pandas as pd
import numpy as np

def calculate_mfi(data, window=14):
    """
    Calculate MFI for each stock for RET_1, RET_10, and RET_20.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', 'RET_i' and 'VOLUME_i' columns (i being 1, 10, and 20).
    window: int, the lookback period for MFI calculation.

    Returns:
    pd.DataFrame with MFI values for each stock.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in [1, 10, 20]:
        ret_col = f'RET_{i}'
        volume_col = f'VOLUME_{i}'
        typical_price_col = f'Typical_Price_{i}'
        money_flow_col = f'Money_Flow_{i}'
        positive_money_flow_col = f'Positive_Money_Flow_{i}'
        negative_money_flow_col = f'Negative_Money_Flow_{i}'
        money_flow_ratio_col = f'Money_Flow_Ratio_{i}'
        mfi_col = f'MFI_{i}'
        overbought_col = f'overbought_mfi_{i}'
        oversold_col = f'oversold_mfi_{i}'

        # new_features.extend([typical_price_col, money_flow_col, positive_money_flow_col, negative_money_flow_col, money_flow_ratio_col, mfi_col, overbought_col, oversold_col])
        new_features.extend([mfi_col, overbought_col, oversold_col])

        # Calculate Typical Price using RET_i
        data[typical_price_col] = data[ret_col]
        
        # Calculate Money Flow
        data[money_flow_col] = data[typical_price_col] * data[volume_col]
        
        # Calculate Positive and Negative Money Flow
        data[positive_money_flow_col] = np.where(data[typical_price_col].diff(1) > 0, data[money_flow_col], 0)
        data[negative_money_flow_col] = np.where(data[typical_price_col].diff(1) < 0, data[money_flow_col], 0)
        
        # Calculate Money Flow Ratio
        data[money_flow_ratio_col] = data.groupby('STOCK')[positive_money_flow_col].transform(lambda x: x.rolling(window, min_periods=1).sum()) / data.groupby('STOCK')[negative_money_flow_col].transform(lambda x: x.rolling(window, min_periods=1).sum())
        
        # Calculate MFI
        data[mfi_col] = 100 - (100 / (1 + data[money_flow_ratio_col]))
        
        # Add overbought and oversold signals
        data[overbought_col] = np.where(data[mfi_col] > 80, 1, 0)
        data[oversold_col] = np.where(data[mfi_col] < 20, 1, 0)

        # Fill NaNs with the median for each stock group
        data[[money_flow_ratio_col, mfi_col]] = data.groupby('STOCK')[[money_flow_ratio_col, mfi_col]].transform(lambda x: x.fillna(x.median()))

        # Fill NaNs with the median for each sector group
        data[[money_flow_ratio_col, mfi_col]] = data.groupby('SECTOR')[[money_flow_ratio_col, mfi_col]].transform(lambda x: x.fillna(x.median()))

    return data, new_features

# Example usage with your dataset:
# Assuming train and test are your actual datasets

# Apply the function
train, mfi_features_train = calculate_mfi(train)
test, _ = calculate_mfi(test)


# %%
new_features.extend(mfi_features_train)

# %%
train[mfi_features_train].isna().sum()

# %%
train[['MFI_1', 'MFI_10', 'MFI_20']].describe()


# %%
# Count values below 0 and above 100
below_0 = (train[['MFI_1', 'MFI_10', 'MFI_20']] < 0).sum().sum()
above_100 = (train[['MFI_1', 'MFI_10', 'MFI_20']] > 100).sum().sum()

print(f"Values below 0: {below_0}")
print(f"Values above 100: {above_100}")




# %%
# Clip the MFI values to the range [0, 100]
train[['MFI_1', 'MFI_10', 'MFI_20']] = train[['MFI_1', 'MFI_10', 'MFI_20']].clip(lower=0, upper=100)

# Verify the changes
train[['MFI_1', 'MFI_10', 'MFI_20']].describe()

# %%
# Clip the MFI values to the range [0, 100]
test[['MFI_1', 'MFI_10', 'MFI_20']] = test[['MFI_1', 'MFI_10', 'MFI_20']].clip(lower=0, upper=100)

# Verify the changes
test[['MFI_1', 'MFI_10', 'MFI_20']].describe()

# %%
train[mfi_features_train]

# %% [markdown]
# **MFI per Sector**
#
# The MFI is also calculated for each sector by grouping the data by 'SECTOR' and 'DATE'.
#
# **Formula:**
# $$ \text{Typical Price} = \frac{\text{High} + \text{Low} + \text{Close}}{3} $$
# $$ \text{Raw Money Flow} = \text{Typical Price} \times \text{Volume} $$
# $$ \text{Money Flow Ratio} = \frac{\text{Positive Money Flow}}{\text{Negative Money Flow}} $$
# $$ \text{MFI} = 100 - \frac{100}{1 + \text{Money Flow Ratio}} $$
#

# %% [markdown]
# **Money Flow Index (MFI) Calculation Per Sector**
#
# In this implementation, we adapt the calculation of the Money Flow Index (MFI) to use `RET_1`, `RET_10`, and `RET_20` as proxies for typical price changes. The MFI is a momentum indicator that measures the inflow and outflow of money into a security over a specific period. This calculation is done per sector.
#
# **Mathematical Expressions**
#
# 1. **Raw Money Flow:**
# $$ \text{Raw Money Flow}_i = \text{RET}_i \times \text{VOLUME}_i $$
#
# 2. **Positive and Negative Money Flow:**
# $$ \text{Positive Money Flow}_i = \begin{cases} 
# \text{Raw Money Flow}_i & \text{if} \ \text{RET}_i > 0 \\
# 0 & \text{otherwise}
# \end{cases} $$
# $$ \text{Negative Money Flow}_i = \begin{cases} 
# \text{Raw Money Flow}_i & \text{if} \ \text{RET}_i < 0 \\
# 0 & \text{otherwise}
# \end{cases} $$
#
# 3. **Money Flow Ratio:**
# $$ \text{Money Flow Ratio}_i = \frac{\sum \text{Positive Money Flow}_i}{\sum \text{Negative Money Flow}_i} $$
#
# 4. **Money Flow Index (MFI):**
# $$ \text{MFI}_i = 100 - \frac{100}{1 + \text{Money Flow Ratio}_i} $$
#
# **Implementation Steps**
#
# 1. **Calculate Raw Money Flow:** Compute the raw money flow using `RET_i` and `VOLUME_i`.
# 2. **Calculate Positive and Negative Money Flow:** Determine the positive and negative money flows based on the changes in `RET_i`.
# 3. **Calculate Money Flow Ratio:** Compute the money flow ratio using the rolling sums of positive and negative money flows.
# 4. **Calculate MFI:** Calculate the Money Flow Index (MFI) using the money flow ratio.
# 5. **Add Overbought and Oversold Signals:** Determine overbought and oversold conditions based on the MFI values.
# 6. **Return New Features:** Return the new feature names along with the modified DataFrame.
#
# This approach ensures that the MFI is calculated for each sector using `RET_1`, `RET_10`, and `RET_20` as proxies for price changes and volume.
#

# %%
import pandas as pd
import numpy as np

def calculate_mfi_per_sector(data, window=14):
    """
    Calculate MFI for each sector for RET_1, RET_10, and RET_20.

    Args:
    data: pd.DataFrame, containing 'SECTOR', 'DATE', 'RET_i' and 'VOLUME_i' columns (i being 1, 10, and 20).
    window: int, the lookback period for MFI calculation.

    Returns:
    pd.DataFrame with MFI values for each sector.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in [1, 10, 20]:
        ret_col = f'RET_{i}'
        volume_col = f'VOLUME_{i}'
        raw_money_flow_col = f'Raw_Money_Flow_{i}_sector'
        positive_money_flow_col = f'Positive_Money_Flow_{i}_sector'
        negative_money_flow_col = f'Negative_Money_Flow_{i}_sector'
        money_flow_ratio_col = f'Money_Flow_Ratio_{i}_sector'
        mfi_col = f'MFI_{i}_sector'
        overbought_col = f'overbought_mfi_{i}_sector'
        oversold_col = f'oversold_mfi_{i}_sector'

        # new_features.extend([raw_money_flow_col, positive_money_flow_col, negative_money_flow_col, money_flow_ratio_col, mfi_col, overbought_col, oversold_col])
        new_features.extend([mfi_col, overbought_col, oversold_col])

        # Calculate Raw Money Flow using RET_i and VOLUME_i
        data[raw_money_flow_col] = data[ret_col] * data[volume_col]
        
        # Calculate Positive and Negative Money Flow
        data[positive_money_flow_col] = np.where(data[ret_col].diff(1) > 0, data[raw_money_flow_col], 0)
        data[negative_money_flow_col] = np.where(data[ret_col].diff(1) < 0, data[raw_money_flow_col], 0)
        
        # Calculate Money Flow Ratio
        data[money_flow_ratio_col] = data.groupby(['SECTOR', 'DATE'])[positive_money_flow_col].transform(lambda x: x.rolling(window, min_periods=1).sum()) / data.groupby(['SECTOR', 'DATE'])[negative_money_flow_col].transform(lambda x: x.rolling(window, min_periods=1).sum())
        
        # Calculate MFI
        data[mfi_col] = 100 - (100 / (1 + data[money_flow_ratio_col]))
        
        # Add overbought and oversold signals
        data[overbought_col] = np.where(data[mfi_col] > 90, 1, 0)
        data[oversold_col] = np.where(data[mfi_col] < 10, 1, 0)

        # Fill NaNs with the median for each stock group
        data[[money_flow_ratio_col,mfi_col ]] = data.groupby('STOCK')[[money_flow_ratio_col,mfi_col]].transform(lambda x: x.fillna(x.median()))


        # Fill NaNs with the median for each sector group
        data[[money_flow_ratio_col,mfi_col]] = data.groupby('SECTOR')[[money_flow_ratio_col,mfi_col]].transform(lambda x: x.fillna(x.median()))
    
    
    
    return data, new_features



# %%
train, f = calculate_mfi_per_sector(train)
test, _ = calculate_mfi_per_sector(test)
new_features.extend(f)

# %%
train[f].isna().sum()


# %%
# Count values below 0 and above 100
below_0 = (train[['MFI_1_sector', 'MFI_10_sector', 'MFI_20_sector']] < 0).sum().sum()
above_100 = (train[['MFI_1_sector', 'MFI_10_sector', 'MFI_20_sector']] > 100).sum().sum()

print(f"Values below 0: {below_0}")
print(f"Values above 100: {above_100}")


# %%
# Clip the MFI values to the range [0, 100]
train[['MFI_1_sector', 'MFI_10_sector', 'MFI_20_sector']] = train[['MFI_1_sector', 'MFI_10_sector', 'MFI_20_sector']].clip(lower=0, upper=100)

# Verify the changes
train[['MFI_1_sector', 'MFI_10_sector', 'MFI_20_sector']].describe()

# %%
# Clip the MFI values to the range [0, 100]
test[['MFI_1_sector', 'MFI_10_sector', 'MFI_20_sector']] = test[['MFI_1_sector', 'MFI_10_sector', 'MFI_20_sector']].clip(lower=0, upper=100)

# Verify the changes
test[['MFI_1_sector', 'MFI_10_sector', 'MFI_20_sector']].describe()

# %%
train[f]

# %% [markdown]
# **11. Exponential Moving Average (EMA) and Simple Moving Average (SMA)**
#
# EMA and SMA are both used to smooth out price data and identify trends.
#
# **Formula for EMA:**
# $$ \text{EMA} = \text{Price}_t \times \frac{2}{n+1} + \text{EMA}_{t-1} \times \left(1 - \frac{2}{n+1}\right) $$
#
# **Formula for SMA:**
# $$ \text{SMA} = \frac{\sum_{i=1}^{n} \text{Price}_i}{n} $$
#
#
# We do not have the prices, but we are using the returns as proxies.
#

# %% [markdown]
# **EMA and SMA Calculation**
#
# In this implementation, we calculate the Exponential Moving Average (EMA) and Simple Moving Average (SMA) for `RET_1` to `RET_5` and `CUM_RET_1` to `CUM_RET_5`. These calculations help smooth out the data to identify trends over a specified window period.
#
# **Mathematical Expressions**
#
# 1. **Exponential Moving Average (EMA):**
# $$ \text{EMA} = \text{Current Price} \times \frac{2}{N+1} + \text{EMA Previous Day} \times \left(1 - \frac{2}{N+1}\right) $$
#
# 2. **Simple Moving Average (SMA):**
# $$ \text{SMA} = \frac{\sum \text{Price}}{N} $$
#
# Where \(N\) is the window period.
#
# **Implementation Steps**
#
# 1. **Calculate EMA:**
#    - Compute the EMA for daily returns (`EMA_RET_i`) and cumulative returns (`EMA_CUM_RET_i`) for each `RET_i` and `CUM_RET_i`.
# 2. **Calculate SMA:**
#    - Compute the SMA for daily returns (`SMA_RET_i`) and cumulative returns (`SMA_CUM_RET_i`) for each `RET_i` and `CUM_RET_i`.
# 3. **Return New Features:**
#    - Return the new feature names along with the modified DataFrame.
#
# This approach ensures that the EMA and SMA are calculated for each stock using the returns data.
#

# %%
import pandas as pd
import numpy as np

def calculate_ema(data, window=20):
    """
    Calculate EMA for each stock for RET_1 to RET_5 and CUM_RET_1 to CUM_RET_5.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', 'RET_i' and 'CUM_RET_i' columns (i from 1 to 5).
    window: int, the lookback period for EMA calculation.

    Returns:
    pd.DataFrame with EMA values for each stock.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in range(1, 6):
        ret_col = f'RET_{i}'
        cum_ret_col = f'CUM_RET_{i}'
        ema_ret_col = f'EMA_{ret_col}'
        ema_cum_ret_col = f'EMA_{cum_ret_col}'

        new_features.extend([ema_ret_col, ema_cum_ret_col])
        
        # Calculate EMA of daily returns
        data[ema_ret_col] = data.groupby('STOCK')[ret_col].transform(lambda x: x.ewm(span=window, adjust=False).mean())
        
        # Calculate EMA of cumulative returns
        data[ema_cum_ret_col] = data.groupby('STOCK')[cum_ret_col].transform(lambda x: x.ewm(span=window, adjust=False).mean())
    
    return data, new_features

def calculate_sma(data, window=20):
    """
    Calculate SMA for each stock for RET_1 to RET_5 and CUM_RET_1 to CUM_RET_5.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', 'RET_i' and 'CUM_RET_i' columns (i from 1 to 5).
    window: int, the lookback period for SMA calculation.

    Returns:
    pd.DataFrame with SMA values for each stock.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in range(1, 6):
        ret_col = f'RET_{i}'
        cum_ret_col = f'CUM_RET_{i}'
        sma_ret_col = f'SMA_{ret_col}'
        sma_cum_ret_col = f'SMA_{cum_ret_col}'

        new_features.extend([sma_ret_col, sma_cum_ret_col])
        
        # Calculate SMA of daily returns
        data[sma_ret_col] = data.groupby('STOCK')[ret_col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        
        # Calculate SMA of cumulative returns
        data[sma_cum_ret_col] = data.groupby('STOCK')[cum_ret_col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    
    return data, new_features




# %%
train, f = calculate_ema(train)
test, _ = calculate_ema(test)
new_features.extend(f)



# %%
train[f].isna().sum()


# %%
train[f]

# %%

train, f = calculate_sma(train)
test, _ = calculate_sma(test)
new_features.extend(f)


# %%
train[f].isna().sum()


# %%
train[f]

# %% [markdown]
#
#
#
# **12. Average True Range (ATR)**
#
# ATR is a measure of volatility that captures the range of price movement for a stock over a given period.
# **Formula:**
# $$ \text{TR} = \max(\text{High} - \text{Low}, |\text{High} - \text{Previous Close}|, |\text{Low} - \text{Previous Close}|) $$
# $$ \text{ATR} = \text{SMA of TR} $$
#
# **Average True Range (ATR) Calculation**
#
#
# In this implementation, we calculate the Average True Range (ATR) for `RET_1`, `RET_3`, `RET_5`, `RET_10`, and `RET_20`. The ATR is a measure of volatility that captures the range of price movement for a security.
#
# **Mathematical Expressions**
#
# 1. **High, Low, and Close:**
#    - $ \text{High}_i = \max(\text{RET}_i) $ over the rolling window
#    - $ \text{Low}_i = \min(\text{RET}_i) $ over the rolling window
#    - $ \text{Close}_i = \text{RET}_i $
#
# 2. **True Range (TR):**
# $$ \text{TR} = \max(\text{High} - \text{Low}, |\text{High} - \text{Close Previous}|, |\text{Low} - \text{Close Previous}|) $$
#
# 3. **Average True Range (ATR):**
# $$ \text{ATR} = \text{SMA}(\text{TR}, \text{window}) $$
#
# **Implementation Steps**
#
# 1. **Calculate High, Low, and Close:** Compute the high and low values over the rolling window, and set the close value as `RET_i`.
# 2. **Calculate True Range (TR):** Calculate the true range using the high, low, and close values.
# 3. **Fill NaNs:** Replace NaNs in the true range calculation with the median of the `TR` column for each stock.
# 4. **Calculate ATR:** Compute the average true range using a rolling mean of the true range.
# 5. **Return New Features:** Return the new feature names along with the modified DataFrame.
#
# This approach ensures that the ATR is calculated for each stock using the returns data, and NaNs are handled more appropriately by filling with the median of the `TR` values.

# %%
import pandas as pd
import numpy as np

def calculate_atr(df, window=14):
    """
    Calculate ATR per stock.

    Args:
    df: pd.DataFrame, containing 'STOCK', 'DATE', and 'RET_i' columns.
    window: int, the lookback period for ATR calculation.

    Returns:
    pd.DataFrame with ATR values for each stock.
    list of new feature names.
    """
    df = df.copy()
    new_features = []

    for day in [1, 5, 14]:
        # Calculate High, Low, and Close
        df[f'HIGH_{day}'] = df.groupby('STOCK')[f'RET_{day}'].transform(lambda x: x.rolling(window=window, min_periods=1).max())
        df[f'LOW_{day}'] = df.groupby('STOCK')[f'RET_{day}'].transform(lambda x: x.rolling(window=window, min_periods=1).min())
        df[f'CLOSE_{day}'] = df[f'RET_{day}']

        # Calculate True Range (TR)
        df[f'TR_{day}'] = df.groupby('STOCK').apply(
            lambda x: np.maximum(
                x[f'HIGH_{day}'] - x[f'LOW_{day}'],
                np.abs(x[f'HIGH_{day}'] - x[f'CLOSE_{day}'].shift(1)),
                np.abs(x[f'LOW_{day}'] - x[f'CLOSE_{day}'].shift(1))
            )
        ).reset_index(level=0, drop=True)

        # Fill NaNs with the median of the TR column
        df[f'TR_{day}'] = df.groupby('STOCK')[f'TR_{day}'].transform(lambda x: x.fillna(x.median()))
        
        # Calculate Average True Range (ATR)
        df[f'ATR_{day}'] = df.groupby('STOCK')[f'TR_{day}'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        new_features.append(f'ATR_{day}')

        # Fill NaNs with the median for each stock group
        df[[f'ATR_{day}' ]] = df.groupby('STOCK')[[f'ATR_{day}']].transform(lambda x: x.fillna(x.median()))


        # Fill NaNs with the median for each sector group
        df[[f'ATR_{day}']] = df.groupby('SECTOR')[[f'ATR_{day}']].transform(lambda x: x.fillna(x.median()))
    
    

    return df, new_features

# Example usage:
# df_with_atr, new_features_atr = calculate_atr(df)

# %%
train, f = calculate_atr(train)
test, _ = calculate_atr(test)


# %%
new_features.extend(f)

# %%
train[f].isna().sum()


# %%
train[f]

# %% [markdown]
# ## Volume related features

# %% [markdown]
# **1. Volume Deviation**
#
# Volume Deviation measures how much the volume of a particular stock deviates from the average relative volume of all stocks on a given day.
#
# **Formula:**
# $$ \text{Deviation}_{RVOL} = \text{Volume} - \text{Average Volume} $$

# %%
import pandas as pd

def calculate_deviation_from_avg_relative_volume(train, test, days=[1, 5, 14]):
    """
    Calculate Deviation from Average Relative Volume for VOLUME_1 to VOLUME_20.

    Args:
    train: pd.DataFrame, training data containing 'STOCK', 'DATE', and 'VOLUME_i' columns.
    test: pd.DataFrame, testing data containing 'STOCK', 'DATE', and 'VOLUME_i' columns.
    days: list, list of days to calculate deviation for (default: [1, 3, 5, 10, 20]).

    Returns:
    pd.DataFrame for train and test with added columns for average relative volume and deviation.
    list of new feature names.
    """
    new_features_volume = []

    for day in days:
        volume_col = f'VOLUME_{day}'
        avg_rvol_col = f'AVG_RVOL_{day}'
        deviation_rvol_col = f'DEVIATION_RVOL_{day}'
        new_features_volume.extend([avg_rvol_col, deviation_rvol_col])

        for df in [train, test]:
            # Calculate average relative volume across all stocks for each day
            df[avg_rvol_col] = df.groupby('DATE')[volume_col].transform('mean')

            # Calculate deviation from average relative volume
            df[deviation_rvol_col] = df[volume_col] - df[avg_rvol_col]

    return train, test, new_features_volume



# %%
train, test, f = calculate_deviation_from_avg_relative_volume(train, test)


# %%
new_features.extend(f)

# %%
train[f].isna().sum()


# %%
train[f]


# %% [markdown]
# **2. Volume Spike Detection**
# Volume Spike Detection identifies significant spikes in volume by comparing the deviation from the average relative volume to a threshold multiple of the standard deviation.
# **Formula:**
# $$ \text{Spike} = \begin{cases}
# 1 & \text{if } \left|\text{Deviation}_{RVOL}\right| > \text{Threshold} \times \sigma \\
# 0 & \text{otherwise}
# \end{cases} $$

# %%
def detect_volume_spikes(df, threshold=2):
    new_features_volume = []

    for day in [1, 5, 14]: #  range(1, 21):
        deviation_rvol_col = f'DEVIATION_RVOL_{day}'
        spike_col = f'SPIKE_{day}'
        new_features_volume.append(spike_col)

        # Detect volume spikes
        df[spike_col] = (np.abs(df[deviation_rvol_col]) > threshold * df.groupby('STOCK')[deviation_rvol_col].transform(lambda x: x.rolling(window=20).std())).astype(int)
    return df, new_features_volume



# %%
train, f = detect_volume_spikes(train)
test, _ = detect_volume_spikes(test)


# %%
new_features.extend(f)

# %%
train[f].isna().sum()


# %%
train[f]

# %% [markdown]
# **3. Accumulation/Distribution Line (ADL)**
#
# The ADL measures the cumulative flow of money into and out of a security.
# **Formula:**
# $$ \text{Money Flow Multiplier} = \frac{(\text{Close} - \text{Low}) - (\text{High} - \text{Close})}{\text{High} - \text{Low}} $$
# $$ \text{Money Flow Volume} = \text{Money Flow Multiplier} \times \text{Volume} $$
# $$ \text{ADL} = \sum \text{Money Flow Volume} $$

# %% [markdown]
# **Accumulation/Distribution Line (ADL) Calculation**
#
# In this implementation, we calculate the Accumulation/Distribution Line (ADL) for `RET_1` to `RET_5`. The ADL is a cumulative measure of the volume flowing into or out of a security.
#
# **Mathematical Expressions**
#
# 1. **High, Low, and Close:**
#    - $\text{High}_i = \max(\text{RET}_i) $ over the rolling window
#    - $ \text{Low}_i = \min(\text{RET}_i) $ over the rolling window
#    - $ \text{Close}_i = \text{RET}_i $
#
# 2. **Money Flow Multiplier:**
# $$ \text{Money Flow Multiplier}_i = \frac{(\text{Close}_i - \text{Low}_i) - (\text{High}_i - \text{Close}_i)}{\text{High}_i - \text{Low}_i + \epsilon} $$
# Where \(\epsilon\) is a small value to prevent division by zero.
#
# 3. **Money Flow Volume:**
# $$ \text{Money Flow Volume}_i = \text{Money Flow Multiplier}_i \times \text{Volume}_i $$
#
# 4. **Accumulation/Distribution Line (ADL):**
# $$ \text{ADL}_i = \sum \text{Money Flow Volume}_i $$
#
# **Implementation Steps**
#
# 1. **Calculate High, Low, and Close:** Compute the high and low values over the rolling window, and set the close value as `RET_i`.
# 2. **Calculate Money Flow Multiplier:** Determine the money flow multiplier for each return column, ensuring no division by zero using a small epsilon value.
# 3. **Calculate Money Flow Volume:** Compute the money flow volume using the money flow multiplier and volume data.
# 4. **Fill NaNs:** Replace NaNs in the money flow volume calculation with the median of the column for each stock.
# 5. **Calculate ADL:** Calculate the accumulation/distribution line as the cumulative sum of the money flow volume.
# 6. **Return New Features:** Return the new feature names along with the modified DataFrame.
#
# This approach ensures that the ADL is calculated for each stock using the returns and volume data, and NaNs are handled more appropriately by filling with the median of the Money Flow Volume values.
#

# %%
import pandas as pd
import numpy as np

def calculate_adl(df, window=14):
    """
    Calculate ADL for each stock.
    
    Args:
    df: pd.DataFrame, containing 'STOCK', 'DATE', 'RET_i' (i from 1 to 5) and 'VOLUME_i' columns.
    
    Returns:
    pd.DataFrame with ADL values for each stock.
    list of new feature names.
    """
    df = df.copy()
    new_features = []
    epsilon = 1e-10  # Small value to prevent division by zero

    for i in [1, 3, 5, 10]:
        high_col = f'HIGH_{i}'
        low_col = f'LOW_{i}'
        close_col = f'RET_{i}'
        volume_col = f'VOLUME_{i}'
        money_flow_multiplier_col = f'Money_Flow_Multiplier_{i}'
        money_flow_volume_col = f'Money_Flow_Volume_{i}'
        adl_col = f'ADL_{i}'

        # new_features.extend([money_flow_multiplier_col, money_flow_volume_col, adl_col])
        new_features.extend([adl_col])

        # Calculate High, Low, and Close as the rolling max, min, and close
        df[high_col] = df.groupby('STOCK')[close_col].transform(lambda x: x.rolling(window=window, min_periods=1).max())
        df[low_col] = df.groupby('STOCK')[close_col].transform(lambda x: x.rolling(window=window, min_periods=1).min())
        
        # Calculate Money Flow Multiplier
        df[money_flow_multiplier_col] = ((df[close_col] - df[low_col]) - (df[high_col] - df[close_col])) / (df[high_col] - df[low_col] + epsilon)

        # Calculate Money Flow Volume
        df[money_flow_volume_col] = df[money_flow_multiplier_col] * df[volume_col]

        # Fill NaNs with the median of the Money Flow Volume column
        df[money_flow_volume_col] = df.groupby('STOCK')[money_flow_volume_col].transform(lambda x: x.fillna(x.median()))

        # Calculate ADL
        df[adl_col] = df.groupby('STOCK')[money_flow_volume_col].cumsum()
    
    return df, new_features




# %%
train, f = calculate_adl(train)
test, _ = calculate_adl(test)
new_features.extend(f)

# %%
train[f]

# %%
train[f].isna().sum()


# %%

# %% [markdown]
# **4. Relative Volume**
#
# Relative Volume compares the current volume to the average volume over a specified period.
#
# **Formula:**
#
# $$ \text{Relative Volume} = \frac{\text{Current Volume}}{\text{Average Volume over period}} $$
#

# %%
import pandas as pd
import numpy as np

def calculate_relative_volume(data, window=20):
    """
    Calculate Relative Volume for each stock for VOLUME_1 to VOLUME_5.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'VOLUME_i' columns (i from 1 to 5).
    window: int, the lookback period for average volume calculation.

    Returns:
    pd.DataFrame with Relative Volume values for each stock.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in range(1, 6):
        volume_col = f'VOLUME_{i}'
        avg_volume_col = f'Average_Volume_{i}'
        rel_volume_col = f'Relative_Volume_{i}'

        new_features.extend([avg_volume_col, rel_volume_col])
        
        # Calculate average volume over the specified period
        data[avg_volume_col] = data.groupby('STOCK')[volume_col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        
        # Calculate relative volume
        data[rel_volume_col] = data[volume_col] / data[avg_volume_col]
    
    return data, new_features




# %%
train, f = calculate_relative_volume(train)
test, _ = calculate_relative_volume(test)
new_features.extend(f)

# %%
train[f].isna().sum()


# %%
train[f]

# %% [markdown]
# **5. On-Balance Volume (OBV)**
#
# On-Balance Volume (OBV) is a momentum indicator that uses volume flow to predict changes in stock price.
#
# **Formula:**
# $$ \text{OBV} = \text{Previous OBV} + \begin{cases}
# \text{Volume} & \text{if close > previous close} \\
#
# -\text{Volume} & \text{if close < previous close} \\
# 0 & \text{if close = previous close}
# \end{cases} $$

# %%
import pandas as pd
import numpy as np

def calculate_obv(data):
    """
    Calculate OBV for each stock for RET_1 to RET_5 and VOLUME_1 to VOLUME_5.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', 'RET_i' (i from 1 to 5) and 'VOLUME_i' columns.
    
    Returns:
    pd.DataFrame with OBV values for each stock.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in [1, 3, 5, 14]:
        ret_col = f'RET_{i}'
        volume_col = f'VOLUME_{i}'
        obv_col = f'OBV_{i}'

        new_features.append(obv_col)
        
        # Calculate OBV
        data[obv_col] = data.groupby('STOCK').apply(
            lambda x: ((x[ret_col].diff() > 0) * x[volume_col] - (x[ret_col].diff() < 0) * x[volume_col]).cumsum()
        ).reset_index(level=0, drop=True)
    
    return data, new_features




# %%
train, f = calculate_obv(train)
test, _ = calculate_obv(test)
new_features.extend(f)

# %%
test

# %%
train[f].isna().sum()


# %%
train[f]

# %% [markdown]
# ### Plotting features

# %% [markdown]
# **MACD for Cumulative Returns**

# %%
import pandas as pd
import numpy as np

def calculate_macd_cum_ret(data, short_window=12, long_window=26, signal_window=9):
    """
    Calculate MACD for cumulative returns for CUM_RET_1 to CUM_RET_20.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'CUM_RET_i' columns (i from 1 to 20).
    short_window: int, the short window period for EMA calculation.
    long_window: int, the long window period for EMA calculation.
    signal_window: int, the signal line period for EMA calculation.

    Returns:
    pd.DataFrame with MACD, Signal Line, and MACD Divergence for cumulative returns.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in [1, 5, 14 ]:
        cum_ret_col = f'CUM_RET_{i}'
        ema_12_col = f'EMA_12_CUM_{i}'
        ema_26_col = f'EMA_26_CUM_{i}'
        macd_col = f'MACD_CUM_{i}'
        signal_col = f'Signal_Line_CUM_{i}'
        divergence_col = f'MACD_Divergence_CUM_{i}'
        
        # new_features.extend([ema_12_col, ema_26_col, macd_col, signal_col, divergence_col])
        new_features.extend([macd_col, signal_col, divergence_col])

        # Calculate short-term EMA for cumulative returns
        data[ema_12_col] = data.groupby('STOCK')[cum_ret_col].transform(lambda x: x.ewm(span=short_window, adjust=False).mean())
        
        # Calculate long-term EMA for cumulative returns
        data[ema_26_col] = data.groupby('STOCK')[cum_ret_col].transform(lambda x: x.ewm(span=long_window, adjust=False).mean())
        
        # Calculate MACD for cumulative returns
        data[macd_col] = data[ema_12_col] - data[ema_26_col]
        
        # Calculate Signal Line for cumulative returns
        data[signal_col] = data.groupby('STOCK')[macd_col].transform(lambda x: x.ewm(span=signal_window, adjust=False).mean())
        
        # Calculate MACD Divergence for cumulative returns
        data[divergence_col] = data[macd_col] - data[signal_col]

    return data, new_features

# Example usage:
# df_with_macd_cum, new_features_macd_cum = calculate_macd_cum_ret(df)



# %%
train, f = calculate_macd_cum_ret(train)
test, _ = calculate_macd_cum_ret(test)


# %%
new_features.extend(f)

# %%
train[f].isna().sum()


# %%
train[f]

# %% [markdown]
# ### Plot

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_stock_plot(data, stock_id, cum_ret_to_plot):
    # Filter the data for the specific stock
    stock_data = data[data['STOCK'] == stock_id]

    # Create the figure with subplots
    fig = make_subplots(rows=10, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.4, 0.10, 0.10, 0.10, 0.10, 0.20, 0.20, 0.20, 0.20, 0.20])

    # Plotting Cumulative Returns
    for i in cum_ret_to_plot:  # Plot specified CUM_RET
        fig.add_trace(go.Scatter(x=stock_data['DATE'],
                                 y=stock_data[f'CUM_RET_{i}'],
                                 mode='lines',
                                 name=f'CUM_RET_{i}',
                                 line=dict(color=f'rgba({i * 50}, {i * 20}, {255 - i * 40}, 1)')),
                      row=1, col=1)

    # Adding Bollinger Bands for Cumulative Returns
    for i in cum_ret_to_plot:
        fig.add_trace(go.Scatter(x=stock_data['DATE'],
                                 y=stock_data[f'Upper_Band_CUM_{i}'],
                                 mode='lines',
                                 line=dict(color='#00BFFF'),
                                 name=f'Upper Band CUM_RET {i}'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data['DATE'],
                                 y=stock_data[f'Lower_Band_CUM_{i}'],
                                 mode='lines',
                                 line=dict(color='#00BFFF'),
                                 name=f'Lower Band CUM_RET {i}'),
                      row=1, col=1)

    # Adding RSI
    fig.add_trace(go.Scatter(x=stock_data['DATE'],
                             y=stock_data['RSI'],
                             mode='lines',
                             line=dict(color='#CBC3E3'),
                             name='RSI'), row=2, col=1)

    # Adding RSI per Sector
    fig.add_trace(go.Scatter(x=stock_data['DATE'],
                             y=stock_data['RSI_SECTOR'],
                             mode='lines',
                             line=dict(color='#A52A2A'),
                             name='RSI (Sector, Date)'), row=3, col=1)

    # Adding marking lines at 70 and 30 levels for RSI
    fig.add_shape(type="line", x0=stock_data['DATE'].min(), y0=70, x1=stock_data['DATE'].max(), y1=70,
                  line=dict(color="red", width=2, dash="dot"), row=2, col=1)
    fig.add_shape(type="line", x0=stock_data['DATE'].min(), y0=30, x1=stock_data['DATE'].max(), y1=30,
                  line=dict(color="#90EE90", width=2, dash="dot"), row=2, col=1)

    # Adding ATR
    for i in cum_ret_to_plot:  # Plot specified ATR
        fig.add_trace(go.Scatter(x=stock_data['DATE'],
                                 y=stock_data[f'ATR_{i}'],
                                 mode='lines',
                                 line=dict(color=f'rgba({i * 50}, {i * 20}, {255 - i * 40}, 1)'),
                                 name=f'ATR_{i}'), row=4, col=1)

    # Adding Volume Deviation
    for i in cum_ret_to_plot:
        fig.add_trace(go.Scatter(x=stock_data['DATE'],
                                 y=stock_data[f'DEVIATION_RVOL_{i}'],
                                 mode='lines',
                                 line=dict(color='orange'),
                                 name=f'DEVIATION_RVOL_{i}'), row=5, col=1)

    # Adding Volume Spike as scatter plot
    for i in cum_ret_to_plot:
        fig.add_trace(go.Scatter(x=stock_data['DATE'],
                                 y=stock_data[f'SPIKE_{i}'],
                                 mode='markers',
                                 marker=dict(color=f'rgba({i * 50}, {i * 20}, {255 - i * 40}, 1)'),
                                 name=f'SPIKE_{i}'), row=6, col=1)

    # Adding Volume
    fig.add_trace(go.Bar(x=stock_data['DATE'],
                         y=stock_data['VOLUME_1'],  # Assuming VOLUME_1 is the relevant volume column
                         name='Volume',
                         marker=dict(color='orange', opacity=1.0)), row=7, col=1)

    # Adding MFI
    fig.add_trace(go.Scatter(x=stock_data['DATE'],
                             y=stock_data['MFI_1'],
                             mode='lines',
                             line=dict(color='#6A5ACD'),
                             name='MFI_1'), row=8, col=1)

    # Adding marking lines at 90 and 10 levels for MFI
    fig.add_shape(type="line", x0=stock_data['DATE'].min(), y0=90, x1=stock_data['DATE'].max(), y1=90,
                  line=dict(color="red", width=2, dash="dot"), row=8, col=1)
    fig.add_shape(type="line", x0=stock_data['DATE'].min(), y0=10, x1=stock_data['DATE'].max(), y1=10,
                  line=dict(color="#90EE90", width=2, dash="dot"), row=8, col=1)

    # Adding ADL
    fig.add_trace(go.Scatter(x=stock_data['DATE'],
                             y=stock_data['ADL_1'],
                             mode='lines',
                             line=dict(color='#FF4500'),
                             name='ADL_1'), row=9, col=1)

    # Adding MACD and its lines
    fig.add_trace(go.Scatter(x=stock_data['DATE'],
                             y=stock_data['MACD_1'],
                             mode='lines',
                             line=dict(color='#1E90FF'),
                             name='MACD_1'), row=10, col=1)
    fig.add_trace(go.Scatter(x=stock_data['DATE'],
                             y=stock_data['Signal_Line_1'],
                             mode='lines',
                             line=dict(color='#FFD700'),
                             name='Signal Line 1'), row=10, col=1)
    
    # Add conditional colors for MACD histogram with brighter colors
    macd_divergence_1 = stock_data['MACD_Divergence_1']
    colors = ['#32CD32' if val > 0 else '#FF6347' for val in macd_divergence_1]
    fig.add_trace(go.Bar(x=stock_data['DATE'],
                         y=macd_divergence_1,
                         name='MACD Divergence 1',
                         marker=dict(color=colors, opacity=0.5)), row=10, col=1)

    # Layout
    fig.update_layout(title=f'Stock {stock_id} Analysis',
                      yaxis=dict(title='Value'),
                      height=2000,
                      template='plotly_dark')

    # Axes and subplots
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=10, col=1)
    fig.update_yaxes(title_text='Cumulative Returns', row=1, col=1)
    fig.update_yaxes(title_text='RSI', row=2, col=1)
    fig.update_yaxes(title_text='RSI (Sector)', row=3, col=1)
    fig.update_yaxes(title_text='ATR', row=4, col=1)
    fig.update_yaxes(title_text='Volume Deviation', row=5, col=1)
    fig.update_yaxes(title_text='Volume Spike', row=6, col=1)
    fig.update_yaxes(title_text='Volume', row=7, col=1)
    fig.update_yaxes(title_text='MFI', row=8, col=1)
    fig.update_yaxes(title_text='ADL', row=9, col=1)
    fig.update_yaxes(title_text='MACD', row=10, col=1)


    
    fig.show()



# Example usage:
# Assuming 'train' is your DataFrame and 'cum_ret_to_plot' contains the cumulative return periods to plot
cum_ret_to_plot = [5]  # Example: plot CUM_RET_1

# Plot for a specific stock (Example: Stock ID 1)
create_stock_plot(train, stock_id=25, cum_ret_to_plot=cum_ret_to_plot)


# %%
new_features

# %% [markdown]
# # More Feature Engineering

# %% [markdown]
# **Grouping by SECTOR/DATE**

# %%
import pandas as pd

def calculate_conditional_aggregated_features(train, test, shifts=[1, 2, 3, 4], statistics=['mean'], gb_features_list=[['SECTOR', 'DATE']], target_features=['RET']):
    """
    Calculate conditional aggregated features for the given shifts, statistics, and group-by features.

    Args:
    train: pd.DataFrame, training data containing target features and group-by features.
    test: pd.DataFrame, testing data containing target features and group-by features.
    shifts: list, list of shifts to calculate (default: [1, 2, 3, 4]).
    statistics: list, list of statistics to calculate (default: ['mean']).
    gb_features_list: list, list of group-by feature lists (default: [['SECTOR', 'DATE']]).
    target_features: list, list of target features to calculate (default: ['RET']).

    Returns:
    pd.DataFrame for train and test with added conditional aggregated features.
    list of new feature names.
    """
    new_features = []

    for target_feature in target_features:
        for gb_features in gb_features_list:
            tmp_name = '_'.join(gb_features)
            for shift in shifts:
                for stat in statistics:
                    name = f'{target_feature}_{shift}_{tmp_name}_{stat}'
                    feat = f'{target_feature}_{shift}'
                    new_features.append(name)
                    for data in [train, test]:
                        data[name] = data.groupby(gb_features)[feat].transform(stat)
    
    return train, test, new_features

# Example usage:
# train, test, new_features_conditional = calculate_conditional_aggregated_features(train, test)



# %%
train, test, new_features_conditional = calculate_conditional_aggregated_features(train, test)


# %%
new_features.extend(new_features_conditional)

# %%
train[new_features_conditional].isna().sum()


# %%
train[new_features_conditional]

# %% [markdown]
# **Grouped Periods**

# %%
import pandas as pd

def compute_statistical_features(train, test, periods=4, statistics=['mean', 'std'], target_features=['RET', 'VOLUME']):
    """
    Compute statistical features for the given target columns over specified periods.

    Args:
    train: pd.DataFrame, training data containing target columns.
    test: pd.DataFrame, testing data containing target columns.
    periods: int, number of periods to calculate (default: 4).
    statistics: list, list of statistics to calculate (default: ['mean', 'std']).
    target_features: list, list of target columns to calculate (default: ['RET', 'VOLUME']).

    Returns:
    pd.DataFrame for train and test with added statistical features.
    list of new feature names.
    """
    new_features = []

    for target_feature in target_features:
        for stat in statistics:
            for period in range(periods):
                feature_name = f'{stat}_{target_feature}_STOCK_PERIOD_{period+1}'
                new_features.append(feature_name)
                for data in [train, test]:
                    if stat == 'mean':
                        if target_feature == 'VOLUME':
                            data[feature_name] = data[[f'{target_feature}_{period*5 + day}' for day in range(1,6)]].mean(axis=1).abs()
                        else:
                            data[feature_name] = data[[f'{target_feature}_{period*5 + day}' for day in range(1,6)]].mean(axis=1)
                    elif stat == 'std':
                        data[feature_name] = data[[f'{target_feature}_{period*5 + day}' for day in range(1,6)]].std(axis=1)
                    elif stat == 'min':
                        data[feature_name] = data[[f'{target_feature}_{period*5 + day}' for day in range(1,6)]].min(axis=1)
                    elif stat == 'max':
                        data[feature_name] = data[[f'{target_feature}_{period*5 + day}' for day in range(1,6)]].max(axis=1)
                    elif stat == 'median':
                        data[feature_name] = data[[f'{target_feature}_{period*5 + day}' for day in range(1,6)]].median(axis=1)
    
    return train, test, new_features




# %%
import pandas as pd
import numpy as np

def compute_group_ratios(train, test, shifts=[1, 2, 3, 4], statistics=['sum'], grouping_features=[['SECTOR', 'DATE']], target_columns=None):
    """
    Compute group ratios of shifted target column sums to total sums for each group of specified features.

    Args:
    train: pd.DataFrame, training data containing target columns and grouping features.
    test: pd.DataFrame, testing data containing target columns and grouping features.
    shifts: list, list of shifts to calculate (default: [1, 2, 3, 4]).
    statistics: list, list of statistics to calculate (default: ['sum']).
    grouping_features: list, list of group-by feature lists (default: [['SECTOR', 'DATE']]).
    target_columns: list, list of target columns to calculate (default: None).

    Returns:
    pd.DataFrame for train and test with added group ratio features.
    list of new feature names.
    """
    generated_features = []

    if target_columns is None:
        target_columns = [col for col in train.columns if any(stat in col for stat in ['mean', 'std', 'min', 'max', 'median'])]

    # Create shifted columns
    for target_column in target_columns:
        for shift in shifts:
            shifted_column = f'{target_column}_SHIFT_{shift}'
            for df in [train, test]:
                df[shifted_column] = df.groupby('STOCK')[target_column].shift(shift)

    # Compute group ratios
    for target_column in target_columns:
        for group in grouping_features:
            group_name = '_'.join(group)
            for shift in shifts:
                for stat in statistics:
                    feature_name = f'{target_column}_SHIFT_{shift}_to_total_{target_column}_of_{group_name}'
                    shifted_column = f'{target_column}_SHIFT_{shift}'
                    generated_features.append(feature_name)
                    for df in [train, test]:
                        df[feature_name] = df[shifted_column] / df.groupby(group)[shifted_column].transform('sum')
                        # Fill NaNs with the median of the group
                        df[feature_name] = df.groupby('SECTOR')[feature_name].transform(lambda x: x.fillna(x.median()))
    
    return train, test, generated_features




# %%

# Example usage:
# Step 1: Compute statistical features
train, test, new_features_stat = compute_statistical_features(train, test)



# %%
new_features.extend(new_features_stat)

# %%
train[new_features_stat].isna().sum()


# %%
train[new_features_stat]


# %%
# Step 2: Compute group ratios based on the statistical features
# Example usage with your dataset
shifts = [1,2,3,4] 
statistics = ['sum']
grouping_features = [['SECTOR', 'DATE']]
target_columns = ['mean_VOLUME_STOCK_PERIOD_1']

train, test, f = compute_group_ratios(train, test, shifts, statistics, grouping_features, target_columns)

# %%
new_features.extend(f)

# %%
train[f].isna().sum()


# %%
train[f]

# %% [markdown]
# ## Volatility features

# %% [markdown]
# Compute the standard deviation of the averages to measure volatility (conditioned on sector). 

# %% [markdown]
# **Volatility for RET and VOLUME for SECTOR**

# %%
import pandas as pd

def compute_volatility(train, test, periods=[2], targets=['RET', 'VOLUME']):
    """
    Compute volatility (standard deviation) for specified targets over given periods.

    Args:
    train: pd.DataFrame, training data containing target columns.
    test: pd.DataFrame, testing data containing target columns.
    periods: list, list of periods in weeks to calculate volatility for (default: [2]).
    targets: list, list of target columns to calculate volatility for (default: ['RET', 'VOLUME']).

    Returns:
    pd.DataFrame for train and test with added volatility features.
    list of new feature names.
    """
    new_features = []

    for period in periods:
        window_size = 5 * period
        for target in targets:
            name = f'{window_size}_day_mean_{target}_VOLATILITY'
            new_features.append(name)
            for data in [train, test]:
                rolling_std_target = (
                    data.groupby(['SECTOR', 'DATE'])
                    [[f'{target}_{day}' for day in range(1, window_size + 1)]]
                    .mean()
                    .std(axis=1)
                    .to_frame(name)
                )
                placeholder = data.join(rolling_std_target, on=['SECTOR', 'DATE'], how='left')
                data[name] = placeholder[name]

    return train, test, new_features

# Example usage:
# train, test, new_features_volatility = compute_volatility(train, test)




# %%
train, test, new_features_volatility = compute_volatility(train, test)


# %%
new_features.extend(new_features_volatility)


# %%
train[new_features_volatility].isna().sum()


# %%
train[new_features_volatility]



# %%
new_features

# %% [markdown]
# **More Volatility Features for SECTOR**

# %% [markdown]
#
# The function `compute_advanced_volatility_features` calculates volatility features for financial time series data, focusing on capturing various aspects of volatility and its behavior over time.
#
# **Mathematical Meaning of Features:**
#
# 1. **Rolling Standard Deviation:**
#    - Measures the average amount by which returns deviate from their mean over a specified window.
#    - **Formula:** 
#      $$ \text{rolling\_std}(t) = \sqrt{\frac{1}{N}\sum_{i=t-N+1}^{t}(X_i - \bar{X})^2} $$
#    - Where $X_i$ are the returns, $\bar{X}$ is the mean of returns, and $N$ is the window size.
#
# 2. **Volatility Skewness:**
#    - Measures the asymmetry of the distribution of returns within a window.
#    - **Formula:**
#      $$ \text{vol\_skew}(t) = \frac{\frac{1}{N}\sum_{i=t-N+1}^{t}(X_i - \bar{X})^3}{\left(\frac{1}{N}\sum_{i=t-N+1}^{t}(X_i - \bar{X})^2\right)^{3/2}} $$
#    - Positive skew indicates a distribution with more extreme positive returns, while negative skew indicates more extreme negative returns.
#
# 3. **Volatility of Volatility:**
#    - Measures the variability of rolling standard deviation, capturing how volatile the volatility itself is over time.
#    - **Formula:**
#      $$ \text{vol\_of\_vol}(t) = \sqrt{\frac{1}{N}\sum_{i=t-N+1}^{t}(\text{rolling\_std}_i - \overline{\text{rolling\_std}})^2} $$
#    - Where $\text{rolling\_std}_i $ is the rolling standard deviation and $ \overline{\text{rolling\_std}}$  is its mean over the window.
#
# These features provide deeper insights into the stock's behavior, helping to understand not just the level of volatility but also its pattern and potential predictability.
#

# %%
import pandas as pd
import numpy as np
from scipy.stats import skew

def compute_advanced_volatility_features(train, test, targets=['RET_1', 'RET_10', 'VOLUME_1', 'VOLUME_10'], min_window_size=1, fill_method='median'):
    """
    Compute advanced volatility features.

    Args:
    train: pd.DataFrame, training data containing target columns.
    test: pd.DataFrame, testing data containing target columns.
    targets: list, list of target columns to calculate volatility for (default: ['RET_1', 'RET_10', 'RET_20']).
    periods: list, list of periods in weeks to calculate volatility for (default: [2]).
    min_window_size: int, minimum window size for rolling calculations to avoid NaNs.
    fill_method: str, method to fill NaN values ('zero' or 'median').

    Returns:
    pd.DataFrame for train and test with added volatility features.
    list of new feature names.
    """
    new_features = []

    def calculate_features(data, target, window_size):
        rolling_std = data[target].rolling(window=window_size, min_periods=min_window_size).std()
        vol_skew = data[target].rolling(window=window_size, min_periods=min_window_size).apply(lambda x: skew(x), raw=True)
        vol_of_vol = rolling_std.rolling(window=window_size, min_periods=min_window_size).std()
        return rolling_std, vol_skew, vol_of_vol

    def fill_missing_values(data, columns, fill_method):
        if fill_method == 'median':
            for col in columns:
                data[col] = data.groupby('SECTOR')[col].transform(lambda x: x.fillna(x.median()))
        elif fill_method == 'zero':
            data[columns] = data[columns].fillna(0)
        return data

    window_size = 10
    for target in targets:
        rolling_std_name = f'{window_size}_day_rolling_std_{target}'
        vol_skew_name = f'{window_size}_day_vol_skew_{target}'
        vol_of_vol_name = f'{window_size}_day_vol_of_vol_{target}'

        new_features.extend([rolling_std_name, vol_skew_name, vol_of_vol_name])

        for data in [train, test]:
            grouped = data.groupby(['SECTOR', 'DATE'])
            rolling_std, vol_skew, vol_of_vol = zip(*grouped.apply(lambda x: calculate_features(x, target, window_size)))
            
            data[rolling_std_name] = np.concatenate(rolling_std)
            data[vol_skew_name] = np.concatenate(vol_skew)
            data[vol_of_vol_name] = np.concatenate(vol_of_vol)

            # Fill missing values
            data = fill_missing_values(data, [rolling_std_name, vol_skew_name, vol_of_vol_name], fill_method)

    return train, test, new_features

# Example usage with your dataset:
# Assuming train and test are your actual datasets

# Apply the function
train, test, new_features_advanced = compute_advanced_volatility_features(train, test, targets=['RET_1', 'RET_10', 'VOLUME_1', 'VOLUME_10'], fill_method='median')




# %%
import pandas as pd
import numpy as np
from scipy.stats import skew

def compute_advanced_volatility_features(train, test, targets=['RET_1', 'RET_10', 'VOLUME_1', 'VOLUME_10'], min_window_size=1, fill_method='median'):
    """
    Compute advanced volatility features.

    Args:
    train: pd.DataFrame, training data containing target columns.
    test: pd.DataFrame, testing data containing target columns.
    targets: list, list of target columns to calculate volatility for (default: ['RET_1', 'RET_10', 'RET_20']).
    periods: list, list of periods in weeks to calculate volatility for (default: [2]).
    min_window_size: int, minimum window size for rolling calculations to avoid NaNs.
    fill_method: str, method to fill NaN values ('zero' or 'median').

    Returns:
    pd.DataFrame for train and test with added volatility features.
    list of new feature names.
    """
    new_features = []

    def calculate_features(data, target, window_size):
        rolling_std = data[target].rolling(window=window_size, min_periods=min_window_size).std()
        vol_skew = data[target].rolling(window=window_size, min_periods=min_window_size).apply(lambda x: skew(x) if len(set(x)) > 1 else np.nan, raw=True)
        vol_of_vol = rolling_std.rolling(window=window_size, min_periods=min_window_size).std()
        return rolling_std, vol_skew, vol_of_vol

    def fill_missing_values(data, columns, fill_method):
        if fill_method == 'median':
            for col in columns:
                data[col] = data.groupby('SECTOR')[col].transform(lambda x: x.fillna(x.median()))
        elif fill_method == 'zero':
            data[columns] = data[columns].fillna(0)
        return data

    window_size = 10
    for target in targets:
        rolling_std_name = f'{window_size}_day_rolling_std_{target}'
        vol_skew_name = f'{window_size}_day_vol_skew_{target}' # just skew of volume and ret, not of volatility
        vol_of_vol_name = f'{window_size}_day_vol_of_vol_{target}'

        new_features.extend([rolling_std_name, vol_skew_name, vol_of_vol_name])

        for data in [train, test]:
            grouped = data.groupby(['SECTOR', 'DATE'])
            rolling_std, vol_skew, vol_of_vol = zip(*grouped.apply(lambda x: calculate_features(x, target, window_size)))
            
            data[rolling_std_name] = np.concatenate(rolling_std)
            data[vol_skew_name] = np.concatenate(vol_skew)
            data[vol_of_vol_name] = np.concatenate(vol_of_vol)

            # Fill missing values
            data = fill_missing_values(data, [rolling_std_name, vol_skew_name, vol_of_vol_name], fill_method)

    return train, test, new_features

# Example usage with your dataset:
# Assuming train and test are your actual datasets

# Apply the function
train, test, new_features_advanced = compute_advanced_volatility_features(train, test, targets=['RET_1', 'RET_10', 'VOLUME_1', 'VOLUME_10'], fill_method='median')


# %%
new_features.extend(new_features_advanced)


# %%
train[new_features_advanced].isna().sum()


# %%
train[new_features_advanced]


# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_dashboard(data, stock_id, ret_col='RET_1', vol_col='VOLUME_1'):
    # Filter the data for the specific stock
    stock_data = data[data['STOCK'] == stock_id]

    # Create the figure with subplots for non-distribution plots
    fig = make_subplots(rows=4, cols=2, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(
        'Daily Returns', 'Volume', 
        'Volatility of Returns', 'Volatility of Volume',
        'Skewness of Returns', 'Skewness of Volume',
        'Volatility of Volatility of Returns', 'Volatility of Volatility of Volume'
    ))

    # Daily Returns
    fig.add_trace(go.Scatter(x=stock_data['DATE'], y=stock_data[ret_col], mode='lines', name=ret_col), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data['DATE'], y=stock_data[vol_col], mode='lines', name=vol_col), row=1, col=2)

    # Volatility (Rolling Std)
    fig.add_trace(go.Scatter(x=stock_data['DATE'], y=stock_data[f'10_day_rolling_std_{ret_col}'], mode='lines', name=f'Volatility {ret_col}'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock_data['DATE'], y=stock_data[f'10_day_rolling_std_{vol_col}'], mode='lines', name=f'Volatility {vol_col}'), row=2, col=2)

    # Skewness
    fig.add_trace(go.Scatter(x=stock_data['DATE'], y=stock_data[f'10_day_vol_skew_{ret_col}'], mode='lines', name=f'Skewness {ret_col}'), row=3, col=1)
    fig.add_trace(go.Scatter(x=stock_data['DATE'], y=stock_data[f'10_day_vol_skew_{vol_col}'], mode='lines', name=f'Skewness {vol_col}'), row=3, col=2)

    # Volatility of Volatility
    fig.add_trace(go.Scatter(x=stock_data['DATE'], y=stock_data[f'10_day_vol_of_vol_{ret_col}'], mode='lines', name=f'Vol of Vol {ret_col}'), row=4, col=1)
    fig.add_trace(go.Scatter(x=stock_data['DATE'], y=stock_data[f'10_day_vol_of_vol_{vol_col}'], mode='lines', name=f'Vol of Vol {vol_col}'), row=4, col=2)

    # Layout
    fig.update_layout(title=f'Stock {stock_id} Analysis', height=1200, template='plotly_dark')
    
    # Show x-axis in all plots
    for row in range(1, 5):
        fig.update_xaxes(showgrid=True, row=row, col=1)
        fig.update_xaxes(showgrid=True, row=row, col=2)
    
    fig.show()

    # Create a separate figure for the distributions
    fig_dist = make_subplots(rows=1, cols=2, subplot_titles=('Distribution of Returns', 'Distribution of Volume'))

    # Distributions
    fig_dist.add_trace(go.Histogram(x=stock_data[ret_col], name=f'Distribution {ret_col}', nbinsx=50), row=1, col=1)
    fig_dist.add_trace(go.Histogram(x=stock_data[vol_col], name=f'Distribution {vol_col}', nbinsx=50), row=1, col=2)

    # Set x-axis limits for distributions to zoom in
    fig_dist.update_xaxes(range=[stock_data[ret_col].quantile(0.01), stock_data[ret_col].quantile(0.99)], row=1, col=1)
    fig_dist.update_xaxes(range=[stock_data[vol_col].quantile(0.01), stock_data[vol_col].quantile(0.99)], row=1, col=2)

    # Layout
    fig_dist.update_layout(title=f'Stock {stock_id} Distributions', height=600, template='plotly_dark')
    
    fig_dist.show()

# Example usage:
# Assuming 'train' is your DataFrame and you want to plot for stock ID 1
create_dashboard(train, stock_id=25, ret_col='RET_1', vol_col='VOLUME_1')


# %%
train.to_csv('train_final.csv', index=True)
test.to_csv('test_final.csv', index=True)

# %% [markdown]
# # Feature Selection

# %%
target = 'RET'

n_shifts_ret = 5  # If you don't want all the shifts to reduce noise
n_shifts_vol = 5
features = ['RET_%d' % (i + 1) for i in range(n_shifts_ret)]
features += ['VOLUME_%d' % (i + 1) for i in range(n_shifts_vol)]
features += new_features  # The conditional features
train[features].head()

# %%
train[features].isna().sum().sum()


# %%
def fillna_with_sector_median(train, features):
    """
    Fill NaN values in the specified features with the median value of the SECTOR.

    Args:
    train: pd.DataFrame, training data containing features.
    features: list, list of feature columns to fill NaNs for.

    Returns:
    pd.DataFrame with NaNs filled.
    """
    for feature in features:
        train[feature] = train.groupby('SECTOR')[feature].transform(lambda x: x.fillna(x.median()))
    
    for feature in features:
        train[feature] = train[feature].transform(lambda x: x.fillna(x.median()))
    return train

# Example usage with your train dataset
train = fillna_with_sector_median(train, features)


# %%
train[features].isna().sum().sum()

# %%
corr_features = features + ['RET']
fig = plt.figure(figsize=(100,100))
plt.matshow(train[corr_features].corr(), fignum=fig.number)
plt.xticks(range(train[corr_features].shape[1]), train[corr_features].columns, rotation=90, fontsize=14)
plt.yticks(range(train[corr_features].shape[1]), train[corr_features].columns, fontsize=14)
plt.colorbar()
plt.show()

# %% [markdown]
# # Model Selection and Local Score

# %% [markdown]
# Let's first remove infinite values.

# %%
import numpy as np

# Check for infinity values in the train dataset
inf_counts_train = np.isinf(train).sum()
inf_columns_train = inf_counts_train[inf_counts_train > 0]
inf_columns_train_list = inf_columns_train.index.tolist()
print("Columns with infinity values in the train dataset:")
print(inf_columns_train_list)

# Check for infinity values in the test dataset
inf_counts_test = np.isinf(test).sum()
inf_columns_test = inf_counts_test[inf_counts_test > 0]
inf_columns_test_list = inf_columns_test.index.tolist()
print("Columns with infinity values in the test dataset:")
print(inf_columns_test_list)

# Total infinity values in train and test datasets
total_inf_train = inf_columns_train.sum()
total_inf_test = inf_columns_test.sum()
print(f"Total infinity values in the train dataset: {total_inf_train}")
print(f"Total infinity values in the test dataset: {total_inf_test}")

# Combine the lists for further use if needed
inf_columns_combined_list = list(set(inf_columns_train_list + inf_columns_test_list))
print("Combined list of columns with infinity values:")
print(inf_columns_combined_list)

# Assuming features is your list of feature column names
features = [col for col in features if col not in inf_columns_combined_list]




# %%
# Assuming features is your list of feature column names
# features = [col for col in features if col not in inf_columns_combined_list]


# %%
features

# %%
X_train = train[features]
y_train = train[target]

# Choose parameters of the LGBM RF such that they coincide with the RandomForestClassifier 
parameters = {
    'boosting_type': 'rf',
    'n_estimators': 500,
    'max_depth': 2**3,
    'random_state': 0,
    'n_jobs': -1, 
    'feature_fraction': np.log(X_train.shape[0])/X_train.shape[0],
    'objective': 'binary',
    'verbose': -1
}

rf_params = {
    'n_estimators': 500,
    'max_depth': 2**3,
    'random_state': 0,
    'n_jobs': -1
}

train_dates = train['DATE'].unique()
test_dates = test['DATE'].unique()

n_splits = 4
scores = []
models = []

splits = KFold(n_splits=n_splits, random_state=0,
               shuffle=True).split(train_dates) # Generates the splits of the indexes to use as train / test

for i, (local_train_dates_ids, local_test_dates_ids) in enumerate(splits):
    local_train_dates = train_dates[local_train_dates_ids]
    local_test_dates = train_dates[local_test_dates_ids]

    local_train_ids = train['DATE'].isin(local_train_dates)
    local_test_ids = train['DATE'].isin(local_test_dates)

    X_local_train = X_train.loc[local_train_ids]
    y_local_train = y_train.loc[local_train_ids]
    X_local_test = X_train.loc[local_test_ids]
    y_local_test = y_train.loc[local_test_ids]

    #model = LGBMClassifier(**parameters)
    model = RandomForestClassifier(**rf_params)
    model.fit(X_local_train, y_local_train.values.reshape(-1))

    y_local_pred = model.predict_proba(X_local_test)[:, 1]
    
    sub = train.loc[local_test_ids].copy()
    sub['pred'] = y_local_pred
    y_local_pred = sub.groupby('DATE')['pred'].transform(lambda x: x > x.median()).values

    models.append(model)
    score = accuracy_score(y_local_test, y_local_pred)
    scores.append(score)
    print(f"Fold {i+1} - Accuracy: {score* 100:.2f}%")

mean = np.mean(scores)*100
std = np.std(scores)*100
u = (mean + std)
l = (mean - std)
print(f'Accuracy: {mean:.2f}% [{l:.2f} ; {u:.2f}] (+- {std:.2f})')

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'model' is your trained model and 'features' is the list of feature names

# Combine feature importances from multiple models if you have an ensemble
feature_importances = pd.DataFrame([model.feature_importances_ for model in models], columns=features)

# Calculate mean feature importance and select top 50
mean_importances = feature_importances.mean(axis=0).sort_values(ascending=False).head(50)

# Plotting
plt.figure(figsize=(15, 12))
sns.barplot(x=mean_importances, y=mean_importances.index, orient='h', order=mean_importances.index)
plt.title('Top 50 Feature Importances')
plt.show()


# %% [markdown]
# # Prediction

# %%
test__ = pd.read_csv('x_test.csv', index_col='ID')

# %%
rf_params = {
    'n_estimators': 500,
    'max_depth': 2**3,
    'random_state': 0,
    'n_jobs': -1
}

target = 'RET'
y_train = train[target]
X_train = train[features]
X_test = test[features]

model = RandomForestClassifier(**rf_params)
model.fit(X_train, y_train)

# Let's compare what to use as threshold.

sub = train.copy()
sub['pred'] = model.predict_proba(X_train)[:,1]
y_pred_train = sub.groupby('DATE')['pred'].transform(
    lambda x: x > x.median()).values
print('MEDIAN', accuracy_score(y_pred_train,y_train))

sub = train.copy()
sub['pred'] = model.predict_proba(X_train)[:,1]
y_pred_train = sub.groupby('DATE')['pred'].transform(
    lambda x: x > x.mean()).values
print('MEAN', accuracy_score(y_pred_train,y_train))

print('LIBRARY BUILT IN THRESHOLD 0.5', accuracy_score(model.predict(X_train),y_train))

y_pred = model.predict_proba(X_test)[:, 1]

sub = test.copy()
sub['pred'] = y_pred
y_pred = sub.groupby('DATE')['pred'].transform(
    lambda x: x > x.median()).values

submission = pd.Series(y_pred)
submission.index = test__.index
submission.name = target

submission.to_csv('./y_test.csv', index=True, header=True)

# %%
submission

# %% [markdown]
# # Submission to Leaderboard

# %% [markdown]
# This submission placed me in the 70th of 399 submissions.

# %%
rf_params = {
    'n_estimators': 500,
    'max_depth': 2**3,
    'random_state': 0,
    'n_jobs': -1
}

target = 'RET'
y_train = train[target]
X_train = train[features]
X_test = test[features]

model = RandomForestClassifier(**rf_params)
model.fit(X_train, y_train)

sub = train.copy()
sub['pred'] = model.predict_proba(X_train)[:,1]
y_pred_train = sub.groupby('DATE')['pred'].transform(
    lambda x: x > x.median()).values
print(accuracy_score(y_pred_train,y_train))

y_pred = model.predict_proba(X_test)[:, 1]

sub = test.copy()
sub['pred'] = y_pred
y_pred = sub.groupby('DATE')['pred'].transform(
    lambda x: x > x.median()).values

submission = pd.Series(y_pred)
submission.index = test__.index
submission.name = target

submission.to_csv('./submission.csv', index=True, header=True)

# %%
submission

# %% [markdown]
# Corresponding feature importances:

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Combine feature importances from multiple models if you have an ensemble
feature_importances = pd.DataFrame([model.feature_importances_ for model in models], columns=features)

# Calculate mean feature importance and select top 50
mean_importances = feature_importances.mean(axis=0).sort_values(ascending=False).head(50)

# Plotting
plt.figure(figsize=(15, 12))
sns.barplot(x=mean_importances, y=mean_importances.index, orient='h', order=mean_importances.index)
plt.title('Top 50 Feature Importances')
plt.show()


# %% [markdown]
# # Appendix

# %% [markdown]
# The following didn't help the prediction accuracy.

# %% [markdown]
# # PCA - Factors

# %% [markdown]
# ### Explanation of PCA Application on Dataset
#
# This script applies Principal Component Analysis (PCA) to the `train` and `test` datasets, specifically targeting the numerical columns (`RET` and `VOLUME` for the past 20 days). PCA is applied separately for each sector, both with and without whitening, to generate new features that capture the underlying variance in the data.
#
# #### Principal Component Analysis (PCA)
#
# **What is PCA?**
#
# PCA is a dimensionality reduction technique that transforms a large set of variables into a smaller one that still contains most of the information in the large set. This is achieved by identifying the directions (principal components) along which the variation in the data is maximal.
#
# **Intuitive Explanation:**
#
# - **Principal Components (PCs)**: These are new axes that represent directions of maximum variance in the data. The first principal component accounts for the most variance, the second for the next most, and so on.
# - **Dimensionality Reduction**: By transforming data into a smaller number of principal components, PCA reduces complexity and noise while preserving the essential patterns in the data.
#
# **Mathematical Explanation:**
#
# 1. **Standardization**:
#    - **Purpose**: To standardize the data to have a mean of 0 and a standard deviation of 1.
#    - **Formula**: 
#      $$ z = \frac{(x - \mu)}{\sigma} $$
#    - Where \( \mu \) is the mean and \( \sigma \) is the standard deviation.
#
# 2. **Covariance Matrix**:
#    - **Purpose**: To understand how variables in the data set vary together.
#    - **Formula**:
#      $$ \Sigma = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \mu)(X_i - \mu)^T $$
#    - Where \( X_i \) is the vector of the \( i \)-th observation.
#
# 3. **Eigen Decomposition**:
#    - **Purpose**: To find the principal components, which are the eigenvectors of the covariance matrix.
#    - **Formula**:
#      $$ \Sigma \mathbf{v} = \lambda \mathbf{v} $$
#    - Where \( \mathbf{v} \) are the eigenvectors (principal components) and \( \lambda \) are the eigenvalues (variance explained by each principal component).
#
# 4. **Projection onto Principal Components**:
#    - **Purpose**: To transform the data onto the new axes defined by the principal components.
#    - **Formula**:
#      $$ Z = X \cdot W $$
#    - Where \( X \) is the standardized data matrix and \( W \) is the matrix of eigenvectors.
#
# 5. **Whitening**:
#    - **Purpose**: To transform the principal components to have unit variance.
#    - **Formula**:
#      $$ Z_{\text{white}} = Z \cdot D^{-1/2} $$
#    - Where \( D \) is the diagonal matrix of eigenvalues.
#
# **Why Use PCA?**
#
# - **Noise Reduction**: By focusing on the main components that explain the most variance, PCA helps reduce noise in the data.
# - **Feature Extraction**: PCA generates new features (principal components) that may be more effective for predictive modeling.
# - **Data Compression**: Reduces the dimensionality of the data, making it easier and faster to process.
#
# **Why Use PCA Here?**
#
# In this context, PCA is used to:
# - Extract meaningful patterns from the historical return (`RET`) and volume (`VOLUME`) data for stocks.
# - Reduce the complexity of the dataset by summarizing the information from 40 numerical columns into fewer principal components.
# - Enhance the dataset with new features that capture the underlying variance, potentially improving the performance of machine learning models.
#
#
#

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

test_ = test.copy(deep=True)
train_ = train.copy(deep=True)

# Function to apply PCA to each group with a given whiten parameter and specified number of components
def apply_pca_to_group(group, numerical_cols, n_components=20, whiten=False):
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(group[numerical_cols])
        pca = PCA(n_components=n_components, whiten=whiten)
        principal_components = pca.fit_transform(scaled_data)
        # Create a DataFrame with the principal components
        suffix = 'W' if whiten else 'NW'
        pc_df = pd.DataFrame(data=principal_components, columns=[f'PC_{suffix}_{i+1}' for i in range(principal_components.shape[1])], index=group.index)
        return pc_df, pc_df.columns.tolist()
    except Exception as e:
        print(f"Error in group {group.name}: {e}")
        return pd.DataFrame(index=group.index), []

# Function to process the dataset
def process_dataset(df, numerical_cols, n_components=20):
    # Perform PCA without whitening
    df_pca_list = df.groupby('SECTOR').apply(lambda group: apply_pca_to_group(group, numerical_cols, n_components=n_components, whiten=False))
    df_pca = pd.concat([x[0] for x in df_pca_list], axis=0)
    new_variables = list({var for x in df_pca_list for var in x[1]})

    # Perform PCA with whitening
    df_pca_list_whiten = df.groupby('SECTOR').apply(lambda group: apply_pca_to_group(group, numerical_cols, n_components=n_components, whiten=True))
    df_pca_whiten = pd.concat([x[0] for x in df_pca_list_whiten], axis=0)
    new_variables_whiten = list({var for x in df_pca_list_whiten for var in x[1]})

    # Combine the two PCA results directly into the original DataFrame
    df_pca_combined = pd.concat([df_pca, df_pca_whiten[new_variables_whiten]], axis=1)

    return df_pca_combined, new_variables, new_variables_whiten

# Define numerical columns for PCA
numerical_cols = [f'RET_{i}' for i in range(1, 21)] + [f'VOLUME_{i}' for i in range(1, 21)]


# Process train_ dataset
train_pca_combined, train_new_variables, train_new_variables_whiten = process_dataset(train_, numerical_cols, n_components=20)
print("List of new variables for train (whiten=False):", train_new_variables)
print("List of new variables for train (whiten=True):", train_new_variables_whiten)

# Fill NaN values with the median for train_pca_combined
for col in train_new_variables:
    train_pca_combined[col].fillna(train_pca_combined[col].median(), inplace=True)

for col in train_new_variables_whiten:
    train_pca_combined[col].fillna(train_pca_combined[col].median(), inplace=True)

# Process test_ dataset
test_pca_combined, test_new_variables, test_new_variables_whiten = process_dataset(test_, numerical_cols, n_components=20)
print("List of new variables for test (whiten=False):", test_new_variables)
print("List of new variables for test (whiten=True):", test_new_variables_whiten)

# Fill NaN values with the median for test_pca_combined
for col in test_new_variables:
    test_pca_combined[col].fillna(test_pca_combined[col].median(), inplace=True)

for col in test_new_variables_whiten:
    test_pca_combined[col].fillna(test_pca_combined[col].median(), inplace=True)

# Ensure original columns do not get duplicated by merging only new PC columns
train_pca_combined = train_pca_combined[train_new_variables + train_new_variables_whiten]
test_pca_combined = test_pca_combined[test_new_variables + test_new_variables_whiten]

# Merge the PCA results back to the original train and test datasets
train = train.join(train_pca_combined, how='left')
test = test.join(test_pca_combined, how='left')

# Fill any remaining NaN values in the merged datasets
for col in train_new_variables + train_new_variables_whiten:
    train[col].fillna(train[col].median(), inplace=True)

for col in test_new_variables + test_new_variables_whiten:
    test[col].fillna(test[col].median(), inplace=True)



# %%
train[train_new_variables]


# %%
train[train_new_variables_whiten].isna().sum()

# %%
new_features.extend(train_new_variables_whiten)


# %%
new_features

# %% [markdown]
# ## Further look into factors

# %%
import numpy as np
import matplotlib.pyplot as plt

def optimal_shrinkage(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    n = len(eigenvalues)
    m = n - 5  # example choice, adjust as needed
    shrinkage_target = np.mean(eigenvalues[:m])
    shrunken_eigenvalues = np.maximum(eigenvalues, shrinkage_target)
    shrunken_cov_matrix = eigenvectors @ np.diag(shrunken_eigenvalues) @ eigenvectors.T
    return shrunken_cov_matrix, eigenvalues, shrunken_eigenvalues

# Example usage with sample data
np.random.seed(0)
sample_data = train_[numerical_cols]  # 100 samples, 20 features
cov_matrix = np.cov(sample_data, rowvar=False)
shrunken_cov_matrix, eigenvalues, shrunken_eigenvalues = optimal_shrinkage(cov_matrix)

# Plot original vs. shrunken eigenvalues
plt.plot(eigenvalues, label='Original Eigenvalues')
plt.plot(shrunken_eigenvalues, label='Shrunken Eigenvalues', linestyle='--')
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.legend()
plt.title('Original vs. Shrunken Eigenvalues')
plt.show()


# %%
def choose_number_of_factors(eigenvalues):
    plt.plot(range(1, len(eigenvalues) + 1), sorted(eigenvalues, reverse=True), marker='o')
    plt.xlabel('Number of Factors')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot')
    plt.show()

# Example usage with sample data
choose_number_of_factors(eigenvalues)


# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def whiten_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pca = PCA(whiten=True)
    whitened_data = pca.fit_transform(scaled_data)
    return whitened_data, pca.explained_variance_

# Example usage with sample data
whitened_data, explained_variance = whiten_data(sample_data)

# Plot explained variance
plt.plot(explained_variance, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Explained Variance by Principal Component (Whitened)')
plt.show()


# %% [markdown]
# # Beta

# %% [markdown]
# ### Explanation of Beta Calculation and Integration into Dataset
#
# #### What is Beta?
#
# **Beta** is a measure of a stock's volatility in relation to the overall market. It quantifies the tendency of a stock's returns to respond to swings in the market. A beta value can provide insights into a stock's risk profile compared to the market.
#
# - **Beta > 1**: Indicates that the stock is more volatile than the market.
# - **Beta < 1**: Indicates that the stock is less volatile than the market.
# - **Beta = 1**: Indicates that the stock's volatility is similar to the market.
#
# #### Mathematical Explanation
#
# **Beta Calculation** involves the following steps:
#
# 1. **Covariance**: Measures how two variables (stock returns and market returns) move together.
#    - **Formula**:
#      $$ \text{Cov}(\text{Stock Returns}, \text{Market Returns}) = \frac{1}{n-1} \sum_{i=1}^{n} (R_{\text{stock}, i} - \overline{R}_{\text{stock}})(R_{\text{market}, i} - \overline{R}_{\text{market}}) $$
#    - Where \( R_{\text{stock}, i} \) and \( R_{\text{market}, i} \) are the returns of the stock and market at time \( i \), and \( \overline{R}_{\text{stock}} \) and \( \overline{R}_{\text{market}} \) are their mean returns.
#
# 2. **Variance**: Measures the dispersion of market returns.
#    - **Formula**:
#      $$ \text{Var}(\text{Market Returns}) = \frac{1}{n-1} \sum_{i=1}^{n} (R_{\text{market}, i} - \overline{R}_{\text{market}})^2 $$
#
# 3. **Beta**: The ratio of the covariance between the stock and market returns to the variance of the market returns.
#    - **Formula**:
#      $$ \beta = \frac{\text{Cov}(\text{Stock Returns}, \text{Market Returns})}{\text{Var}(\text{Market Returns})} $$
#
# #### Why Use Beta?
#
# **Risk Assessment**: Beta provides a measure of a stock's risk in relation to the market. It helps investors understand how much a stock is expected to move relative to market movements.
#
# **Portfolio Management**: By knowing the beta of individual stocks, investors can construct portfolios with desired risk profiles.
#
# #### Overview of Implementation
#
# 1. **Loading Data**:
#    - The script assumes that the `train_` and `test_` datasets are loaded from CSV files.
#
# 2. **Defining Numerical Columns**:
#    - The numerical columns for stock returns are defined as `RET_1`, `RET_5`, and `RET_10`.
#
# 3. **Calculating Market Returns**:
#    - Market returns are calculated by averaging the stock returns for each day across all stocks. This provides a benchmark for comparing individual stock performance.
#
# 4. **Function to Calculate Beta**:
#    - A function `calculate_beta` is defined to calculate beta for each stock. It computes the covariance between stock returns and market returns and divides it by the variance of market returns.
#
# 5. **Calculating Beta Values**:
#    - The script calculates beta values for `RET_1`, `RET_5`, and `RET_10` for both `train_` and `test_` datasets. This is done by grouping the data by stock and applying the `calculate_beta` function.
#
# 6. **Merging Beta Values**:
#    - The calculated beta values are merged back into the original `train` and `test` datasets based on the `STOCK` column. New variables for beta values are added to the datasets.
#
# 7. **Handling NaN Values**:
#    - Any NaN values in the new beta variables are filled with the median of the respective columns to ensure completeness of the data.
#
# #### Summary
#
# This process enhances the original dataset by adding new features that represent the beta values of the returns for different periods. Beta is a measure of volatility and risk relative to the market, providing valuable insights for financial analysis and modeling. By calculating and integrating these beta values, the dataset is enriched with features that capture the relationship between individual stocks and the overall market.
#

# %%
import pandas as pd
import numpy as np

# Load the separate datasets
# train_ = pd.read_csv("x_train.csv", index_col='ID')
# test_ = pd.read_csv('x_test.csv', index_col='ID')

# Define numerical columns for stock returns
numerical_cols = ['RET_1', 'RET_5', 'RET_10']

# Calculate market returns by averaging stock returns for each day
train_['MARKET_RET_1'] = train_[['RET_1']].mean(axis=1)
train_['MARKET_RET_5'] = train_[['RET_5']].mean(axis=1)
train_['MARKET_RET_10'] = train_[['RET_10']].mean(axis=1)

test_['MARKET_RET_1'] = test_[['RET_1']].mean(axis=1)
test_['MARKET_RET_5'] = test_[['RET_5']].mean(axis=1)
test_['MARKET_RET_10'] = test_[['RET_10']].mean(axis=1)

# Function to calculate beta
def calculate_beta(df, ret_col, market_col):
    betas = []
    for stock in df['STOCK'].unique():
        stock_df = df[df['STOCK'] == stock]
        market_returns = stock_df[market_col]
        stock_returns = stock_df[ret_col]
        cov = np.cov(stock_returns, market_returns)[0, 1]
        var = np.var(market_returns)
        beta = cov / var
        betas.append((stock, beta))
    beta_df = pd.DataFrame(betas, columns=['STOCK', f'BETA_{ret_col}'])
    return beta_df

# Calculate beta for RET_1, RET_5, and RET_10
train_beta_1 = calculate_beta(train_, 'RET_1', 'MARKET_RET_1')
train_beta_5 = calculate_beta(train_, 'RET_5', 'MARKET_RET_5')
train_beta_10 = calculate_beta(train_, 'RET_10', 'MARKET_RET_10')

test_beta_1 = calculate_beta(test_, 'RET_1', 'MARKET_RET_1')
test_beta_5 = calculate_beta(test_, 'RET_5', 'MARKET_RET_5')
test_beta_10 = calculate_beta(test_, 'RET_10', 'MARKET_RET_10')

# Initialize the new variables list
new_variables = []

# Merge beta values back to the original datasets and update new variables list
train = train.merge(train_beta_1, on='STOCK', how='left')
new_variables.append('BETA_RET_1')
train = train.merge(train_beta_5, on='STOCK', how='left')
new_variables.append('BETA_RET_5')
train = train.merge(train_beta_10, on='STOCK', how='left')
new_variables.append('BETA_RET_10')

test = test.merge(test_beta_1, on='STOCK', how='left')
test = test.merge(test_beta_5, on='STOCK', how='left')
test = test.merge(test_beta_10, on='STOCK', how='left')

new_variables.extend(['BETA_RET_1', 'BETA_RET_5', 'BETA_RET_10'])

# Fill NaN values with the median for the new beta variables in train and test datasets
for col in new_variables:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)

# Save the processed data
# train.to_csv("train_with_beta.csv", index=False)
# test.to_csv("test_with_beta.csv", index=False)

# Display the list of new variables
print("List of new variables:", new_variables)


# %% [markdown]
#

# %%
new_features.extend(new_variables)


# %%
train[new_variables].isna().sum()

# %%
test[new_variables].isna().sum()


# %%
# Define a function to print beta statistics
def print_beta_statistics(df, beta_cols):
    for col in beta_cols:
        greater_than_1 = df[df[col] > 1].shape[0]
        less_than_1 = df[df[col] < 1].shape[0]
        close_to_1 = df[(df[col] >= 0.98) & (df[col] <= 1.01)].shape[0]
        print(f"Statistics for {col}:")
        print(f"  Observations where {col} > 1: {greater_than_1}")
        print(f"  Observations where {col} < 1: {less_than_1}")
        print(f"  Observations where {col} ≈ 1: {close_to_1}\n")


# %%
# Print beta statistics for train dataset
print("Beta statistics for train dataset:")
print_beta_statistics(train, new_variables)

# Print beta statistics for test dataset
print("Beta statistics for test dataset:")
print_beta_statistics(test, new_variables)

# %% [markdown]
# # Feature Selection

# %%
target = 'RET'

n_shifts_ret = 5  # If you don't want all the shifts to reduce noise
n_shifts_vol = 5
features = ['RET_%d' % (i + 1) for i in range(n_shifts_ret)]
features += ['VOLUME_%d' % (i + 1) for i in range(n_shifts_vol)]
features += new_features  # The conditional features
train[features].head()

# %%
corr_features = features + ['RET']
fig = plt.figure(figsize=(20,20))
plt.matshow(train[corr_features].corr(), fignum=fig.number)
plt.xticks(range(train[corr_features].shape[1]), train[corr_features].columns, rotation=90, fontsize=14)
plt.yticks(range(train[corr_features].shape[1]), train[corr_features].columns, fontsize=14)
plt.colorbar()
plt.show()

# %%
import numpy as np

# Check for infinity values in the train dataset
inf_counts_train = np.isinf(train).sum()
inf_columns_train = inf_counts_train[inf_counts_train > 0]
inf_columns_train_list = inf_columns_train.index.tolist()
print("Columns with infinity values in the train dataset:")
print(inf_columns_train_list)

# Check for infinity values in the test dataset
inf_counts_test = np.isinf(test).sum()
inf_columns_test = inf_counts_test[inf_counts_test > 0]
inf_columns_test_list = inf_columns_test.index.tolist()
print("Columns with infinity values in the test dataset:")
print(inf_columns_test_list)

# Total infinity values in train and test datasets
total_inf_train = inf_columns_train.sum()
total_inf_test = inf_columns_test.sum()
print(f"Total infinity values in the train dataset: {total_inf_train}")
print(f"Total infinity values in the test dataset: {total_inf_test}")

# Combine the lists for further use if needed
inf_columns_combined_list = list(set(inf_columns_train_list + inf_columns_test_list))
print("Combined list of columns with infinity values:")
print(inf_columns_combined_list)

# Assuming features is your list of feature column names
features = [col for col in features if col not in inf_columns_combined_list]


# %%
features

# %% [markdown]
# # Model

# %%
X_train = train[features]
y_train = train[target]

# Choose parameters of the LGBM RF such that they coincide with the RandomForestClassifier 
parameters = {
    'boosting_type': 'rf',
    'n_estimators': 500,
    'max_depth': 2**3,
    'random_state': 0,
    'n_jobs': -1, 
    'feature_fraction': np.log(X_train.shape[0])/X_train.shape[0],
    'objective': 'binary',
    'verbose': -1
}

rf_params = {
    'n_estimators': 500,
    'max_depth': 2**3,
    'random_state': 0,
    'n_jobs': -1
}

train_dates = train['DATE'].unique()
test_dates = test['DATE'].unique()

n_splits = 4
scores = []
models = []

splits = KFold(n_splits=n_splits, random_state=0,
               shuffle=True).split(train_dates) # Generates the splits of the indexes to use as train / test

for i, (local_train_dates_ids, local_test_dates_ids) in enumerate(splits):
    local_train_dates = train_dates[local_train_dates_ids]
    local_test_dates = train_dates[local_test_dates_ids]

    local_train_ids = train['DATE'].isin(local_train_dates)
    local_test_ids = train['DATE'].isin(local_test_dates)

    X_local_train = X_train.loc[local_train_ids]
    y_local_train = y_train.loc[local_train_ids]
    X_local_test = X_train.loc[local_test_ids]
    y_local_test = y_train.loc[local_test_ids]

    #model = LGBMClassifier(**parameters)
    model = RandomForestClassifier(**rf_params)
    model.fit(X_local_train, y_local_train.values.reshape(-1))

    y_local_pred = model.predict_proba(X_local_test)[:, 1]
    
    sub = train.loc[local_test_ids].copy()
    sub['pred'] = y_local_pred
    y_local_pred = sub.groupby('DATE')['pred'].transform(lambda x: x > x.median()).values

    models.append(model)
    score = accuracy_score(y_local_test, y_local_pred)
    scores.append(score)
    print(f"Fold {i+1} - Accuracy: {score* 100:.2f}%")

mean = np.mean(scores)*100
std = np.std(scores)*100
u = (mean + std)
l = (mean - std)
print(f'Accuracy: {mean:.2f}% [{l:.2f} ; {u:.2f}] (+- {std:.2f})')

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'model' is your trained model and 'features' is the list of feature names

# Combine feature importances from multiple models if you have an ensemble
feature_importances = pd.DataFrame([model.feature_importances_ for model in models], columns=features)

# Calculate mean feature importance and select top 50
mean_importances = feature_importances.mean(axis=0).sort_values(ascending=False).head(60)

# Plotting
plt.figure(figsize=(15, 12))
sns.barplot(x=mean_importances, y=mean_importances.index, orient='h', order=mean_importances.index)
plt.title('Top 50 Feature Importances')
plt.show()

# %% [markdown]
# **End of Notebook**
