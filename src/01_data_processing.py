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

# %% [markdown]
# # <a id='toc1_'></a>[Stock Market Movement Prediction](#toc0_)
# ### <a id='toc1_1_1_'></a>[Using ML Techniques to Forecast Next-Day Stock Price Movements](#toc0_)
#
# This project tackles a challenging binary classification problem in quantitative finance - predicting the directional movement
# (up or down) of individual US stocks for the following trading day. The goal is to develop a machine learning model
# that can provide data-driven insights to support investment decision-making.
#
# The model leverages a feature set including:
# - **20** days of historical price returns
# - **20** days of trading volume data
# - Categorical stock metadata (industry, sector, etc.)
# - Technical indicators and statistical features
#
# A public benchmark accuracy of **51.31%** was previously achieved using a Random Forest model with **5** days of historical data and
# sector-level information. Our approach aims to improve upon this by incorporating more features and advanced modeling techniques.

# %% [markdown]
# **Table of contents**<a id='toc0_'></a>
# - [Stock Market Movement Prediction](#toc1_)
#     - [Using ML Techniques to Forecast Next-Day Stock Price Movements](#toc1_1_1_)
#   - [Data Description](#toc1_2_)
#     - [Input Dataset Structure](#toc1_2_1_)
#     - [Output Dataset Structure](#toc1_2_2_)
#     - [Key Financial Calculations](#toc1_2_3_)
#     - [Dataset Scale](#toc1_2_4_)
#     - [Submission Guidelines](#toc1_2_5_)
#   - [Implementation Strategy](#toc1_3_)
#       - [Project Roadmap](#toc1_3_1_1_)
#     - [Library Imports](#toc1_3_2_)
#     - [Environment Configuration and Setup](#toc1_3_3_)
#       - [Pipeline Node Execution Configuration](#toc1_3_3_1_)
#         - [Environment Initialization](#toc1_3_3_1_1_)
#   - [Data Loading and Initial Processing](#toc1_4_)
#     - [Preprocessing Strategy](#toc1_4_1_)
#       - [Visual Data Analysis](#toc1_4_1_1_)
#       - [Initial Data Inspection](#toc1_4_1_2_)
#       - [Dataset Information](#toc1_4_1_3_)
#     - [Missing Value Across Categories](#toc1_4_2_)
#       - [Possible Causes of Missing Data](#toc1_4_2_1_)
#       - [Class Balance Analysis](#toc1_4_2_2_)
#       - [Correlation Analysis](#toc1_4_2_3_)
#     - [Key Findings:](#toc1_4_3_)
#     - [Preprocessing data](#toc1_4_4_)
#   - [Feature Engineering](#toc1_5_)
#     - [Technical Indicator Implementation Using TA-Lib](#toc1_5_1_)
#       - [Implemented Indicators:](#toc1_5_1_1_)
#     - [Feature Selection Strategy](#toc1_5_2_)
#       - [Columns Targeted for Removal](#toc1_5_2_1_)
#     - [Technical Indicator Optimization](#toc1_5_3_)
#       - [Indicator Selection Criteria](#toc1_5_3_1_)
#     - [Data Quality Enhancement](#toc1_5_4_)
#       - [Infinity Value Management](#toc1_5_4_1_)
#     - [Final Data Cleaning](#toc1_5_5_)
#       - [Quality Assurance Steps](#toc1_5_5_1_)
#
# <!-- vscode-jupyter-toc-config
# 	numbering=false
# 	anchor=true
# 	flat=false
# 	minLevel=1
# 	maxLevel=6
# 	/vscode-jupyter-toc-config -->
# <!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

# %% [markdown]
# ## <a id='toc1_2_'></a>[Data Description](#toc0_)
#
# The project utilizes three primary datasets provided in CSV format, split between training (inputs/outputs) and test inputs.
#
# ### <a id='toc1_2_1_'></a>[Input Dataset Structure](#toc0_)
# * **Identifier Features:**
#   - ID: Unique row identifier
#   - DATE: Anonymized date index
#   - STOCK: Stock identifier
#
# * **Categorical Features:**
#   - INDUSTRY: Stock's primary industry classification
#   - INDUSTRY_GROUP: Broader industry grouping
#   - SUB_INDUSTRY: Detailed industry subcategory
#   - SECTOR: High-level sector classification
#
# * **Time Series Features:**
#   - RET_1 to RET_20: Historical residual returns (20-day window)
#   - VOLUME_1 to VOLUME_20: Historical relative trading volumes (20-day window)
#
# ### <a id='toc1_2_2_'></a>[Output Dataset Structure](#toc0_)
# * ID: Corresponding identifier
# * RET: Binary target indicating return direction (1 = up, 0 = down)
#
# ------------------------------------------------------------------------------------------------
# ### <a id='toc1_2_3_'></a>[Key Financial Calculations](#toc0_)
#
# **One-Day Return Formula:**
# $$R^t = \frac{P_j^t}{P_j^{t-1}} - 1$$
# where:
# - $P_j^t$ is the price of stock j at time t
# - $P_j^{t-1}$ is the price at previous time period
#
# **Volume Normalization Process:**
# 1. Calculate relative volume using 20-day median:
# $$
# \tilde{V}^t_j = \frac{V^t_j}{\text{median}( \{ V^{t-1}_j, \ldots, V^{t-20}_j \} )}
# $$
#
# 2. Adjust for market-wide effects:
# $$
# V^t_j = \tilde{V}^t_j - \frac{1}{n} \sum_{i=1}^{n} \tilde{V}^t_i
# $$
#
# ------------------------------------------------------------------------------------------------
# ### <a id='toc1_2_4_'></a>[Dataset Scale](#toc0_)
# * Training set: 418,595 observations
# * Test set: 198,429 observations
#
# ### <a id='toc1_2_5_'></a>[Submission Guidelines](#toc0_)
# Predictions must be formatted as a two-column file (ID, RET) matching test data IDs.
#

# %% [markdown]
# ## <a id='toc1_3_'></a>[Implementation Strategy](#toc0_)
#
# Our implementation follows a systematic approach across four major phases:
# #### <a id='toc1_3_1_1_'></a>[Project Roadmap](#toc0_)
# 1. **Data Loading and Preprocessing**
#    - Robust data validation and quality checks
#    - Sophisticated missing value imputation
#    - Implementation of TA-Lib technical indicators
#    - Data cleaning including:
#      - Infinity value handling
#      - Duplicate column removal
#      - Training/test split optimization (75%/25%)
#
# 2. **Feature Engineering**
#    - Technical indicator calculation:
#      - Relative Strength Index (RSI)
#      - On-Balance Volume (OBV)
#      - Exponential Moving Averages (EMA)
#    - Advanced feature engineering
#    - Feature selection and dimensionality reduction
#    - Target encoding of categorical variables
#    - Removal of redundant technical indicators
#
# 3. **Model Development and Evaluation**
#    - Decision Tree Classifier
#       - Baseline implementation (accuracy: **0.510**)
#       - Advanced hyperparameter optimization (accuracy: **0.5325**)
#    - XGBoost Classifier
#       - Initial implementation (accuracy: **0.53**)
#       - Extensive hyperparameter tuning (accuracy: **0.8775**)
#    - Neural Network Architecture
#       - Custom feed-forward design
#       - Advanced loss function implementation
#       - Final accuracy: **0.5144**
#
# 4. **Analysis and Validation**
#    - Cross-validation assessment
#    - Feature importance ranking and interpretation
#    - ROC curve analysis
#    - Confusion matrix evaluation
#    - Technical indicator assessment
#    - Benchmark comparison and validation

# %% [markdown]
# ## Library Imports

# %%
import sys
from pathlib import Path

try:
    # VS Code
    path = Path(__file__).parent.parent
    path = path / "src"
except NameError:
    # jupyter notebook
    path = Path().absolute().parent
sys.path.append(str(path))

import kedro.ipython
from kedro.ipython import get_ipython

kedro.ipython.load_ipython_extension(get_ipython())


# %%
import logging as log
import warnings

from IPython.display import Markdown as md

from src.ml_in_finance_i_project.utils import get_node_idx, get_node_outputs

# %%
# Load the datasets
x_train_raw = catalog.load("x_train_raw")
y_train_raw = catalog.load("y_train_raw")
x_test_raw = catalog.load("x_test_raw")


# %% [markdown]
# ### <a id='toc1_3_3_'></a>[Environment Configuration and Setup](#toc0_)


# %%
## Google Colab Integration
## Note: Must be executed before other library imports
def setup_colab_environment():
    """
    Establishes Google Colab environment with necessary configurations:
    - Mounts Google Drive
    - Creates required symbolic links
    - Sets up project directory structure

    Returns:
        bool: True if running in Colab, False otherwise
    """
    try:
        import os

        from google.colab import drive

        drive.mount("/content/drive")
        req_symlinks = [
            ("data", "ml_in_finance_i_project/data"),
            ("src", "ml_in_finance_i_project/src"),
        ]
        # Create symlinks if they don't exist
        for dest, src in req_symlinks:
            if not os.path.exists(dest):
                os.symlink(f"/content/drive/Othercomputers/My Mac/{src}", dest)
        return True

    except ImportError:
        return False


# %% [markdown]
# #### <a id='toc1_3_3_1_'></a>[Pipeline Node Execution Configuration](#toc0_)
# Defining function for running specific pipeline nodes


# %%
def run_pipeline_node(pipeline_name: str, node_name: str, inputs: dict):
    """
    Executes a specific node within the data processing pipeline.

    Parameters:
        pipeline_name (str): Target pipeline identifier
        node_name (str): Specific node to execute
        inputs (dict): Node input parameters

    Returns:
        Output from node execution
    """
    node_idx = get_node_idx(pipelines[pipeline_name], node_name)
    return pipelines[pipeline_name].nodes[node_idx].run(inputs)


# %% [markdown]
# ##### <a id='toc1_3_3_1_1_'></a>[Environment Initialization](#toc0_)

# %%
IN_COLAB = setup_colab_environment()

# Configure logging to stdout
log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[log.StreamHandler()],
)
conf_params = context.config_loader.get("parameters")
target = conf_params["model_options"]["target"]
kfold = conf_params["model_options"]["kfold"]

# %% [markdown]
# ## <a id='toc1_4_'></a>[Data Loading and Initial Processing](#toc0_)
#
# ### <a id='toc1_4_1_'></a>[Preprocessing Strategy](#toc0_)
# * Handling of missing values (NA removal)
# * Target variable encoding (boolean to binary conversion)
# * Optional data subsetting capabilities:
#   - Fraction loading for rapid prototyping
#   - Time window selection (e.g., using first n days)

# %%
out = run_pipeline_node(
    "data_processing",
    "load_data_node",
    {
        "x_train_raw": x_train_raw,
        "y_train_raw": y_train_raw,
        "x_test_raw": x_test_raw,
        "params:sample_n": conf_params["sample_n"],
    },
)
# %% [markdown]
# #### <a id='toc1_4_1_1_'></a>[Visual Data Analysis](#toc0_)
#
# * Time series visualization of returns and volume
# * Target variable (RET) highlighted in red for clarity

# %%
run_pipeline_node(
    "reporting",
    "plot_returns_volume_node",
    {
        "train_df": out["train_df"],
        "params:example_row_id": 4,
    },
)["returns_volume_plot"]

# %% [markdown]
# #### <a id='toc1_4_1_2_'></a>[Initial Data Inspection](#toc0_)
#
# From visual inspection, we can see that there are missing / invalid values in the data

# %%
out["test_df"].head()

# %% [markdown]
# #### <a id='toc1_4_1_3_'></a>[Dataset Information](#toc0_)

# %%
print("Training Dataset Info:")
out["train_df"].info()
print("\nTest Dataset Info:")
out["test_df"].info()

# %% [markdown]
# ### <a id='toc1_4_2_'></a>[Missing Value Across Categories](#toc0_)
#
#
# #### <a id='toc1_4_2_1_'></a>[Possible Causes of Missing Data](#toc0_)
#
# 1. **Market Structure**
#    - Weekend/holiday market closures
#    - Different trading venues (NYSE, NASDAQ, CBOE)
#
# 2. **Data Quality Issues**
#    - Collection inconsistencies
#    - Date anonymization effects
#
# 3. **Technical Computation Effects**
#    - Rolling window calculations (20-day impact)
#    - Weekend/holiday volume calculations
#
# 4. **Market Events**
#    - Trading suspensions
#    - Stock delistings
#    - Low liquidity periods
#
# 5. **Intentional Design**

# %%
run_pipeline_node(
    "reporting",
    "plot_nan_percentages_node",
    {"train_df": out["train_df"]},
)["nan_percentages_plot"]

# %% [markdown]
# #### <a id='toc1_4_2_2_'></a>[Class Balance Analysis](#toc0_)
#
# The target variable shows near-perfect class balance, which is expected given:
# 1. The target represents return sign (+/-)
# 2. Market efficiency theory implies roughly equal probability of up/down movements
# 3. Slight positive bias may indicate:
#    - General market upward trend
#    - Risk premium effects
#    - Survivorship bias in the dataset

# %%
md(
    f"Class imbalance: {out['train_df']['RET'].value_counts(normalize=True)[0] * 100:.2f}%"
    + f" {out['train_df']['RET'].value_counts(normalize=True)[1] * 100:.2f}%"
)

# %% [markdown]
# #### <a id='toc1_4_2_3_'></a>[Correlation Analysis](#toc0_)
#
# ### <a id='toc1_4_3_'></a>[Key Findings:](#toc0_)
#
# 1. **Return Correlations**
#    - Minimal cross-period return correlation (market efficiency)
#    - Stronger correlations between adjacent time periods
#    - Pattern suggests market efficiency (limited predictability)
#
# 2. **Volume Relationships**
#    - Relatively high volume autocorrelation
#    - Volume clustering indicates market regime patterns
#    - Strong volume-volatility relationship

# %%
out_corr = run_pipeline_node(
    "reporting",
    "plot_correlation_matrix_node",
    {"train_df": out["train_df"]},
)
out_corr["correlation_matrix_plot"]
# %% [markdown]
# ### <a id='toc1_4_4_'></a>[Preprocessing data](#toc0_)
# * Dropping rows with missing returns
# * Dropping NA values and ID columns
# * Converting target to binary

# %%
out_preprocessed = run_pipeline_node(
    "data_processing",
    "preprocess_data_node",
    {"train_df": out["train_df"], "test_df": out["test_df"]},
)
# %% [markdown]
# ## <a id='toc1_5_'></a>[Feature Engineering](#toc0_)
#
# * Extending variable set from competition organizers
# * Feature development based on technical analysis indicators
# * Statistical feature calculation

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    out2 = run_pipeline_node(
        "data_processing",
        "calculate_statistical_features_node",
        {
            "train_df_preprocessed": out_preprocessed["train_df_preprocessed"],
            "test_df_preprocessed": out_preprocessed["test_df_preprocessed"],
        },
    )

# %% [markdown]
# ### <a id='toc1_5_1_'></a>[Technical Indicator Implementation Using TA-Lib](#toc0_)
#
# This section implements advanced technical analysis indicators for both training and test datasets.
# Results are persisted to optimize computation time.
#
# #### <a id='toc1_5_1_1_'></a>[Implemented Indicators:](#toc0_)
# 1. **Volume Indicators**
#    - On-Balance Volume (OBV)
#
# 2. **Momentum Indicators**
#    - Relative Strength Index (RSI)
#    - Momentum (MOM) - 5 period
#    - Rate of Change Ratio (ROCR) - 5 period
#    - Chande Momentum Oscillator (CMO) - 14 period
#
# 3. **Moving Averages**
#    - Exponential Moving Average (EMA) - 5 period
#    - Simple Moving Average (SMA) - 5 period
#    - Weighted Moving Average (WMA) - 5 period
#    - Midpoint Price (MIDPOINT) - 10 period
#
# Note: Computation is time-intensive for full dataset

# %%
calculate = False

if calculate:
    out3 = run_pipeline_node(
        "data_processing",
        "calculate_technical_indicators_node",
        {
            "train_df_statistical_features": out2["train_df_statistical_features"],
            "test_df_statistical_features": out2["test_df_statistical_features"],
            "params:features_ret_vol": conf_params["features_ret_vol"],
        },
    )
else:
    out3 = get_node_outputs(
        pipelines["data_processing"].nodes[
            get_node_idx(
                pipelines["data_processing"], "calculate_technical_indicators_node"
            )
        ],
        catalog,
    )

# %% [markdown]
# ### <a id='toc1_5_2_'></a>[Feature Selection Strategy](#toc0_)
#
# #### <a id='toc1_5_2_1_'></a>[Columns Targeted for Removal](#toc0_)
# 1. **Identifier Columns**
#    - While potentially predictive, excluded for model generalization
#    - Includes: ID, STOCK, DATE
#
# 2. ** Remaining Categorical Features**
#    - INDUSTRY, INDUSTRY_GROUP, SECTOR, SUB_INDUSTRY
#    - Used in feature engineering but removed from final model
#
# Rationale: Focus on price/volume dynamics rather than static characteristics

# %%
out4 = run_pipeline_node(
    "data_processing",
    "drop_id_cols_node",
    {
        "train_ta_indicators": out3["train_ta_indicators"],
        "test_ta_indicators": out3["test_ta_indicators"],
    },
)

# %% [markdown]
# ### <a id='toc1_5_3_'></a>[Technical Indicator Optimization](#toc0_)
#
# #### <a id='toc1_5_3_1_'></a>[Indicator Selection Criteria](#toc0_)
# 1. **Relevance**
#    - Remove indicators with minimal predictive value
#    - Focus on non-redundant signals
#
# 2. **Complexity Reduction**
#    - Reduce feature space to prevent overfitting
#    - Example: Similar-period SMAs offer limited additional value
#
# 3. **Model Performance Impact**
#    - Retain indicators that demonstrate predictive power
#    - Remove those that may introduce noise

# %%
out5 = run_pipeline_node(
    "data_processing",
    "drop_obsolete_technical_indicators_node",
    {
        "train_ta_indicators_dropped": out4["train_ta_indicators_dropped"],
        "test_ta_indicators_dropped": out4["test_ta_indicators_dropped"],
        "params:target": conf_params["model_options"]["target"],
    },
)

# %% [markdown]
# ### <a id='toc1_5_4_'></a>[Data Quality Enhancement](#toc0_)
#
# #### <a id='toc1_5_4_1_'></a>[Infinity Value Management](#toc0_)
# * Critical for model stability
# * Particularly important for neural network training
# * Prevents numerical computation issues

# %%
out6 = run_pipeline_node(
    "data_processing",
    "filter_infinity_values_node",
    {
        "train_df_technical_indicators": out5["train_df_technical_indicators"],
        "test_df_technical_indicators": out5["test_df_technical_indicators"],
        "params:target": conf_params["model_options"]["target"],
    },
)

# %% [markdown]
# ### <a id='toc1_5_5_'></a>[Final Data Cleaning](#toc0_)
#
# #### <a id='toc1_5_5_1_'></a>[Quality Assurance Steps](#toc0_)
# 1. **Duplicate Resolution**
#    - Remove redundant columns
#    - Ensure data uniqueness
#
# 2. **Missing Value Treatment**
#    - Second-pass NaN handling
#    - Address gaps from technical indicator calculation
#
# Purpose: Ensure data quality for model training

# %%
# Remove duplicated columns and handle NaN values
out7 = run_pipeline_node(
    "data_processing",
    "remove_duplicates_and_nans_node",
    {
        "train_df_filtered": out6["train_df_filtered"],
        "test_df_filtered": out6["test_df_filtered"],
    },
)
