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
# # Stock Market Movement Prediction: A Machine Learning Approach
# ### Using Advanced ML Techniques to Forecast Next-Day Stock Price Movements
#
# This project tackles a challenging binary classification problem in quantitative finance - predicting the directional movement
# (up or down) of individual US stocks for the following trading day. The goal is to develop a sophisticated machine learning model
# that can provide data-driven insights to support investment decision-making.
#
# The model leverages a comprehensive feature set including:
# - 20 days of historical price returns
# - 20 days of trading volume data
# - Categorical stock metadata (industry, sector, etc.)
# - Technical indicators and statistical features
#
# A public benchmark accuracy of 51.31% was previously achieved using a Random Forest model with 5 days of historical data and
# sector-level information. Our approach aims to improve upon this by incorporating more features and advanced modeling techniques.
#
# #### Project Roadmap
# 1. **Comprehensive Data Preprocessing**
#    - Robust data loading and validation
#    - Sophisticated missing value imputation
#    - Advanced feature engineering including technical indicators
#    - Target encoding of categorical variables
#
# 2. **Model Implementation and Rigorous Evaluation**
#    - Decision Tree Classifier
#       - Baseline implementation (accuracy: 0.510)
#       - Advanced hyperparameter optimization (accuracy: 0.5325)
#    - XGBoost Classifier
#       - Initial implementation (accuracy: 0.53)
#       - Extensive hyperparameter tuning (accuracy: 0.8775)
#    - Neural Network Architecture
#       - Custom implementation achieving 0.5144 accuracy
#
# 3. **In-depth Model Analysis**
#    - Comprehensive cross-validation assessment
#    - Feature importance ranking and interpretation
#    - Detailed ROC curve analysis
#    - Confusion matrix evaluation
#
# %% [markdown]
# ## Detailed Data Description
#
# The project utilizes three primary datasets provided in CSV format, split between training (inputs/outputs) and test inputs.
#
# ### Input Dataset Structure (47 columns):
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
# ### Output Dataset Structure:
# * ID: Corresponding identifier
# * RET: Binary target indicating return direction (1 = up, 0 = down)
#
# ------------------------------------------------------------------------------------------------
# ### Key Financial Calculations
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
# ### Dataset Scale:
# * Training set: 418,595 observations
# * Test set: 198,429 observations
#
# ### Submission Guidelines:
# Predictions must be formatted as a two-column file (ID, RET) matching test data IDs.
#
# %% [markdown]
# ## Detailed Implementation Strategy
#
# Our implementation follows a systematic approach across four major phases:
#
# 1. **Advanced Data Loading and Preprocessing**
#    - Robust data validation and quality checks
#    - Sophisticated missing value treatment
#    - Implementation of TA-Lib technical indicators
#    - Comprehensive data cleaning including:
#      - Infinity value handling
#      - Duplicate column removal
#      - Training/test split optimization (75%/25%)
#
# 2. **Sophisticated Feature Engineering**
#    - Advanced technical indicator calculation:
#      - Relative Strength Index (RSI)
#      - On-Balance Volume (OBV)
#      - Exponential Moving Averages (EMA)
#    - Efficient data persistence using pickle format
#    - Strategic feature selection and dimensionality reduction
#    - Removal of redundant technical indicators
#
# 3. **Comprehensive Model Development**
#    - Decision Tree Classifier
#       - Baseline model development
#       - Advanced hyperparameter optimization
#    - Gradient Boosting Implementation
#       - Iterative parameter tuning
#       - Multi-stage optimization process
#    - Neural Network Architecture
#       - Custom feed-forward design
#       - Advanced loss function implementation
#
# 4. **Thorough Model Evaluation**
#    - Cross-model performance comparison
#    - Detailed feature importance analysis
#    - Comprehensive technical indicator assessment
#    - In-depth overfitting analysis
#    - Benchmark comparison and validation
#
# %% [markdown]
# ### Essential Library Imports
# %%
import sys
from pathlib import Path

try:
    # vscode
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

from IPython.display import Markdown as md

from src.ml_in_finance_i_project.utils import get_node_idx, get_node_outputs

# %%
# Load the datasets
x_train_raw = catalog.load("x_train_raw")
y_train_raw = catalog.load("y_train_raw")
x_test_raw = catalog.load("x_test_raw")


# %% [markdown]
# ### Environment Configuration and Setup
# Configuring the computational environment for optimal performance


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
# #### Pipeline Node Execution Configuration
# Defining core functionality for running specific pipeline nodes


# %% Run a specific node from a pipeline.
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
# ##### Environment Initialization

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
# ## Data Loading and Initial Processing
#
# ### Preprocessing Strategy
# * Systematic handling of missing values (NA removal)
# * Target variable encoding (boolean to binary conversion)
# * Optional data subsetting capabilities:
#   - Fraction loading for rapid prototyping
#   - Time window selection (e.g., using first n days)
# %%
# Set data directory based on environment
# Run data processing pipeline node
out = run_pipeline_node(
    "data_processing",
    "load_and_preprocess_data_node",
    {
        "x_train_raw": x_train_raw,
        "y_train_raw": y_train_raw,
        "x_test_raw": x_test_raw,
        "params:remove_id_cols": conf_params["remove_id_cols"],
        "params:n_days": conf_params["n_days"],
        "params:sample_n": conf_params["sample_n"],
    },
)


# %% [markdown]
# #### Visual Data Analysis
#
# * Time series visualization of returns and volume
# * Target variable (RET) highlighted in red for clarity
# %%
# Plot returns and volume
run_pipeline_node(
    "reporting",
    "plot_returns_volume_node",
    {
        "train_df": out["train_df"],
        "params:example_row_id": 2,
    },
)["returns_volume_plot"]


# %% [markdown]
# #### Initial Data Inspection
#
# * Raw data quality assessment
# * Identification of data cleaning requirements
# * Missing value patterns
# %%
out["test_df"].head()
# %% [markdown]
# #### Comprehensive Dataset Information
# %%
print("Training Dataset Info:")
out["train_df"].info()
print("\nTest Dataset Info:")
out["test_df"].info()

# %% [markdown]
# #### Missing Value Analysis Across Categories
#
# ### Common Causes of Missing Data:
#
# 1. **Market Structure**
#    - Weekend/holiday market closures
#    - Different trading venues (NYSE, NASDAQ, CBOE)
#
# 2. **Data Quality Issues**
#    - Collection inconsistencies
#    - Date anonymization effects
#
# 3. **Calculation Artifacts**
#    - Rolling window calculations (20-day impact)
#    - Weekend/holiday volume calculations
#
# 4. **Market Events**
#    - Trading suspensions
#    - Stock delistings
#    - Low liquidity periods
#
# 5. **Intentional Design**
#    - Challenge complexity enhancement
#    - Realistic market simulation

# %%
run_pipeline_node(
    "reporting",
    "plot_nan_percentages_node",
    {"train_df": out["train_df"]},
)["nan_percentages_plot"]

# %% [markdown]
# #### Class Balance Analysis
#
# The target variable shows near-perfect class balance, which is expected given:
# 1. The target represents return sign (+/-)
# 2. Market efficiency theory suggests roughly equal probability of up/down movements
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
# #### Correlation Analysis
#
# ### Key Findings:
#
# 1. **Return Correlations**
#    - Minimal cross-stock return correlation (market efficiency)
#    - Stronger correlations between adjacent time periods
#    - Pattern suggests market efficiency (limited predictability)
#
# 2. **Volume Relationships**
#    - High volume autocorrelation due to calculation method
#    - Volume clustering indicates market regime patterns
#    - Strong volume-volatility relationship
#
# 3. **Lead-Lag Effects**
#    - Significant volume-return relationship
#    - Previous day's volume predicts next day's return
#    - Potential trading signal indicator
# %%
out_corr = run_pipeline_node(
    "reporting",
    "plot_correlation_matrix_node",
    {"train_df": out["train_df"]},
)
out_corr["correlation_matrix_plot"]
# %% [markdown]
#

# %% [markdown]
# ## Advanced Feature Engineering
#
# * Extended variable set from competition organizers
# * Custom feature development
# * Statistical feature calculation

# %%


import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    out2 = run_pipeline_node(
        "data_processing",
        "calculate_statistical_features_node",
        {
            "train_df": out["train_df"],
            "test_df": out["test_df"],
        },
    )

# %% [markdown]
# ### Technical Indicator Implementation Using TA-Lib
#
# This section implements advanced technical analysis indicators for both training and test datasets.
# Results are persisted to optimize computation time.
#
# ### Implemented Indicators:
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
calculate = True

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
    out5 = get_node_outputs(
        pipelines["data_processing"].nodes[
            get_node_idx(
                pipelines["data_processing"], "calculate_technical_indicators_node"
            )
        ],
        catalog,
    )

# %% [markdown]
# ### Feature Selection Strategy
#
# #### Columns Targeted for Removal:
# 1. **Identifier Columns**
#    - While potentially predictive, excluded for model generalization
#    - Includes: ID, STOCK, DATE
#
# 2. **Categorical Features**
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
# #### Technical Indicator Optimization
#
# ### Indicator Selection Criteria:
# 1. **Relevance**
#    - Remove indicators with minimal predictive value
#    - Focus on non-redundant signals
#
# 2. **Complexity Management**
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
# #### Data Quality Enhancement
#
# ### Infinity Value Management
# * Critical for model stability
# * Particularly important for neural network training
# * Prevents numerical computation issues

# %% Filter out infinity values
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
# #### Final Data Cleaning
#
# ### Quality Assurance Steps:
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
