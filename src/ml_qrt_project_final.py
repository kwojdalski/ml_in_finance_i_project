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
# # Stock Market Movement Prediction
# ### Using Machine Learning to Forecast Next-Day Stock Price Movements
# This project tackles a binary classification problem in financial markets -
# predicting whether individual US stocks will move up or down the following day.
# The goal is to develop a model that can assist in making data-driven investment decisions.
# The model uses 20 days of historical price returns and trading volumes, along with
# categorical stock metadata like industry and sector classifications, to identify
# predictive patterns in market behavior.
# The public benchmark accuracy of 51.31% was achieved using a Random Forest model that considered
# the previous 5 days of data along with the average sector returns from the prior day.
#
# #### Agenda
# 1. **Data Preprocessing**
#    - Loading training and test datasets
#    - Handling missing values and target encoding
#    - Feature engineering (technical indicators of different types)
# 2. **Model Implementation and Evaluation**
#    - Decision Tree Classifier
#       - Baseline model (accuracy: 0.510)
#       - Tuned model with hyperparameter optimization (accuracy: 0.5325)
#    - XGBoost Classifier
#       - Baseline model (accuracy: 0.53)
#       - Tuned model with hyperparameter optimization (accuracy: 0.8775)
#    - Neural Network
#       - Accuracy: 0.5144
# 3. **Model Comparison**
#    - Cross-validation results
#    - Feature importance analysis
#    - ROC curves and confusion matrices
#
# %% [markdown]
# ## Data description
#
# 3 datasets are provided as csv files, split between training inputs and outputs, and test inputs.
#
# Input datasets comprise 47 columns: the first ID column contains unique row identifiers while the other 46 descriptive features correspond to:
#
# * **DATE**: an index of the date (the dates are randomized and anonymized so there is no continuity or link between any dates),
# * **STOCK**: an index of the stock,
# * **INDUSTRY**: an index of the stock industry domain (e.g., aeronautic, IT, oil company),
# * **INDUSTRY_GROUP**: an index of the group industry,
# * **SUB_INDUSTRY**: a lower level index of the industry,
# * **SECTOR**: an index of the work sector,
# * **RET_1 to RET_20**: the historical residual returns among the last 20 days (i.e., RET_1 is the return of the previous day and so on),
# * **VOLUME_1 to VOLUME_20**: the historical relative volume traded among the last 20 days (i.e., VOLUME_1 is the relative volume of the previous day and so on),
#
# Output datasets are only composed of 2 columns:
#
# * **ID**: the unique row identifier (corresponding to the input identifiers)
# and the binary target:
# * **RET**: the sign of the residual stock return at time $t$
#
# ------------------------------------------------------------------------------------------------
# The one-day return of a stock :
# $$R^t = \frac{P_j^t}{P_j^{t-1}} - 1$$
#
# The volume is the ratio of the stock volume to the median volume of the past 20 days.
# The relative volumes are computed using the median of the past 20 days' volumes.
# If any day within this 20-day window has a missing volume value, it will cause NaN values in the calculation for subsequent days.
# For example, if there is a missing value on day $D$, then the relative volumes for days $D$ to $D+19$ will be affected.
#
# The relative volume $\tilde{V}^t_j$ at time $t$ of a stock $j$ is calculated as:
# $$
# \tilde{V}^t_j = \frac{V^t_j}{\text{median}( \{ V^{t-1}_j, \ldots, V^{t-20}_j \} )}
# $$
#
# The adjusted relative volume $V^t_j$ is then given by:
# $$
# V^t_j = \tilde{V}^t_j - \frac{1}{n} \sum_{i=1}^{n} \tilde{V}^t_i
# $$
# ------------------------------------------------------------------------------------------------
# Guidelines from the organizers:
# The solution files submitted by participants shall follow this output dataset format (i.e contain only two columns, ID and RET, where the ID values correspond to the input test data).
# An example submission file containing random predictions is provided.
#
# **418595 observations (i.e. lines) are available for the training datasets while 198429 observations are used for the test datasets.**
#


# %% [markdown]
# ## Implementation Steps
#
# This notebook implements the following steps:
#
# 1. **Data Loading and Preprocessing**
#    - Load training and test datasets
#    - Handle missing values and data cleaning
#    - Calculate technical indicators using TA-Lib
#    - Filter out infinity values and remove duplicated columns
#    - Split data into training and test sets (75%/25% split)
#
# 2. **Feature Engineering**
#    - Calculate technical indicators like RSI, OBV, EMA etc.
#    - Save indicators to pickle files for reuse
#    - Drop unnecessary ID and categorical columns
#    - Remove redundant technical indicators
#
# 3. **Model Development and Tuning**
#    - Decision Tree Classifier
#       - Baseline model (accuracy: 0.510)
#       - Tuned model with hyperparameters (accuracy: 0.533)
#    - Gradient Boosting
#       - Stepwise tuning of n_estimators, tree params, leaf params
#       - Best model achieves significant improvement
#    - Neural Network
#       - Simple feed-forward architecture
#       - Training with BCE loss and Adam optimizer
#
# 4. **Model Comparison and Analysis**
#    - Compare accuracy across all models
#    - Analyze feature importance
#    - Key findings on model performance and technical indicators
#    - Discussion of overfitting and benchmark results


# %% [markdown]
# ### Importing libraries
import sys
from pathlib import Path

path = Path(__file__).parent.parent
path = path / "src"
sys.path.append(str(path))

import kedro.ipython

kedro.ipython.load_ipython_extension(get_ipython())


# %%
import logging as log
from itertools import compress
from pathlib import Path

import numpy as np
import torch
from IPython.display import Markdown as md
from kedro.framework.session import KedroSession
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.GBClassifierGridSearch import HistGBClassifierGridSearch
from src.ml_in_finance_i_project.pipelines.reporting.nodes import (
    feature_importance,
    plot_model_accuracy,
    simulate_strategy,
)
from src.ml_in_finance_i_project.utils import get_node_idx, get_node_outputs
from src.nn import Net


def run_pipeline_node(pipeline_name: str, node_name: str, inputs: dict) -> dict:
    """Run a specific node from a pipeline."""
    with KedroSession.create() as session:
        context = session.load_context()
        pipeline = context.pipelines[pipeline_name]
        node = [n for n in pipeline.nodes if n.name == node_name][0]
        return node.run(inputs)


# %%
# load the configuration file


# conf_loader = OmegaConfigLoader(".", base_env="", default_run_env="")
# # Read the configuration file
# conf_params = conf_loader["parameters"]
# conf_catalog = conf_loader["catalog"]

# # Create the DataCatalog instance from the configuration
# catalog = DataCatalog.from_config(conf_catalog)
# Load the datasets
x_train_raw = catalog.load("x_train_raw")
y_train_raw = catalog.load("y_train_raw")
x_test_raw = catalog.load("x_test_raw")


# %% [markdown]
# ### Checking and configuring environment


# %%
## Google Colab used to speed up the computation in xgboost model
## warning: this function must be run before importing libraries
# if running in Google Colab
def setup_colab_environment():
    """
    Set up Google Colab environment by mounting drive and creating symlinks.
    Returns True if running in Colab, False otherwise.
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


# %%
# Run a specific node from a pipeline.
def run_pipeline_node(pipeline_name: str, node_name: str, inputs: dict):
    """Run a specific node from a pipeline.

    Args:
        pipeline_name: Name of the pipeline
        node_name: Name of the node to run
        inputs: Dictionary of input parameters for the node

    Returns:
        Output from running the node
    """
    node_idx = get_node_idx(pipelines[pipeline_name], node_name)
    return pipelines[pipeline_name].nodes[node_idx].run(inputs)


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
# ## Loading data
# %%
# Set data directory based on environment
# Run data processing pipeline node
out = run_pipeline_node(
    "data_processing",
    "load_data_node",
    {
        "x_train_raw": x_train_raw,
        "y_train_raw": y_train_raw,
        "x_test_raw": x_test_raw,
    },
)


#  %% [markdown]
# #### Problem visualization
# %%
# Plot returns and volume
run_pipeline_node(
    "reporting",
    "plot_returns_volume_node",
    {"train_df": out["train_df"], "params:example_row_id": 28},
)["returns_volume_plot"]


# %% [markdown]
# #### Head of the data
# %%
out["test_df"].head()
# %% [markdown]
# #### Info about the dataset
# %%
print("Training Dataset Info:")
out["train_df"].info()
print("\nTest Dataset Info:")
out["test_df"].info()

# %% [markdown]
# #### Plot nan percentages across categorical var

# %%
run_pipeline_node(
    "reporting",
    "plot_nan_percentages_node",
    {"cleaned_train": out["train_df"]},
)["nan_percentages_plot"]

# %% [markdown]

# #### Possible reasons for missing values:
# 1. Market closures (market data might be missing for weekends and public holidays) - some dates clearly show a very high
# percentage of missing values
# 2. Data collection issues (for instance, market data might come from different US venues, e.g. NYSE, NASDAQ, CBOE, etc.)
# 3. Randomization and anonymization of dates
# 4. The way relative volumes are calculated (one day missing causes missing values for the next 19 days) - could have something to do with calculating volumes on weekends / public holidays
# 5. Done on purpose by the organizers to make the problem more challenging
# 6. Some stocks might be delisted or suspended from trading (reference data problem) - some stocks in fact have up to 100% missing values
# 7. Some stocks might be barely trading (either due to low volume or in a non-continuous manner)


# %% Dropping rows with NAs for the most important variables (RET_1 to RET_5)
# The assumption is that the most recent values for regressors are the most important

out2 = run_pipeline_node(
    "data_processing", "drop_missing_returns_node", {"train_df": out["train_df"]}
)

# %% [markdown]
# #### Preprocessing data
# * Dropping rows with NAs
# * If arg set to true, removing ID columns
# * RET is encoded from bool to binary
# %%
out3 = run_pipeline_node(
    "data_processing",
    "preprocess_data_node",
    {
        "train_df_dropped": out2["train_df_dropped"],
        "test_df": out["test_df"],
        "params:remove_id_cols": conf_params["remove_id_cols"],
    },
)

# %% [markdown]
# #### Check Class Imbalance
#
# **Class Imbalance**:
# Classes seem to be balanced almost perfectly. This is expected, as the target variable is the sign of the return.
# Intuitively, it is expected that the sign of the return is more likely to be positive (by a small margin) than negative
# unless data comes from a bear market period.
# %%
md(
    f"Class imbalance: {out3['train_df_preprocessed']['RET'].value_counts(normalize=True)[0] * 100:.2f}%"
    + f" {out3['train_df_preprocessed']['RET'].value_counts(normalize=True)[1] * 100:.2f}%"
)


# %% [markdown]
# #### Plot correlation matrix

# Findings:
# * Most stock returns are nearly not correlated with each other (this is expected).
# Otherwise, someone could make a lot of money
# by exploiting this non-subtle pattern.
#     * Eventually, excess alpha would converge to 0
# * Among stock returns the strongest correlation is within stock returns adjacent to each other (e.g. $RET_1$ and $RET_2$)
#     * This is expected as the magnitude return of a stock is likely to be correlated with the magnitude stock return of the previous day
# * Volumes are highly correlated (this is kind of expected) due to the way $VOLUME_i$ variables are calculated.
# Moreover, Volatility and Volumes tend to cluster. Hence, correlation is positive.
# * There is a strong positive correlation between the volume of the previous day and the return of the following day
# %%
out_corr = run_pipeline_node(
    "reporting",
    "plot_correlation_matrix_node",
    {"train_df": out3["train_df_preprocessed"]},
)
out_corr
# %% [markdown]
# ## Feature Engineering - Technical Indicators using TA-Lib
# In this part, we calculate the technical indicators for the train and test data.
# We save the results in pickle files to avoid recalculating them every time.
# The following functions inside the function are used:
# - talib.OBV, {"data_type": "both"}),
# - talib.RSI, {"data_type": "ret"}),
# - talib.MOM, {"timeperiod": 5, "data_type": "ret"}),
# - talib.ROCR, {"timeperiod": 5, "data_type": "ret"}),
# - talib.CMO, {"timeperiod": 14, "data_type": "ret"}),
# - talib.EMA, {"timeperiod": 5, "data_type": "ret"}),
# - talib.SMA, {"timeperiod": 5, "data_type": "ret"}),
# - talib.WMA, {"timeperiod": 5, "data_type": "ret"}),
# - talib.MIDPOINT, {"timeperiod": 10, "data_type": "ret"}),

# %% [markdown]
# #### Feature engineering - cont'd

# %%
# This comes from organizers' notebook, it's an extended version of variables they used
# Feature engineering
# Calculate statistical features

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    out4 = run_pipeline_node(
        "data_processing",
        "calculate_statistical_features_node",
        {
            "train_df_preprocessed": out3["train_df_preprocessed"],
            "test_df_preprocessed": out3["test_df_preprocessed"],
        },
    )

# %% [markdown]
# ### Calculate technical indicators
# %%
calculate = True

if calculate:
    out5 = run_pipeline_node(
        "data_processing",
        "calculate_technical_indicators_node",
        {
            "train_df_statistical_features": out4["train_df_statistical_features"],
            "test_df_statistical_features": out4["test_df_statistical_features"],
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
# #### Columns to drop
# They could bring in some predictive power, but we don't want to use them in this case
# as the scope is limited for this project
# ['ID', 'STOCK', 'DATE', 'INDUSTRY', 'INDUSTRY_GROUP', 'SECTOR', 'SUB_INDUSTRY']
# %%
out6 = run_pipeline_node(
    "data_processing",
    "drop_id_cols_node",
    {
        "train_ta_indicators": out5["train_ta_indicators"],
        "test_ta_indicators": out5["test_ta_indicators"],
    },
)

# %% [markdown]
# Assumption is that probably some technical indicators are not useful for the prediction.
# For instance SMA(10), SMA(11) etc. dont give any information in the context of RET.
# It's an arbitrary choice, but we want to keep the number of features low

out7 = run_pipeline_node(
    "data_processing",
    "drop_technical_indicators_node",
    {
        "train_ta_indicators_dropped": out6["train_ta_indicators_dropped"],
        "params:target": conf_params["model_options"]["target"],
    },
)
# %% Filter out infinity values
out8 = run_pipeline_node(
    "data_processing",
    "drop_infinity_values_node",
    {
        "train_df_technical_indicators": out7["train_df_technical_indicators"],
        "test_df_technical_indicators": out7["test_df_technical_indicators"],
    },
)

# %%
# Remove duplicated columns
out9 = run_pipeline_node(
    "data_processing",
    "drop_duplicated_columns_node",
    {
        "train_df_rm_duplicates": out8["train_df_rm_duplicates"],
        "test_df_rm_duplicates": out8["test_df_rm_duplicates"],
    },
)

# %%
# Features
# Default selection
features = out9["train_df_rm_duplicates"].columns.drop(target).tolist()

# %% [markdown]
# ## ML DecisionTreeClassifier

# %%
# Train and test set splitting
out10 = run_pipeline_node(
    "data_processing",
    "split_data_node",
    {
        "train_df_rm_duplicates": out9["train_df_rm_duplicates"],
        "test_df_rm_duplicates": out9["test_df_rm_duplicates"],
    },
)


# %%
x_train = out10["X_train"]
x_test = out10["X_test"]
y_train = out10["y_train"]
y_test = out10["y_test"]


# %%
# Decison tree baseline model
base_dt = tree.DecisionTreeClassifier()

# %% [markdown]
# #### Fit the model
# `model_fit()` function a model, makes predictions, and evaluates performance
# using confusion matrix, accuracy score,
# cross-validation, ROC curve and feature importance analysis.

# %%
model_fit(base_dt, x_train, y_train, features, performCV=False)
log.info(f"Accuracy on test set: {base_dt.score(x_test, y_test):.3f}")

# %% [markdown]
# #### Tunning Decision tree model  With Gridsearch

# %%
log.info("Decision tree with Classifier")
params = {"max_depth": np.arange(2, 7), "criterion": ["gini", "entropy"]}
tree_estimator = tree.DecisionTreeClassifier()

grid_tree = GridSearchCV(
    tree_estimator, params, cv=kfold, scoring="accuracy", n_jobs=1, verbose=False
)

grid_tree.fit(x_train, y_train)
best_est = grid_tree.best_estimator_
log.info(best_est)
log.info(grid_tree.best_score_)


# %% [markdown]
# #### Summarize results and choose best parameters

# %%
log.info(f"Best: {grid_tree.best_score_} using {grid_tree.best_params_}")
means = grid_tree.cv_results_["mean_test_score"]
stds = grid_tree.cv_results_["std_test_score"]
params = grid_tree.cv_results_["params"]

# Store best parameters
for mean, stdev, param in zip(means, stds, params):
    log.info(f"{mean} ({stdev}) with: {param}")
best_params = grid_tree.best_params_

md(
    "the best Hyperparameters for our Decision tree"
    + f"model using gridsearch cv is {best_params}"
)
max_depth_ = best_params["max_depth"]
# %%
base_dt = tree.DecisionTreeClassifier(max_depth=max_depth_, criterion="gini")
model_fit(base_dt, x_train, y_train, features, printFeatureImportance=True)

dt = tree.DecisionTreeClassifier()
dt.fit(x_train, y_train)

# %% [markdown]
# #### Feature selection
# Based on feature importances
# %%
# ## visualize feature importance
threshold = 0.01
feature_importance(base_dt, features, threshold)

# %%
# **features with less than 1% of feature importance**
n_features = list(compress(features, base_dt.feature_importances_ >= threshold))

# %% [markdown]
# New sets with only the selected features

# %%
x_train_sl, x_test_sl, y_train_sl, y_test_sl = train_test_split(
    train_df.loc[:, train_df[n_features].columns], train_df["RET"], random_state=0
)

# %% [markdown]
# #### Decision tree tuned model
# %%
grid_dt = tree.DecisionTreeClassifier(max_depth=max_depth_, criterion="gini")

# %%
log.info("Fitting with train set")
model_fit(grid_dt, x_train_sl, y_train_sl, n_features, printFeatureImportance=True)

# %%
log.info("Fitting with test set")
model_fit(
    grid_dt,
    x_test_sl,
    y_test_sl,
    n_features,
    printFeatureImportance=False,
    roc=True,
)

# %%
# Prediction on the test dataframe
test_df = test_df[n_features]
prediction = grid_dt.predict(test_df)
log.info(f"{prediction}")

# %% [markdown]
# ## Gradient Boosting Classifier
# * HistGradientBoostingClassifier used as it is faster than GradientBoostingClassifier
# * All the features are used
# * Remove parameters not accepted by HistGradientBoostingClassifier

# %%
# **Tunning parameters with gridsearch**
# Remove parameters not accepted by HistGradientBoostingClassifier

# Set default parameters based on classifier type
gbm_classifier = HistGBClassifierGridSearch()
gbm_classifier.run(x_train, y_train)
model_fit(
    gbm_classifier.model,
    x_train_sl,
    y_train_sl,
    n_features,
    roc=True,
    printFeatureImportance=True,
)
# %% [markdown]
# HistGradientBoostingClassifier
# 2025-01-04 16:58:12,969 - INFO - Accuracy : 0.5785

# 2025-01-04 16:58:19,405 - INFO - Cross Validation Accuracy: 0.53241 (+/- 0.00)

# TA indicators simple v1

# 2025-01-06 10:13:11,081 - INFO - Accuracy : 0.6215

# 2025-01-06 10:13:24,791 - INFO - Cross Validation Accuracy: 0.56680 (+/- 0.00)

# Ta indicators parametrzied v2 (overfitting)

# 2025-01-07 19:19:10,628 - INFO - Accuracy : 0.6517

# 2025-01-07 19:19:15,360 - INFO - Cross Validation Accuracy: 0.51616 (+/- 0.00)

# GradientBoostingClassifier

# 2025-01-04 17:34:54,242 - INFO - Accuracy : 0.6225

# 2025-01-04 17:51:19,832 - INFO - Cross Validation Accuracy: 0.53658

# Ta indicators v3 (feature selection)

# 2025-01-08 10:33:25,881 - INFO - Accuracy : 0.5866

# 2025-01-08 10:33:34,478 - INFO - Cross Validation Accuracy: 0.52654 (+/- 0.00)

# %% [markdown]
# ### Parameters tuning :
# Steps:
# 1. n_estimators (30-80): Number of boosting stages to perform
# 2. max_depth (5-15): Maximum depth of individual trees
# 3. min_samples_split (400-1000): Minimum samples required to split internal node
# 4. min_samples_leaf (40): Minimum samples required at leaf node
# 5. max_features (7-20): Number of features to consider for best split
# 6. Fixed parameters:
#    - learning_rate=0.1
#    - subsample=0.8
#    - random_state=10


# %% [markdown]
# ### Run sequential parameter tuning

# %%
n_estimators_result = gbm_classifier.tune_n_estimators(x_train, y_train)

tree_params_result = gbm_classifier.tune_tree_params(
    x_train, y_train, {**n_estimators_result.best_params_}
)

# %% [markdown]
# Results from previous run (HistGradientBoostingClassifier):

# Best: 0.5368177574059928 using {'max_depth': 9, 'min_samples_leaf': 50}

# Best (simple TA): 0.578087598675834 using {'max_depth': 11, 'min_samples_leaf': 50}

# Results from previous run (GradientBoostingClassifier):

# Best: 0.540790 using {'max_depth': 15, 'min_samples_split': 400}

# %%
leaf_params_result = gbm_classifier.tune_leaf_params(
    x_train,
    y_train,
    {**n_estimators_result.best_params_, **tree_params_result.best_params_},
)
# %% [markdown]
# Best (simple TA): 0.578087598675834 using {'l2_regularization': 0.001}
# %% [markdown]
# #### Use the model with best parameters

# %%
max_features_result = gbm_classifier.tune_max_features(
    x_train,
    y_train,
    {
        **n_estimators_result.best_params_,
        **tree_params_result.best_params_,
        **leaf_params_result.best_params_,
    },
)
# %% [markdown]
# Best (simple TA): 0.578087598675834 using {'max_bins': 255}

# %%
model_fit(
    max_features_result.best_estimator_,
    x_train,
    y_train,
    features,
    roc=True,
    printFeatureImportance=True,
)


# %% [markdown]
# ## Neural Network
# * Standardization of the data
# * Initialize model, loss function and optimizer
# Convert data to tensors
# %% [Markdown]
# #### Decisions about architecture:
# * Tanh has been used in the first two layers because it outputs values from -1 to 1, which can be beneficial for centered data.
# * ReLU is used in the later layers since it helps to mitigate the vanishing gradient problem and enhances computational efficiency.
# * Sigmoid in the final layer is used for binary classification tasks, where you want to output a probability between 0 and 1.
# * Dropout is used to prevent overfitting.
# * The use of different activation functions (Tanh, ReLU, Sigmoid) introduces non-linearity into the model.
# * Tanh is typically used in the first two layers because it outputs values between -1 and 1, which can be beneficial for centered data.
# * ReLU (Rectified Linear Unit) is generally used in the later layers since it helps to mitigate the vanishing gradient problem and enhances computational efficiency.

# Layers:
# * 5 layers
# * 100 neurons in the first layer (Tanh)
# * 50 neurons in the second layer (Tanh) with dropout (0.33)
# * 150 neurons in the third layer (ReLU)
# * 50 neurons in the fourth layer (ReLU)
# * 35 neurons in the fifth layer (Sigmoid)

# %%
# Preparing standardization and normalization
scaler = StandardScaler()
scaler.fit(x_train[features])
X_train = scaler.fit_transform(x_train[features])
scaler.fit(x_test[features])
X_test = scaler.fit_transform(x_test[features])
nn_model = Net(len(features))
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(nn_model.parameters())

# Convert data to tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
Y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

train_neural_network(train_df, parameters)
# %% [markdown]
# Training part

# %%
n_epochs = 250
batch_size = 5000

for epoch in range(n_epochs):
    for i in range(0, len(train_df), batch_size):
        batch_X = X_train_tensor[i : i + batch_size]
        batch_y = y_train_tensor[i : i + batch_size]

        # Forward pass
        outputs = nn_model(batch_X)
        loss = criterion(outputs, batch_y.view(-1, 1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Evaluate the model
nn_model.eval()
with torch.no_grad():
    outputs = nn_model(X_test_tensor)
    # Convert probabilities to binary predictions using 0.5 threshold
    y_predict = (outputs >= 0.5).squeeze().numpy()

print(classification_report(Y_test_tensor, y_predict, digits=5))


# %% [markdown]
# Model comparison plot
# Convert model results to DataFrame for plotting
# %%
model_results = {
    "Decision Tree (Base)": base_dt.score(x_test, y_test),
    "Decision Tree (Tuned)": grid_dt.score(x_test_sl, y_test),
    "GB (n_estimators)": n_estimators_result.best_estimator_.score(x_test, y_test),
    "GB (+ tree params)": tree_params_result.best_estimator_.score(x_test, y_test),
    "GB (+ leaf params)": leaf_params_result.best_estimator_.score(x_test, y_test),
    "GB (+ max features)": max_features_result.best_estimator_.score(x_test, y_test),
    "Neural Network": accuracy_score(Y_test_tensor, y_predict),
}

# %% [markdown]
# Create dictionary of model results including stepping stone models

# %%
plot_model_accuracy(model_results)

# %% [markdown]
# ### Simulation
# Simulation of actual returns based on predictions coming from models.
# It must be stated that there are a few assumptions:
# * Zero transaction costs
# * No slippage (no market impact)
# * Unconstrained shorting
# Takeaways:
# - If, on average, we are right >50% of the time, and the sizing of the trade is constant,
# then we can expect to make money. Hence, the line with some positive drift is expected.
# - The slope of this line depends on the accuracy of the model. The higher the accuracy, the higher the slope.
# - As previously stated, this is a very simplified model and does not take into account many factors
# that could affect the real performance of the strategy.
# - The scope of this project is limited, i.e. to generate a buy/sell signal that in real application
# is just a small part of actual trading decision.

# %%
# Run simulation
simulate_strategy(Y_test_tensor, y_predict, n_simulations=1, n_days=100)

# plot simulation
# Create dataframe for plotting


# %% [markdown]
# ### Key Findings:
#
# - All models are compared against the benchmark accuracy of 51.31%
# - The tuned Gradient Boosting model significantly outperforms other models
# - Hyperparameter tuning improved both Decision Tree and Gradient Boosting performance
# - Gradient Boosting shows superior performance compared to Decision Trees, which is expected
# - HistGradientBoostingClassifier is much faster than GradientBoostingClassifier without
#   much compromising the performance
# - Further improvement in out-of-sample performance is possible by both
# better feature engineering and further hyperparameter tuning
#    * More technical indicators could be introduced (e.g. ROC, Golden Cross, etc.)
#    * More variables based on the categorical variables (which are dropped as of now)
# could bring in some value
# - Even simple technical indicators can improve the performance of the model more than right choice of hyperparameters
# - Using too many features caused extreme overfitting (expected)
# - Incorrectly calculated technical indicators had some predictive power (unexpected)
# - Neural network-based model is not able to beat the benchmark accuracy of 51.31% (NN was only marginally better)
# - More sophisticated MLOps tools would be useful to track the performance of the model and the changes in the code
