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

# %%
import logging as log
import re
from itertools import compress
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import Markdown as md
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from src.GBClassifierGridSearch import HistGBClassifierGridSearch
from src.nn import Net
from src.ta_indicators import (
    calculate_all_ta_indicators,
    filter_infinity_values,
    remove_duplicated_columns,
)
from src.utils import (
    ID_COLS,
    feature_importance,
    load_data,
    model_fit,
    plot_correlation_matrix,
    plot_model_accuracy,
    plot_nan_percentages,
    plot_ret_and_vol,
    preprocess_data,
    simulate_strategy,
)

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


IN_COLAB = setup_colab_environment()

# Configure logging to stdout
log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[log.StreamHandler()],
)
target = "RET"
kfold = 2


# %% [markdown]
# ## Loading data
# %%
# Set data directory based on environment

data_dir = Path("/content/data") if IN_COLAB else Path("./data")
x_train_path: Path = data_dir / "x_train.csv"
y_train_path: Path = data_dir / "y_train.csv"
x_test_path: Path = data_dir / "x_test.csv"
train_df, test_df = load_data(x_train_path, y_train_path, x_test_path)

#  %% [markdown]
# #### Problem visualization
# %%
# Plot returns and volume
plot_ret_and_vol(train_df, 28)

# %% [markdown]
# #### Head of the data
# %%
test_df.head()
# %% [markdown]
# #### Info about the dataset
# %%
print("Training Dataset Info:")
train_df.info()
print("\nTest Dataset Info:")
test_df.info()

# %% [markdown]
# #### Plot nan percentages across categorical var

# %%
plot_nan_percentages(train_df)

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
return_features = [f"RET_{day}" for day in range(1, 6)]
return_to_drop = train_df[
    (train_df[return_features].isna().sum(axis=1) / train_df[return_features].shape[1])
    >= 1
][return_features]
return_to_drop
train_df.drop(index=return_to_drop.index, inplace=True)

# %% [markdown]
# #### Preprocessing data
# * Dropping rows with NAs
# * If arg set to true, removing ID columns
# * RET is encoded from bool to binary

# %%
train_df, test_df = preprocess_data(
    train_df,
    test_df,
    remove_id_cols=False,
    # sample_n=50000,
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
    f"Class imbalance: {train_df['RET'].value_counts(normalize=True)[0] * 100:.2f}%"
    + f" {train_df['RET'].value_counts(normalize=True)[1] * 100:.2f}%"
)


# %% [markdown]
# #### Plot correlation matrix
# %%
plot_correlation_matrix(train_df)

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
new_features = []

# Conditional aggregated features
shifts = [1, 2, 3, 4, 5]  # Choose some different shifts
statistics = ["median", "std"]  # the type of stat
gb_features = [
    ["DATE", "INDUSTRY"],
    ["DATE", "INDUSTRY_GROUP"],
    ["DATE", "SECTOR"],
    ["DATE", "SUB_INDUSTRY"],
]
target_feature = "RET"
# Create a name by joining the last element of each gb_feature list
for gb_feature in gb_features:
    for shift in shifts:
        for stat in statistics:
            name = f"{target_feature}_{shift}_{gb_feature[-1]}_{stat}"
            feat = f"{target_feature}_{shift}"
            new_features.append(name)
            for data in [train_df, test_df]:
                data[name] = data.groupby(gb_feature)[feat].transform(stat)


# %%
ta_indicators_path = Path("./data/ta_indicators.pkl")
test_ta_indicators_path = Path("./data/test_ta_indicators.pkl")

if ta_indicators_path.exists() and test_ta_indicators_path.exists():
    log.info("Loading pre-calculated technical indicators from pickle files")
    ta_indicators_df = pd.read_pickle(ta_indicators_path)
    test_ta_indicators_df = pd.read_pickle(test_ta_indicators_path)
else:
    log.info("Calculating technical indicators for train and test data")
    ta_indicators_df = calculate_all_ta_indicators(train_df)
    test_ta_indicators_df = calculate_all_ta_indicators(test_df)
    # Save calculated indicators to pickle files
    ta_indicators_df.to_pickle(ta_indicators_path)
    test_ta_indicators_df.to_pickle(test_ta_indicators_path)
    log.info(
        f"Saved technical indicators to {ta_indicators_path} and {test_ta_indicators_path}"
    )

# %% [markdown]
# #### Concatenate the technical indicators to the train and test data
# %%
train_df = pd.concat([train_df, ta_indicators_df], axis=1)
test_df = pd.concat([test_df, test_ta_indicators_df], axis=1)
# %%
# Save/load processed training data
train_df_path = Path("./data/processed_train_df.pkl")
test_df_path = Path("./data/processed_test_df.pkl")
if train_df_path.exists() and test_df_path.exists():
    log.info(f"Loading processed training data from {train_df_path}")
    train_df = pd.read_pickle(train_df_path)
    test_df = pd.read_pickle(test_df_path)
else:
    log.info(f"Saving processed training data to {train_df_path}")
    train_df.to_pickle(train_df_path)
    test_df.to_pickle(test_df_path)

# %% [markdown]
# #### Columns to drop
# They could bring in some predictive power, but we don't want to use them in this case
# as the scope is limited for this project
# ['ID', 'STOCK', 'DATE', 'INDUSTRY', 'INDUSTRY_GROUP', 'SECTOR', 'SUB_INDUSTRY']
# %%
train_df.drop(columns=ID_COLS, inplace=True, errors="ignore")
test_df.drop(columns=ID_COLS, inplace=True, errors="ignore")

# %% [markdown]
# Assumption is that probably some technical indicators are not useful for the prediction.
# For instance SMA(10), SMA(11) etc. dont give any information in the context of RET.
# It's an arbitrary choice, but we want to keep the number of features low

# %% [markdown]
# #### Further data wrangling
# %%
cols_to_drop = [
    col
    for col in train_df.columns
    if re.search(
        r"\D(?:([1-2]{1}[0-9])|([8-9]{1})\_)",
        str(col),
    )
    and not col.startswith(("RET", "VOLUME"))  # don't drop RET and VOLUME
]

train_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
# %%
# Define features before using them
features = [col for col in train_df.columns if col != target]

# %% Filter out infinity values
train_df, test_df, features = filter_infinity_values(
    train_df, test_df, features, target
)

# Remove duplicated columns
train_df, test_df, features = remove_duplicated_columns(train_df, test_df, features)

# %%
# Features
# Default selection
features = train_df.columns.drop(target).tolist()

# %% [markdown]
# ## ML DecisionTreeClassifier

# %%
# Train and test set splitting
x_train, x_test, y_train, y_test = train_test_split(
    train_df[features],
    train_df["RET"],
    test_size=0.25,
    random_state=0,
)


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
# Initialize model, loss function and optimizer
# Convert data to tensors
# %%
# preparing standardization and normalization
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
