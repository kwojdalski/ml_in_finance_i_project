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
# ### Environment setup
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

# %% [markdown]
# ## Importing libraries
# %%
import logging as log
import sys
from itertools import compress

import kedro.ipython
import torch
from sklearn.metrics import classification_report
from utils import model_fit

from src.ml_in_finance_i_project.pipelines.reporting.nodes import feature_importance
from src.ml_in_finance_i_project.utils import get_node_idx, get_node_outputs


# %% [markdown]
# #### Run pipeline node definition. This one must be evaluated within the notebook
# %%
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


# %% [markdown]
# #### Kedro configuration loading
# %%
# load the configuration file from kedro and load into conf_params
conf_params = context.config_loader.get("parameters")
target = conf_params["model_options"]["target"]
kfold = conf_params["model_options"]["kfold"]


# %% [markdown]
# ### Loading data from data_processing pipeline
# %%
out9 = get_node_outputs(
    pipelines["data_processing"].nodes[
        get_node_idx(pipelines["data_processing"], "remove_duplicated_columns_node")
    ],
    catalog,
)
# %% [markdown]
# %%
# Drop target column if existsand convert to list
features = out9["train_df_rm_duplicates"].columns.drop(target).tolist()

# %% [markdown]
# ## Train and test set splitting
# %%
out10 = run_pipeline_node(
    "data_science",
    "split_data_node",
    {
        "train_df_rm_duplicates": out9["train_df_rm_duplicates"],
        "params:model_options": conf_params["model_options"],
    },
)
# %%
X_train = out10["X_train"].head(1000)
X_test = out10["X_test"]
y_train = out10["y_train"].head(1000)
y_test = out10["y_test"]
nan_mask = X_train.isna().any(axis=1)
X_train = X_train[~nan_mask]
y_train = y_train[~nan_mask]

# %% [markdown]
# ### Decison tree baseline model
# %%
base_dt = run_pipeline_node(
    "data_science",
    "train_decision_tree_node",
    {
        "X_train": X_train,
        "y_train": y_train,
    },
)["dt_model"]

# %% [markdown]
# ### Fit the base model
# `model_fit()` function a model, makes predictions, and evaluates performance
# using confusion matrix, accuracy score,
# cross-validation, ROC curve and feature importance analysis.

# %%
log.info(f"Accuracy on test set: {base_dt.score(X_test, y_test):.3f}")

# %% [markdown]
# ### Tunning Decision tree model with Gridsearch
# %%
tuned_dt = run_pipeline_node(
    "data_science",
    "tune_decision_tree_node",
    {
        "X_train": X_train,
        "y_train": y_train,
        "parameters": conf_params["model_options"],
    },
)["tuned_dt_model"]
# %% [markdown]
# ### Feature selection for the tuned model
# Based on feature importances
# %%
# ## visualize feature importance
threshold = conf_params["model_options"]["feature_importance_threshold"]
feature_importance(tuned_dt["best_model"], features, threshold)

# %% [markdown]
# **filtering out features with less than 1% of feature importance**
# %%
n_features = list(
    compress(
        features,
        tuned_dt["best_model"].feature_importances_ >= threshold,
    )
)

# %% [markdown]
# #### New sets with only the selected features
# %%
X_train_sl = X_train.loc[:, X_train[n_features].columns]
X_test_sl = X_test.loc[:, X_test[n_features].columns]

# %% [markdown]
# ### Decision tree tuned model
# %%
grid_dt = run_pipeline_node(
    "data_science",
    "tune_decision_tree_node",
    {
        "X_train": X_train_sl,
        "y_train": y_train,
        "parameters": conf_params["model_options"],
    },
)

# %%
log.info("Fitting with train set")
model_fit(
    grid_dt["tuned_dt_model"]["best_model"],
    X_train_sl,
    y_train,
    n_features,
    printFeatureImportance=True,
)
tuned_dt = run_pipeline_node(
    "data_science",
    "tune_decision_tree_node",
    {
        "X_train": X_train_sl,
        "y_train": y_train,
        "parameters": conf_params["model_options"],
    },
)
# %%
log.info("Fitting with test set")
model_fit(
    tuned_dt["tuned_dt_model"]["best_model"],
    X_test_sl,
    y_test,
    n_features,
    printFeatureImportance=False,
    roc=True,
)

# %% [markdown]
# #### Prediction on the test dataframe
# %%
X_test_sl = X_test_sl[n_features]
prediction = grid_dt["tuned_dt_model"]["best_model"].predict(X_test_sl)
log.info(f"{prediction}")

# %% [markdown]
# ## Gradient Boosting Classifier
# * HistGradientBoostingClassifier used as it is faster than GradientBoostingClassifier
# * All the features are used
# * Remove parameters not accepted by HistGradientBoostingClassifier

# %% [markdown]
# **Tunning parameters with gridsearch**
# Remove parameters not accepted by HistGradientBoostingClassifier

# %%
# Set default parameters based on classifier type
gbm_classifier = run_pipeline_node(
    "data_science",
    "train_gradient_boosting_node",
    {
        "X_train": X_train.head(1000),
        "y_train": y_train.head(1000),
        "parameters": conf_params["model_options"],
    },
)["gb_model"]
# %%
model_fit(
    gbm_classifier.model,
    X_train_sl,
    y_train,
    n_features,
    roc=True,
    printFeatureImportance=True,
)

# %% [markdown]
# ### Parameters tuning:
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
n_estimators_result = gbm_classifier.tune_n_estimators(X_train, y_train)

tree_params_result = gbm_classifier.tune_tree_params(
    X_train, y_train, {**n_estimators_result.best_params_}
)

# %% [markdown]
# Results from previous run (HistGradientBoostingClassifier):
# Best: 0.5368177574059928 using {'max_depth': 9, 'min_samples_leaf': 50}

# %%
leaf_params_result = gbm_classifier.tune_leaf_params(
    X_train,
    y_train,
    {**n_estimators_result.best_params_, **tree_params_result.best_params_},
)
# %% [markdown]
# Best (simple TA): 0.578087598675834 using {'l2_regularization': 0.001}
# #### Use the model with best parameters

# %%
max_features_result = gbm_classifier.tune_max_features(
    X_train,
    y_train,
    {
        **n_estimators_result.best_params_,
        **tree_params_result.best_params_,
        **leaf_params_result.best_params_,
    },
)

# %%
model_fit(
    max_features_result.best_estimator_,
    X_train,
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
# %% [markdown]
# #### Decisions about architecture:
# * Tanh has been used in the first two layers because it outputs values from -1 to 1, which can be beneficial
# for centered data.
# * ReLU is used in the later layers since it helps to mitigate the vanishing gradient problem
# and enhances computational efficiency.
# * Sigmoid in the final layer is used for binary classification tasks,
# where you want to output a probability between 0 and 1.
# * Dropout is used to prevent overfitting.
# * The use of different activation functions (Tanh, ReLU, Sigmoid) introduces non-linearity into the model.
# * Tanh is typically used in the first two layers because it outputs values between -1 and 1,
# which can be beneficial for centered data.
# * ReLU (Rectified Linear Unit) is generally used in the later layers since it helps to mitigate
# the vanishing gradient problem and enhances computational efficiency.

# Layers:
# * **5** layers
# * **100** neurons in the first layer (Tanh)
# * **50** neurons in the second layer (Tanh) with dropout (0.33)
# * **150** neurons in the third layer (ReLU)
# * **50** neurons in the fourth layer (ReLU)
# * **35** neurons in the fifth layer (Sigmoid)
# %% [markdown]
# #### Neural Network
# Preparing standardization and normalization
# %%
X_test_tensor = torch.FloatTensor(X_test.values)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)
nn_model = run_pipeline_node(
    "data_science",
    "train_neural_network_node",
    {
        "X_train": X_train,
        "y_train": y_train,
        "parameters": conf_params["model_options"],
    },
)["nn_model"]

with torch.no_grad():
    outputs = nn_model(X_test_tensor)
    # Convert probabilities to binary predictions using 0.5 threshold
    y_predict = (outputs >= 0.5).squeeze().numpy()

print(classification_report(y_test_tensor, y_predict, digits=5))


# %% [markdown]
# Model comparison plot
# Convert model results to DataFrame for plotting
# %%
model_results = run_pipeline_node(
    "reporting",
    "aggregate_model_results_node",
    {
        "base_dt": base_dt,
        "grid_dt": grid_dt["tuned_dt_model"]["best_model"],
        "n_estimators_result": n_estimators_result,
        "tree_params_result": tree_params_result,
        "leaf_params_result": leaf_params_result,
        "max_features_result": max_features_result,
        "nn_model": nn_model,
        "X_test": X_test,
        "y_test": y_test,
        "X_test_sl": X_test_sl,
    },
)["model_results_dict"]

# %% [markdown]
# Create dictionary of model results including stepping stone models

# %%
run_pipeline_node(
    "reporting",
    "plot_model_accuracy_node",
    {
        "model_results": model_results,
    },
)["model_accuracy_plot"]
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

# %% [markdown]
# Run simulation
run_pipeline_node(
    "reporting",
    "simulate_strategy_node",
    {
        "y_test": y_test,
        "y_predict": y_predict,
        "params:n_simulations": 1,
        "params:n_days": 100,
    },
)["strategy_simulation_plot"]


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
