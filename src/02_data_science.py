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
# ## Stock Market Movement Prediction - Data Science
# %% [markdown]
# ### Environment Setup
# %%
import sys
from pathlib import Path

try:
    # If we're in VSCode, find the src path this way
    path = Path(__file__).parent.parent
    path = path / "src"
except NameError:
    # If we're in Jupyter, find the path this way instead
    path = Path().absolute().parent

sys.path.append(str(path))

import kedro.ipython
from kedro.ipython import get_ipython

kedro.ipython.load_ipython_extension(get_ipython())

# %% [markdown]
# ## Library Imports
# %%
import logging as log
import sys

import kedro.ipython
import torch
from sklearn.metrics import classification_report

from src.ml_in_finance_i_project.pipelines.data_science.nodes import model_fit
from src.ml_in_finance_i_project.utils import get_node_idx, get_node_outputs


# %% [markdown]
# #### Helper Function
# - Run pipeline nodes right here in the notebook
# %%
def run_pipeline_node(pipeline_name: str, node_name: str, inputs: dict):
    """Run a specific node from a pipeline."""
    from kedro.framework.session import KedroSession

    with KedroSession.create() as session:
        context = session.load_context()
        pipeline = context.pipelines[pipeline_name]
        node = [n for n in pipeline.nodes if n.name == node_name][0]
        return node.run(inputs)


# %% [markdown]
# ### Loading Up Kedro Config
# Grab all our parameters from the config file
# * This has elements like our target variable and k-fold settings
# %%
conf_params = context.config_loader.get("parameters")
target = conf_params["model_options"]["target"]
kfold = conf_params["model_options"]["kfold"]


# %% [markdown]
# ### Getting Data from the Data Processing Pipeline
# %%
out9 = get_node_outputs(
    pipelines["data_processing"].nodes[
        get_node_idx(pipelines["data_processing"], "remove_duplicates_and_nans_node")
    ],
    catalog,
)

# %% [markdown]
# ## Split Data into Training and Test Sets
# %%
out10 = run_pipeline_node(
    "data_science",
    "split_data_node",
    {
        "train_df_clean": out9["train_df_clean"],
        "params:model_options": conf_params["model_options"],
    },
)
# %%
X_train = out10["X_train"]
X_test = out10["X_test"]
y_train = out10["y_train"]
y_test = out10["y_test"]
# %% [markdown]
# ## Basic Decision Tree Model
# %%
base_dt = run_pipeline_node(
    "data_science",
    "train_decision_tree_node",
    {
        "X_train": X_train,
        "y_train": y_train,
        "parameters": conf_params,
    },
)["base_dt"]

# %% [markdown]
# ### Fitting Baseline Model
# The model_fit() function does the following:
# - Trains the model and makes predictions
# - Shows us how well it's doing with confusion matrix and accuracy
# - Does cross-validation to check if we're overfitting
# - Draws ROC curves to visualize performance
# - Shows which features are most important

# %%
log.info(f"Accuracy on test set: {base_dt['model'].score(X_test, y_test):.3f}")

# %% [markdown]
# ### Tuning Decision Tree Model with GridSearch

# %%
tuned_dt = run_pipeline_node(
    "data_science",
    "tune_decision_tree_node",
    {
        "X_train": X_train,
        "y_train": y_train,
        "parameters": conf_params,
    },
)["grid_dt"]

# %% [markdown]
# ### Feature Selection for the Tuned Model
# Based on feature importances
# %% [markdown]
# ## Visualize Feature Importance
# %%
run_pipeline_node(
    "reporting",
    "plot_feature_importance_node",
    {
        "grid_dt": tuned_dt,
        "X_train": X_train,
        "params:feature_importance_threshold": conf_params["model_options"][
            "feature_importance_threshold"
        ],
    },
)["feature_importance_plot"]

# %% [markdown]
# **Filtering Out Features with Less Than 1% of Feature Importance**
# %%
out12 = run_pipeline_node(
    "data_science",
    "select_important_features_node",
    {
        "X_train": X_train,
        "X_test": X_test,
        "grid_dt": tuned_dt,
        "parameters": conf_params,
    },
)
X_train_selected = out12["X_train_selected"]
X_test_selected = out12["X_test_selected"]
important_features = out12["important_features"]

# %% [markdown]
# ## Decision Tree Tuned Model
# New sets with only the selected features
# %%
grid_dt = run_pipeline_node(
    "data_science",
    "tune_decision_tree_selected_node",
    {
        "X_train_selected": X_train_selected,
        "y_train": y_train,
        "parameters": conf_params,
    },
)["grid_dt_selected"]

# %%
log.info("Fitting with train set")
model_fit(
    grid_dt["model"],
    X_train_selected,
    y_train,
    X_train_selected.columns,
)

# %%
log.info("Fitting with test set")
model_fit(
    tuned_dt["model"],
    X_test_selected,
    y_test,
    important_features,
    roc=True,
)

# %% [markdown]
# ### Prediction on the Test Dataframe
# %%

prediction = grid_dt["model"].predict(X_test_selected)
log.info(f"{prediction}")

# %% [markdown]
# ## Gradient Boosting Classifier
# * `HistGradientBoostingClassifier` used as it is faster than `GradientBoostingClassifier`
# * All the features are used
# * Remove parameters not accepted by `HistGradientBoostingClassifier`

# %% [markdown]
# ### Tuning Parameters with GridSearch

# %%

gbm_classifier = run_pipeline_node(
    "data_science",
    "train_gradient_boosting_node",
    {
        "X_train_selected": X_train_selected,
        "y_train": y_train,
        "parameters": conf_params,
    },
)["base_gb"]
# %%
model_fit(
    gbm_classifier["model"].model,
    X_train_selected,
    y_train,
    important_features,
    roc=True,
)

# %% [markdown]
# ### Parameters Tuning
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
# ### Run Sequential Parameter Tuning

# %%
tuned_gb = run_pipeline_node(
    "data_science",
    "tune_gradient_boosting_node",
    {
        "base_gb": gbm_classifier["model"],
        "X_train_selected": X_train_selected,
        "y_train": y_train,
        "parameters": conf_params,
    },
)["tuned_gb"]

# %%
n_estimators_result = tuned_gb["n_estimators_result"]
tree_params_result = tuned_gb["tree_params_result"]

# %% [markdown]
# Results from Previous Run (HistGradientBoostingClassifier):
# Best: 0.5368177574059928 using {'max_depth': 9, 'min_samples_leaf': 50}

# %%
leaf_params_result = tuned_gb["leaf_params_result"]

# %% [markdown]
# Best (Simple TA): 0.578087598675834 using {'l2_regularization': 0.001}
# #### Use the Model with Best Parameters

# %%
max_features_result = tuned_gb["max_features_result"]
# %%
model_fit(
    max_features_result.best_estimator_,
    X_train_selected,
    y_train,
    X_train_selected.columns,
    roc=True,
)

# %% [markdown]
# ## Neural Network
# * Standardization of the data
# * Initialize model, loss function and optimizer
# Convert data to tensors
# %% [markdown]
# ### Decisions About Architecture
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
#
# Layers:
# * **5** layers
# * **100** neurons in the first layer (Tanh)
# * **50** neurons in the second layer (Tanh) with dropout (0.33)
# * **150** neurons in the third layer (ReLU)
# * **50** neurons in the fourth layer (ReLU)
# * **35** neurons in the fifth layer (Sigmoid)
# %% [markdown]
# ### Running the Model
# %%
nn_model = run_pipeline_node(
    "data_science",
    "train_neural_network_node",
    {
        "X_train": X_train,
        "y_train": y_train,
        "parameters": conf_params,
    },
)["nn_model"]

X_test_tensor = torch.FloatTensor(X_test.values)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)
with torch.no_grad():
    outputs = nn_model["model"](X_test_tensor)
    # Convert probabilities to binary predictions using 0.5 threshold
    y_predict = (outputs >= 0.5).squeeze().numpy()

# y_predict.to_csv("data/07_model_output/y_predict.csv", index=False)
print(classification_report(y_test_tensor, y_predict, digits=5))
# %% [markdown]
# ## Model Comparison Plot
# Convert model results to DataFrame for plotting
# %%
model_results = run_pipeline_node(
    "reporting",
    "aggregate_model_results_node",
    {
        "base_dt": base_dt,
        "grid_dt": grid_dt,
        "tuned_gb": tuned_gb,
        "nn_model": nn_model,
        "X_test": X_test,
        "y_test": y_test,
    },
)["model_results"]

# %% [markdown]
# Create Dictionary of Model Results Including Stepping Stone Models

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
# Run Simulation
# run_pipeline_node(
#     "reporting",
#     "simulate_strategy_node",
#     {
#         "y_test": y_test,
#         "y_predict": y_predict,
#         "params:n_simulations": 1,
#         "params:n_days": 100,
#     },
# )["strategy_simulation_plot"]

# %% [markdown]
# max_features_result.best_estimator_.predict(out9["test_df_clean"][important_features])
# y_pred = max_features_result.best_estimator_.predict_proba(
#     out9["test_df_clean"][important_features].fillna(0)
# )[:, 1]
# sub = out9["test_df_clean"].copy()
# sub["pred"] = y_pred
# y_pred = sub["pred"].transform(lambda x: x > x.median()).values
#
# submission = pd.Series(y_pred)
# submission.index = sub.index
# submission.name = target
#
# submission.to_csv("./data/07_model_output/submission.csv", index=True, header=True)
# %% [markdown]
# ## Key Findings
#
# - All models are compared against the benchmark accuracy of 51.31%
#    + It's not a perfect benchmark as the number comes from completely unseen data while
#    models performance is evaluated on the part of the training set
# - The tuned Gradient Boosting model significantly outperforms other models
#    + But when submitted via QRT data challenge the performance wasn't great (**0.5066**)
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
