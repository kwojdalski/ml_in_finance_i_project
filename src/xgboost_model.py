# %%
import logging
import pickle

import pandas as pd

from src.qrt_stock_returns.pipelines.reporting.nodes import plot_outliers_analysis
from src.qrt_stock_returns.utils import conf_params, get_node_output, run_pipeline_node

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info("Starting XGBoost")

# %% [markdown]
# ### Loading Up Kedro Config
# Grab all our parameters from the config file
# * This has elements like our target variable and k-fold settings
# %%
target = conf_params["model_options"]["target"]
kfold = conf_params["model_options"]["kfold"]
# %%


# %%
out10 = get_node_output("handle_outliers_node")

# %%
# %% [markdown]
# ## Split Data into Training and Test Sets
# %%
out11 = run_pipeline_node(
    "split_data_node",
    {
        "train_df_winsorized": out10["train_df_winsorized"],
        "params:model_options": conf_params["model_options"],
    },
)

# %%
X_train = out11["X_train"]
X_test = out11["X_test"]
y_train = out11["y_train"]
y_test = out11["y_test"]


# Plot outliers for training data
plot_outliers_analysis(X_train)


# %%
out10 = run_pipeline_node(
    "train_xgboost_node",
    {
        "X_train": X_train,
        "y_train": y_train,
        "parameters": conf_params["model_options"],
    },
)

# %%
# get_node_outputs(
#     pipelines["data_science"].nodes[
# ## Load Models
# %%
with open("data/06_models/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# %%
# Plot feature importance
importance_df = pd.DataFrame(
    {
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }
)

# Print importance scores
log.info("\nTop 20 most important features:")
for idx, row in importance_df[:20].iterrows():
    log.info(f"{row['feature']}: {row['importance']:.4f}")


# %%
run_pipeline_node(
    "evaluate_xgboost_node",
    {
        "xgboost_model": model,
        "X_test": X_test,
        "y_test": y_test,
        "parameters": conf_params["model_options"],
    },
)

# %%
out11 = run_pipeline_node(
    "generate_predictions_node",
    {
        # "xgboost_model": out10["xgboost_model"],
        "xgboost_model": model,
        # "gru_model": out10["gru_model"],
        "test_df_winsorized": out10["test_df_winsorized"],
    },
)

# %%
out11["predictions"].index = out11["predictions"].index.astype(int)
# %%
