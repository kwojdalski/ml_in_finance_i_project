# %%
import logging

import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score

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
out10 = get_node_output("handle_outliers_node")
out9 = get_node_output("transform_volret_features_node")


plot_outliers_analysis(out10["train_df_winsorized"])


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


# %%
import pickle

with open("data/06_models/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)


# %%
def train_xgboost(X_train, y_train, parameters: dict, timeout: int = 60):
    """Train XGBoost model using Optuna for hyperparameter optimization.

    Args:
        X_train: Training features
        y_train: Training labels
        parameters: Dictionary of model parameters
        timeout: Maximum optimization time in seconds, defaults to 60

    Returns:
        Trained XGBoost model with optimized hyperparameters
    """

    def objective(trial, X, y):
        param = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": 1,
            "verbosity": 2,
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            # "learning_rate": 0.3,
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "gamma": trial.suggest_float("gamma", 0, 0.2),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 2),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 0.5),
            "random_state": 42,
        }

        model = xgb.XGBClassifier(
            **param,
            tree_method="hist",
        )
        model.fit(
            X,
            y,
            eval_set=[(X, y)],
            verbose=False,
        )
        scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="accuracy",
            # error_score="raise"
        )
        return scores.mean()

    # Create and run Optuna study
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="ensemble_xgb",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    import time

    tic = time.time()
    while time.time() - tic < timeout:  # noqa
        study.optimize(
            lambda trial: objective(trial, X_train, y_train),
            n_trials=1,
        )

    # Get best parameters and train final model
    best_params = study.best_params
    best_model = xgb.XGBClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

    return best_model


# %%
# Train model and get parameters
model = train_xgboost(X_train, y_train, conf_params["model_options"], timeout=1)
print("\nModel Parameters:")
for param, value in model.get_params().items():
    print(f"{param}: {value}")
    # Fit model and check accuracy
model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
train_accuracy = model.score(X_train, y_train)
print(f"Training accuracy: {train_accuracy:.4f}")

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
out11["predictions"].to_csv("data/07_model_output/submission5.csv")
# Compare submissions 2 and 3
import pandas as pd

submission = pd.read_csv("data/07_model_output/submission.csv", index_col=0)
submission2 = pd.read_csv("data/07_model_output/submission2.csv", index_col=0)
submission3 = pd.read_csv("data/07_model_output/submission3.csv", index_col=0)

# Calculate differences
differences = abs(submission3 - submission)
sum(differences["pred"])
print("\nSubmission Comparison:")
print(f"Number of different predictions: {(differences != 0).sum().values[0]}")
print(f"Mean absolute difference: {differences.abs().mean().values[0]:.4f}")
print(f"Max absolute difference: {differences.abs().max().values[0]:.4f}")

# Show correlation
correlation = submission2.corrwith(submission3)
print(f"Correlation between submissions: {correlation.values[0]:.4f}")

# %%
