# %%
import functools
import inspect
import logging
import os
import random
import warnings
from typing import Any, Optional, Union

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.qrt_stock_returns.utils import conf_params, get_node_output, run_pipeline_node

warnings.filterwarnings("ignore")

# %%

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info("Starting XGBoost")

# %% [markdown]
# ### Loading Up Kedro Config
# Grab all our parameters from the config file
# * This has elements like our target variable and k-fold settings


# %%
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything()


# %%
class Config:
    DATA_DIR: str = "data"
    L1_N_TRIALS: int = 1
    L2_N_TRIALS: int = 100
    N_JOBS: int = 2
    OPTUNA_STORAGE: str = "sqlite:///db.sqlite3"

    MODELS: dict[str, str] = {
        # "lr": "Logistic Regression",
        # "ada": "AdaBoost",
        # "rf": "Random Forest",
        "xgb": "XGBoost",
        "lgb": "LightGBM",
    }

    @classmethod
    def filepath(cls, filename: str):
        return os.path.join(cls.DATA_DIR, filename)


# %%
class LRConfig:
    DEFAULT_VALUES: dict[str, Union[float, str]] = {
        "tol": 1e-4,
        "C": 1.0,
        "solver": "lbfgs",
    }
    STATIC_PARAMS: dict[str, Union[int, bool]] = {
        "max_iter": 1000,
        "verbose": False,
    }

    USE_PRUNER: bool = False

    @classmethod
    def get_fit_params(
        cls,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict[str, Any],
    ):
        return {}


class AdaConfig:
    DEFAULT_VALUES: dict[str, Union[None, int, float]] = {
        "estimator": None,
        "n_estimators": 50,
        "learning_rate": 1.0,
    }
    STATIC_PARAMS: dict[str, Any] = {}

    USE_PRUNER: bool = False

    @classmethod
    def get_fit_params(
        cls,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict[str, Any],
    ):
        return {}


# %%
class RFConfig:
    DEFAULT_VALUES: dict[str, Union[None, int, str, float, bool]] = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "ccp_alpha": 0.0,
        "max_samples": None,
    }

    STATIC_PARAMS: dict[str, int] = {
        "n_jobs": Config.N_JOBS,
    }

    USE_PRUNER: bool = False

    @classmethod
    def get_fit_params(
        cls,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict[str, Any],
    ):
        return {}


# %%
class XGBConfig:
    EVAL_METRIC: str = "logloss"

    DEFAULT_VALUES: dict[str, Union[int, float, str, None]] = {
        "max_depth": 6,
        "n_estimators": 100,
        "alpha": 0.0,
        "lambda": 1.0,
        "learning_rate": 0.3,
        "colsample_bytree": 1.0,
        "colsample_bylevel": 1.0,
        "min_child_weight": 1.0,
        "early_stopping_rounds": None,
    }

    STATIC_PARAMS: dict[str, Union[str, bool, int]] = {
        "tree_method": "hist",
        "use_label_encoder": False,
        "n_jobs": Config.N_JOBS,
        "predictor": "gpu_predictor",
        "max_bin": 1024,
        "eval_metric": EVAL_METRIC,
    }

    USE_PRUNER: bool = True

    @classmethod
    def get_fit_params(
        cls,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict[str, Any],
    ):
        return {
            "eval_set": [(X_train, y_train), (X_val, y_val)],
            "verbose": False,
        }


# %%
class LGBConfig:
    DEFAULT_VALUES: dict[str, Union[int, float]] = {
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "min_child_samples": 20,
        "subsample_for_bin": 200000,
    }

    STATIC_PARAMS: dict[str, Union[int, str]] = {
        "n_jobs": Config.N_JOBS,
        "verbose": -1,
        "objective": "binary",
    }

    USE_PRUNER: bool = True

    @classmethod
    def get_fit_params(
        cls,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict[str, Any],
    ):
        callbacks = params.get("callbacks", []) + [lgb.log_evaluation(period=0)]
        return {
            "eval_set": [(X_train, y_train), (X_val, y_val)],
            "eval_metric": "logloss",
            "callbacks": callbacks,
        }


# %%
CONFIG_MAP: dict[str, Any] = {
    "lr": LRConfig,
    "ada": AdaConfig,
    "rf": RFConfig,
    "xgb": XGBConfig,
    "lgb": LGBConfig,
}


MODEL_MAP: dict[str, Any] = {
    "lr": LogisticRegression,
    "ada": AdaBoostClassifier,
    "rf": RandomForestClassifier,
    "xgb": xgb.XGBClassifier,
    "lgb": lgb.LGBMClassifier,
}


# %%
def train(  # noqa
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model: str,
    params: dict[str, Any],
    verbose: bool = True,
):
    klass = MODEL_MAP[model]
    config = CONFIG_MAP[model]

    params = {k: v for k, v in params.items() if k in config.DEFAULT_VALUES}
    params.update({k: v for k, v in config.DEFAULT_VALUES.items() if k not in params})
    params.update(config.STATIC_PARAMS)

    clf = klass(**params)

    fit_params = config.get_fit_params(
        X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, params=params
    )

    clf.fit(
        X=X_train,
        y=y_train,
        **fit_params,
    )

    train_preds = clf.predict_proba(X_train)[:, 1]
    test_preds = clf.predict_proba(X_test)[:, 1]

    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)

    if verbose:
        log.info(f"Accuracy = {acc:.4f}")

    return train_preds, test_preds, acc


# %%
def lr_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    params = {
        "tol": trial.suggest_float("tol", 1e-6, 1e-4, log=True),
        "C": trial.suggest_float("C", 0.5, 2.0, log=True),
        "solver": trial.suggest_categorical(
            "solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
        ),
    }

    _, _, acc = train(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model="lr",
        params=params,
        verbose=False,
    )
    return acc


# %%
def adaboost_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.1, 5.0, log=True),
    }

    tune_estimator = trial.suggest_categorical("tune_estimator", [True, False])

    if tune_estimator:
        max_depth = trial.suggest_int("max_depth", 1, 50)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10, log=True)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5, log=True)
        ccp_alpha = trial.suggest_float("ccp_alpha", 0.01, 1.0, log=True)

        params["estimator"] = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
        )

    _, _, acc = train(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model="ada",
        params=params,
        verbose=False,
    )
    return acc


# %%
def rf_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1, 500),
        "max_depth": trial.suggest_int("max_depth", 1, 50),
        "min_samples_split": trial.suggest_int("min_samples_plit", 2, 10, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5, log=True),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.01, 1.0, log=True),
    }

    if params["bootstrap"] is True:
        params["max_samples"] = trial.suggest_float("max_samples", 0.01, 1.0, log=True)

    _, _, acc = train(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model="rf",
        params=params,
        verbose=False,
    )
    return acc


# %%
def xgb_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    params = {
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
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.2, 1.0),
        "early_stopping_rounds": trial.suggest_int(
            "early_stopping_rounds", 5, 20, step=5
        ),
    }

    obs_k = f"validation_1-{XGBConfig.EVAL_METRIC}"
    params["callbacks"] = [optuna.integration.XGBoostPruningCallback(trial, obs_k)]

    _, _, acc = train(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model="xgb",
        params=params,
        verbose=False,
    )
    return acc


# %%
def lgb_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 31, 100, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 100, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 50, log=True),
        "subsample_for_bin": trial.suggest_int("subsample_for_bin", 2000, 8000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }

    params["callbacks"] = [
        optuna.integration.LightGBMPruningCallback(trial, "logloss", "valid_1")
    ]

    _, _, acc = train(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model="lgb",
        params=params,
        verbose=False,
    )
    return acc


OBJECTIVE_MAP: dict[str, Any] = {
    "lr": lr_objective,
    "ada": adaboost_objective,
    "rf": rf_objective,
    "xgb": xgb_objective,
    "lgb": lgb_objective,
}


# %%
def hyperparameter_search(  # noqa
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model: str,
    n_trials: int = Config.L1_N_TRIALS,
):
    # Check if function is being called from parent frame
    frame = inspect.currentframe()
    if frame is not None:
        caller = frame.f_back
        if caller is not None:
            caller_name = caller.f_code.co_name
            log.info(f"Caller name: {caller_name}")
            if caller_name not in ["fit_level_one_models", "fit_level_two_model"]:
                raise RuntimeError(
                    "This function should only be called from fit_level_one_models or fit_level_two_model"
                )
            if caller_name == "fit_level_two_model":
                study_name = self.level_two_study_name
            else:
                study_name = f"ensemble_{model}"

    v = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective = OBJECTIVE_MAP[model]
    objective = functools.partial(
        objective, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    pruner = (
        optuna.pruners.HyperbandPruner()
        if CONFIG_MAP[model].USE_PRUNER is True
        else None
    )

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        storage=Config.OPTUNA_STORAGE,  # Specify the storage URL here.
        study_name=study_name,
        direction="maximize",
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler,
    )

    study.optimize(objective, n_trials=n_trials)

    optuna.logging.set_verbosity(v)

    return study.best_params


# %%
class Ensemble:
    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        exclude: Optional[set] = None,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.level_two_study_name = (
            f"ensemble_lr_l2_{'_'.join(Config.MODELS.keys())}"  # for now only lr
        )
        self.level_one_study_names = {
            model: f"ensemble_l1_{model}" for model in Config.MODELS.keys()
        }

        models = Config.MODELS.keys()

        if exclude is not None:
            models = models - exclude

        self.models: list[str] = list(models)

        columns = [f"{m}_preds" for m in self.models]

        self.meta_train_df = pd.DataFrame(index=X_train.index)
        self.meta_test_df = pd.DataFrame(index=X_test.index)

        # Add storage for trained models
        self.level_one_models = {}
        self.level_two_model = None
        self.level_one_params = {}
        self.level_two_params = None

    def fit_level_one_models(self, search: bool = True):
        """Train level 1 models of the ensemble.

        For each model in self.models, either search for optimal hyperparameters using Optuna
        or load existing best parameters, then train the model and store predictions in
        meta_train_df and meta_test_df.

        Args:
                search: If True, perform hyperparameter search. If False, load existing best parameters.
        """
        log.info("Training level 1 models...")

        for model in self.models:
            log.info(f"{Config.MODELS[model]}:")

            if search:
                log.info("\tFinding optimal hyperparameters using Optuna...")
                params = hyperparameter_search(
                    X_train=self.X_train,
                    X_test=self.X_test,
                    y_train=self.y_train,
                    y_test=self.y_test,
                    model=model,
                )
                log.info(f"\n\tBest params: {params}\n")
            else:
                study = optuna.load_study(
                    storage=Config.OPTUNA_STORAGE,
                    study_name=self.level_one_study_names[model],
                )
                params = study.best_params
                log.info("\tUsing existing best parameters...\n")

            log.info("\tTraining model with optimal parameters...\n")

            # Store params
            self.level_one_params[model] = params

            # Train and store the model
            klass = MODEL_MAP[model]
            config = CONFIG_MAP[model]

            params.update(config.STATIC_PARAMS)
            clf = klass(**params)

            fit_params = config.get_fit_params(
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_test,
                y_val=self.y_test,
                params=params,
            )

            clf.fit(X=self.X_train, y=self.y_train, **fit_params)
            self.level_one_models[model] = clf

            # Get predictions for meta features
            train_preds = clf.predict_proba(self.X_train)[:, 1]
            test_preds = clf.predict_proba(self.X_test)[:, 1]

            self.meta_train_df[f"{model}_preds"] = train_preds
            self.meta_test_df[f"{model}_preds"] = test_preds

    def fit_level_two_model(self, search: bool = True):
        """Train a Logistic Regression model as level 2 model.

        If search is True, perform hyperparameter search using Optuna.
        If search is False, load existing best parameters.
        """
        log.info("Training a Logistic Regression model as level 2 model...")

        if search:
            log.info("\tFinding optimal hyperparameters using Optuna...")
            params = hyperparameter_search(
                X_train=self.meta_train_df,
                X_test=self.meta_test_df,
                y_train=self.y_train,
                y_test=self.y_test,
                model="lr",
                n_trials=Config.L2_N_TRIALS,
            )
            log.info(f"\n\tBest params: {params}\n")
        else:
            study = optuna.load_study(
                storage=Config.OPTUNA_STORAGE,
                study_name=self.level_two_study_name,
            )
            params = study.best_params
            log.info("\tUsing existing best parameters...\n")

        self.level_two_params = params

        # Train and store the level 2 model
        klass = MODEL_MAP["lr"]
        config = CONFIG_MAP["lr"]

        params.update(config.STATIC_PARAMS)
        clf = klass(**params)

        fit_params = config.get_fit_params(
            X_train=self.meta_train_df,
            y_train=self.y_train,
            X_val=self.meta_test_df,
            y_val=self.y_test,
            params=params,
        )

        clf.fit(X=self.meta_train_df, y=self.y_train, **fit_params)
        self.level_two_model = clf

        # Get predictions
        test_preds = clf.predict_proba(self.meta_test_df)[:, 1]
        self.meta_test_df["target"] = test_preds >= 0.5

        return self.meta_test_df

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions on new data using the trained ensemble.

        Args:
            X: DataFrame with features to predict on

        Returns:
            Series with binary predictions
        """
        if not self.level_one_models or not self.level_two_model:
            raise RuntimeError(
                "Models have not been trained. Call fit_level_one_models and fit_level_two_model first."
            )

        # Get meta-features from level 1 models
        meta_features = self.get_meta_features(X)

        # Use trained level 2 model (with its learned coefficients) to make predictions
        predictions = self.level_two_model.predict(meta_features)

        return pd.Series(predictions, index=X.index)

    def get_meta_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get meta features for new data using the trained ensemble.

        Args:
            X: DataFrame with features to predict on

        Returns:
            DataFrame with meta features
        """
        if not self.level_one_models:
            raise RuntimeError(
                "Level 1 models have not been trained. Call fit_level_one_models first."
            )

        meta_features = pd.DataFrame(index=X.index)
        for model_name, model in self.level_one_models.items():
            meta_features[f"{model_name}_preds"] = model.predict_proba(X)[:, 1]

        return meta_features

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions on new data using the trained ensemble.

        Args:
            X: DataFrame with features to predict on

        Returns:
            Array of probability predictions
        """
        if not self.level_one_models or not self.level_two_model:
            raise RuntimeError(
                "Models have not been trained. Call fit_level_one_models and fit_level_two_model first."
            )

        # Get level 1 predictions
        meta_features = self.get_meta_features(X)

        # Get level 2 predictions
        predictions = self.level_two_model.predict_proba(meta_features)

        return predictions

    def get_best_params(self, model: Optional[str] = None):
        """
        Retrieve best parameters from study for a specific model or all models.

        Args:
                model: Optional model name. If None, returns parameters for all models.

        Returns:
                Dictionary of best parameters from study for requested model(s)
        """
        if model is not None:
            # Load existing study for specific model
            study = optuna.load_study(
                storage=Config.OPTUNA_STORAGE,
                study_name=self.level_one_study_names[model],
            )
            return study.best_params

        # Load studies for all models
        best_params = {}
        for model_, study_name in self.level_one_study_names.items():
            study = optuna.load_study(
                storage=Config.OPTUNA_STORAGE, study_name=study_name
            )
            best_params[model_] = study.best_params
        study = optuna.load_study(
            storage=Config.OPTUNA_STORAGE, study_name=self.level_two_study_name
        )
        best_params["lr_l2"] = study.best_params

        return best_params

    def print_best_params(self):
        """Print best parameters for all models in a formatted way."""
        for model, params in self.best_params.items():
            if model == "level2_lr":
                log.info("\nLevel 2 (Logistic Regression) Best Parameters:")
            else:
                log.info(
                    f"\nLevel 1 - {Config.MODELS.get(model, model)} Best Parameters:"
                )
            for param, value in params.items():
                log.info(f"\t{param}: {value}")


# %%
out10 = get_node_output("handle_outliers_node")


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
# Replace NaN values with median for each column
log.info(
    f"Removed NaN rows. New shapes - X_train: {X_train.shape}, X_test: {X_test.shape}"
)

ensemble = Ensemble(X_train, X_test, y_train, y_test)
ensemble.models
ensemble.get_best_params()

# %%
ensemble.fit_level_one_models(search=False)
ensemble.meta_train_df.head()
ensemble.meta_test_df.head()
test_predictions = ensemble.fit_level_two_model(search=False)
# Get predictions for test data using the trained ensemble

test_predictions = ensemble.predict(out10["test_df_winsorized"])
test_predictions.to_csv("data/07_model_output/submission6.csv")

log.info("\nTest Set Predictions (first 5 rows):")
log.info(test_predictions[:5])


# Count disagreements between XGB and LGB predictions
# Get predictions for first 5 rows of test data

# Get level 1 predictions for each model
