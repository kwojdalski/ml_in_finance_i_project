# %%
import logging as log
import sys
from itertools import compress
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scikeras.wrappers import KerasRegressor
from sklearn import tree
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from torch import nn

path = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(path))

from src.qrt_stock_returns.GBClassifierGridSearch import (  # noqa
    HistGBClassifierGridSearch,
)
from src.qrt_stock_returns.nn import Net  # noqa

logger = log.getLogger(__name__)
logger.setLevel(log.INFO)


def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    y = data[parameters["target"]]
    X = data.drop(parameters["target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return X_train, X_test, y_train, y_test


def train_decision_tree(
    X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: dict
) -> tree.DecisionTreeClassifier:
    """Train and tune decision tree model.

    Args:
        train_df: Processed training data
        parameters: Model parameters

    Returns:
        Trained decision tree model
    """
    base_dt = tree.DecisionTreeClassifier()
    features = [col for col in X_train.columns if col != parameters["target"]]
    x_train = X_train[features]
    model_fit(base_dt, x_train, y_train, performCV=False)
    return {"model": base_dt, "features": features}


def train_gradient_boosting(
    X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: dict
) -> HistGBClassifierGridSearch:
    """Train and tune gradient boosting model.

    Args:
        train_df: Processed training data
        parameters: Model parameters

    Returns:
        Trained gradient boosting model
    """
    # Keep original gradient boosting code
    gbm_classifier = HistGBClassifierGridSearch()
    features = [col for col in X_train.columns if col != parameters["target"]]
    x_train = X_train[features]
    gbm_classifier.run(x_train, y_train)

    return {"model": gbm_classifier, "features": features}


def train_neural_network(
    X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: dict
) -> Net:
    """Train neural network model.

    Args:
        train_df: Processed training data
        parameters: Model parameters

    Returns:
        Trained neural network model
    """
    # Keep original neural network code
    features = [col for col in X_train.columns if col != parameters["target"]]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    nn_model = Net(len(features))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(nn_model.parameters())

    # Keep original training loop
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)

    n_epochs = parameters["model_options"]["nn_epochs"]
    batch_size = parameters["model_options"]["batch_size"]

    for epoch in range(n_epochs):
        log.info(f"Epoch {epoch+1}/{n_epochs}")
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i : i + batch_size]
            batch_y = y_train_tensor[i : i + batch_size]
            outputs = nn_model(batch_X)
            loss = criterion(outputs, batch_y.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return {"model": nn_model, "features": features}


def tune_decision_tree(
    X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: dict
) -> dict:
    """Tune decision tree hyperparameters using grid search cross validation.

    Args:
        X_train: Training features
        y_train: Training target
        parameters: Model parameters including kfold value

    Returns:
        Dictionary containing:
        - best_model: The tuned decision tree model
        - best_params: Best hyperparameters found
        - cv_results: Cross validation results
    """
    features = [col for col in X_train.columns if col != parameters["target"]]
    params = {"max_depth": np.arange(2, 7), "criterion": ["gini", "entropy"]}
    tree_estimator = tree.DecisionTreeClassifier()

    grid_tree = GridSearchCV(
        tree_estimator,
        params,
        cv=parameters["model_options"]["kfold"],
        scoring="accuracy",
        n_jobs=1,
        verbose=False,
    )

    grid_tree.fit(X_train, y_train)

    cv_results = {
        "mean_scores": grid_tree.cv_results_["mean_test_score"],
        "std_scores": grid_tree.cv_results_["std_test_score"],
        "params": grid_tree.cv_results_["params"],
    }

    best_params = grid_tree.best_params_

    tuned_model = tree.DecisionTreeClassifier(
        max_depth=best_params["max_depth"], criterion="gini"
    )
    model_fit(
        tuned_model,
        X_train,
        y_train,
    )

    return {
        "model": tuned_model,
        "best_params": best_params,
        "cv_results": cv_results,
        "features": features,
    }


def remove_nan_rows(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> tuple:
    """Remove rows containing NaN values from training data.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Tuple containing cleaned X_train, X_test, y_train, y_test
    """
    nan_mask = X_train.isna().any(axis=1)
    X_train_clean = X_train[~nan_mask]
    y_train_clean = y_train[~nan_mask]
    # Fill NaN values in test set using nearest neighbor imputation

    imputer = KNNImputer(n_neighbors=5)
    X_test_clean = pd.DataFrame(
        imputer.fit_transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_clean, X_test_clean, y_train_clean, y_test


def tune_gradient_boosting(
    gbm_classifier,
    X_train_clean: pd.DataFrame,
    y_train_clean: pd.Series,
    parameters: dict,
) -> dict:
    """Tune gradient boosting hyperparameters sequentially.

    Args:
        gbm_classifier: Base gradient boosting classifier
        X_train: Training features
        y_train: Training target

    Returns:
        Dictionary containing tuning results for each parameter group:
        - n_estimators_result: Results from tuning number of estimators
        - tree_params_result: Results from tuning tree parameters
        - leaf_params_result: Results from tuning leaf parameters
        - max_features_result: Results from tuning max features
    """
    # Tune number of estimators
    n_estimators_result = gbm_classifier["model"].tune_n_estimators(
        X_train_clean, y_train_clean
    )
    features = [col for col in X_train_clean.columns if col != parameters["target"]]
    # Tune tree parameters using best n_estimators
    tree_params_result = gbm_classifier["model"].tune_tree_params(
        X_train_clean, y_train_clean, {**n_estimators_result.best_params_}
    )

    # Tune leaf parameters using best n_estimators and tree params
    leaf_params_result = gbm_classifier["model"].tune_leaf_params(
        X_train_clean,
        y_train_clean,
        {**n_estimators_result.best_params_, **tree_params_result.best_params_},
    )

    # Tune max features using all previous best parameters
    max_features_result = gbm_classifier["model"].tune_max_features(
        X_train_clean,
        y_train_clean,
        {
            **n_estimators_result.best_params_,
            **tree_params_result.best_params_,
            **leaf_params_result.best_params_,
        },
    )

    return {
        "n_estimators_result": n_estimators_result,
        "tree_params_result": tree_params_result,
        "leaf_params_result": leaf_params_result,
        "max_features_result": max_features_result,
        "features": features,
    }


def select_important_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    grid_dt: dict,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Select features based on importance threshold from decision tree model.

    Args:
        X_train: Training features
        X_test: Test features
        grid_dt: Tuned decision tree model dictionary
        features: List of feature names

    Returns:
        Tuple of filtered training and test datasets
    """
    threshold = parameters["model_options"]["feature_importance_threshold"]
    features = X_train.columns.drop(parameters["target"], errors="ignore").tolist()
    important_features = list(
        compress(
            features,
            grid_dt["model"].feature_importances_ >= threshold,
        )
    )

    X_train_selected = X_train.loc[:, important_features]
    X_test_selected = X_test.loc[:, important_features]

    return X_train_selected, X_test_selected, important_features


def evaluation(model, X: pd.DataFrame, Y: pd.Series, kfold: int):
    """
    Evaluate a model using k-fold cross validation and print performance metrics.

    This function performs k-fold cross validation on the given model and prints the mean
    and standard deviation of accuracy, precision and recall scores. This helps assess
    model performance and detect potential overfitting.

    Args:
        model: A fitted sklearn model object that implements predict()
        X (pd.DataFrame): Feature matrix
        Y (pd.Series): Target variable
        kfold (int): Number of folds for cross validation

    Returns:
        None: Prints cross validation metrics to log
    """
    # Cross Validation to test and anticipate overfitting problem
    scores1 = cross_val_score(model, X, Y, cv=kfold, scoring="accuracy")
    scores2 = cross_val_score(model, X, Y, cv=kfold, scoring="precision")
    scores3 = cross_val_score(model, X, Y, cv=kfold, scoring="recall")
    # The mean score and standard deviation of the score estimate
    log.info(
        f"Cross Validation Accuracy: {scores1.mean():.5f} (+/- {scores1.std():.2f})"
    )
    log.info(
        f"Cross Validation Precision: {scores2.mean():.5f} (+/- {scores2.std():.2f})"
    )
    log.info(f"Cross Validation Recall: {scores3.mean():.5f} (+/- {scores3.std():.2f})")


# %%
def compute_roc(
    Y: pd.Series, y_pred: pd.Series, plot: bool = True
) -> tuple[float, float, float]:
    fpr = dict()
    tpr = dict()
    auc_score = dict()
    fpr, tpr, _ = roc_curve(Y, y_pred)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=3, linestyle="--")
        plt.legend(loc="upper right")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic")
        plt.show()
    return fpr, tpr, auc_score


def model_fit(  # noqa
    model,
    X: pd.DataFrame,
    Y: pd.Series,
    features: list[str] | None = None,
    performCV: bool = True,
    roc: bool = False,
) -> None:
    """
    Fits a model, makes predictions, and evaluates performance using confusion matrix, accuracy score,
    cross-validation, ROC curve and feature importance analysis.
    """
    # Fitting the model on the data_set
    if features is not None:
        X = X[features]

    model.fit(X, Y)
    # Predict training set:
    predictions = model.predict(X)
    # Create and print confusion matrix
    cfm = confusion_matrix(Y, predictions)
    log.info("\nModel Confusion matrix")
    log.info(cfm)

    # Print model report:
    log.info("\nModel Report")
    log.info(f"Accuracy : {accuracy_score(Y.values, predictions):.4g}")

    # Perform cross-validation: evaluate using 10-fold cross validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    if performCV:
        evaluation(model, X, Y, kfold)
    if roc:
        compute_roc(Y, predictions, plot=True)


def train_xgboost(X_train, y_train, parameters: dict):
    """Train XGBoost model and perform grid search CV."""
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_jobs=1,
        reg_alpha=0.5,
        reg_lambda=0.5,
        eval_metric="logloss",
        verbosity=2,
        # early_stopping_rounds=10,
        # callbacks=[xgb.callback.EarlyStopping(rounds=10, save_best=True)]
    )

    optimization_dict = {
        "max_depth": [2, 4, 6, 8],
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "max_delta_step": [0, 1, 2],
    }
    # Common search parameters
    SEARCH_PARAMS = {
        "scoring": "accuracy",
        "verbose": 3,
        "cv": 5,
        "random_state": 42,
        "n_jobs": 1,
    }

    # Search space definitions
    BAYES_SEARCH_SPACE = {
        "max_depth": Integer(2, 8),
        "n_estimators": Integer(50, 300),
        "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
        "min_child_weight": Integer(1, 5),
        "gamma": Real(0, 0.2),
        "subsample": Real(0.8, 1.0),
        "colsample_bytree": Real(0.8, 1.0),
        "max_delta_step": Integer(0, 2),
    }

    def _create_search_model(search_type, xgb_model, param_space=None):
        """Factory function to create search model based on type"""
        search_models = {
            "grid": lambda: GridSearchCV(
                estimator=xgb_model, param_grid=param_space, **SEARCH_PARAMS
            ),
            "random": lambda: RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=param_space,
                n_iter=50,
                **SEARCH_PARAMS,
            ),
            "bayes": lambda: BayesSearchCV(
                estimator=xgb_model,
                search_spaces=BAYES_SEARCH_SPACE,
                n_iter=50,
                **SEARCH_PARAMS,
            ),
        }

        return search_models.get(search_type, search_models["bayes"])()

    # Get optimization method from parameters with "grid" as default
    optimization_method = parameters.get("optimization_method", "bayes")
    model = _create_search_model(optimization_method, xgb_model, optimization_dict)
    model.fit(X_train, y_train)

    return model.best_estimator_


def train_lightgbm(X_train, y_train, parameters: dict):
    """Train LightGBM model and perform hyperparameter optimization."""
    lgb_model = lgb.LGBMClassifier(
        objective="binary", metric="binary_logloss", verbose=-1, random_state=42
    )

    optimization_dict = {
        "max_depth": [2, 4, 6, 8],
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [10, 20, 30],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0, 0.1, 0.5],
    }

    # Common search parameters
    SEARCH_PARAMS = {
        "scoring": "accuracy",
        "verbose": 3,
        "cv": 5,
        "random_state": 42,
        "n_jobs": 1,
    }

    # Search space definitions
    BAYES_SEARCH_SPACE = {
        "max_depth": Integer(2, 8),
        "n_estimators": Integer(50, 300),
        "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
        "num_leaves": Integer(15, 63),
        "min_child_samples": Integer(10, 30),
        "subsample": Real(0.8, 1.0),
        "colsample_bytree": Real(0.8, 1.0),
        "reg_alpha": Real(0, 0.5),
        "reg_lambda": Real(0, 0.5),
    }

    def _create_search_model(search_type, lgb_model, param_space=None):
        """Factory function to create search model based on type"""
        search_models = {
            "grid": lambda: GridSearchCV(
                estimator=lgb_model, param_grid=param_space, **SEARCH_PARAMS
            ),
            "random": lambda: RandomizedSearchCV(
                estimator=lgb_model,
                param_distributions=param_space,
                n_iter=50,
                **SEARCH_PARAMS,
            ),
            "bayes": lambda: BayesSearchCV(
                estimator=lgb_model,
                search_spaces=BAYES_SEARCH_SPACE,
                n_iter=50,
                **SEARCH_PARAMS,
            ),
        }

        return search_models.get(search_type, search_models["bayes"])()

    # Get optimization method from parameters with "grid" as default
    optimization_method = parameters.get("optimization_method", "bayes")
    model = _create_search_model(optimization_method, lgb_model, optimization_dict)
    model.fit(X_train, y_train)

    return model.best_estimator_


def train_gru(X_train, y_train, parameters: dict):
    """Train GRU model and perform hyperparameter optimization."""

    def create_gru_model(units=64, dropout=0.2, learning_rate=0.001):
        model = Sequential(
            [
                GRU(
                    units=units,
                    return_sequences=True,
                    input_shape=(X_train.shape[1], 1),
                ),
                Dropout(dropout),
                GRU(units=units // 2),
                Dropout(dropout),
                Dense(1),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
        return model

    optimization_dict = {
        "units": [32, 64, 128],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [0.001, 0.01, 0.1],
        "batch_size": [32, 64, 128],
        "epochs": [10, 20, 30],
    }

    # Common search parameters
    SEARCH_PARAMS = {
        "scoring": "neg_mean_squared_error",
        "verbose": 3,
        "cv": 5,
        "random_state": 42,
        "n_jobs": 1,
    }

    # Search space definitions
    BAYES_SEARCH_SPACE = {
        "units": Integer(32, 128),
        "dropout": Real(0.1, 0.3),
        "learning_rate": Real(0.001, 0.1, prior="log-uniform"),
        "batch_size": Integer(32, 128),
        "epochs": Integer(10, 30),
    }

    def _create_search_model(search_type, param_space=None):
        """Factory function to create search model based on type"""
        model_builder = KerasRegressor(build_fn=create_gru_model)

        search_models = {
            "grid": lambda: GridSearchCV(
                estimator=model_builder, param_grid=param_space, **SEARCH_PARAMS
            ),
            "random": lambda: RandomizedSearchCV(
                estimator=model_builder,
                param_distributions=param_space,
                n_iter=50,
                **SEARCH_PARAMS,
            ),
            "bayes": lambda: BayesSearchCV(
                estimator=model_builder,
                search_spaces=BAYES_SEARCH_SPACE,
                n_iter=50,
                **SEARCH_PARAMS,
            ),
        }

        return search_models.get(search_type, search_models["bayes"])()

    # Reshape input for GRU (samples, timesteps, features)
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Get optimization method from parameters with "grid" as default
    optimization_method = parameters.get("optimization_method", "bayes")
    model = _create_search_model(optimization_method, optimization_dict)
    model.fit(X_train_reshaped, y_train)

    return model.best_estimator_
