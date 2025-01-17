from itertools import compress

import numpy as np
import pandas as pd
import torch
from GBClassifierGridSearch import HistGBClassifierGridSearch
from sklearn import tree
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from utils import model_fit

from ml_in_finance_i_project.nn import Net


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
    X_train: pd.DataFrame, y_train: pd.DataFrame
) -> tree.DecisionTreeClassifier:
    """Train and tune decision tree model.

    Args:
        train_df: Processed training data
        parameters: Model parameters

    Returns:
        Trained decision tree model
    """
    base_dt = tree.DecisionTreeClassifier()
    model_fit(base_dt, X_train, y_train, performCV=False)
    return base_dt


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

    return gbm_classifier


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
        print(f"Epoch {epoch+1}/{n_epochs}")
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i : i + batch_size]
            batch_y = y_train_tensor[i : i + batch_size]
            outputs = nn_model(batch_X)
            loss = criterion(outputs, batch_y.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return nn_model


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
        printFeatureImportance=True,
    )

    return {
        "best_model": tuned_model,
        "best_params": best_params,
        "cv_results": cv_results,
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
    gbm_classifier, X_train_clean: pd.DataFrame, y_train_clean: pd.Series
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
    n_estimators_result = gbm_classifier.tune_n_estimators(X_train_clean, y_train_clean)

    # Tune tree parameters using best n_estimators
    tree_params_result = gbm_classifier.tune_tree_params(
        X_train_clean, y_train_clean, {**n_estimators_result.best_params_}
    )

    # Tune leaf parameters using best n_estimators and tree params
    leaf_params_result = gbm_classifier.tune_leaf_params(
        X_train_clean,
        y_train_clean,
        {**n_estimators_result.best_params_, **tree_params_result.best_params_},
    )

    # Tune max features using all previous best parameters
    max_features_result = gbm_classifier.tune_max_features(
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
            grid_dt["best_model"].feature_importances_ >= threshold,
        )
    )

    X_train_selected = X_train.loc[:, important_features]
    X_test_selected = X_test.loc[:, important_features]

    return X_train_selected, X_test_selected, important_features
