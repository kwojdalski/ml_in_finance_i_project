import numpy as np
import pandas as pd
import torch
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from utils import model_fit

from GBClassifierGridSearch import HistGBClassifierGridSearch
from nn import Net


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

    n_epochs = parameters["nn_epochs"]
    batch_size = parameters["batch_size"]

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


def evaluate_models(
    dt_model: tree.DecisionTreeClassifier,
    gb_model: HistGBClassifierGridSearch,
    nn_model: Net,
    test_df: pd.DataFrame,
) -> dict:
    """Evaluate all trained models.

    Args:
        dt_model: Trained decision tree
        gb_model: Trained gradient boosting
        nn_model: Trained neural network
        test_df: Test data

    Returns:
        Dictionary of model evaluation metrics
    """
    # Keep original evaluation code
    model_results = {
        "Decision Tree": dt_model.score(test_df.drop("RET", axis=1), test_df["RET"]),
        "Gradient Boosting": gb_model.score(
            test_df.drop("RET", axis=1), test_df["RET"]
        ),
    }

    # Neural network evaluation
    nn_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(test_df.drop("RET", axis=1).values)
        outputs = nn_model(X_test_tensor)
        y_predict = (outputs >= 0.5).squeeze().numpy()
        model_results["Neural Network"] = accuracy_score(test_df["RET"], y_predict)

    return model_results


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
        cv=parameters["kfold"],
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
