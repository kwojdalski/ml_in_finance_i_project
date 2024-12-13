import logging

import numpy as np
import pandas as pd
import pytest
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner

from ml_in_finance_i_project.pipelines.data_science import (
    create_pipeline as create_ds_pipeline,
)
from ml_in_finance_i_project.pipelines.data_science.nodes import (
    split_data,
    train_decision_tree,
    tune_decision_tree,
)


@pytest.fixture
def dummy_data():
    # Create synthetic financial data
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "Open": np.random.uniform(100, 200, n_samples),
            "High": np.random.uniform(150, 250, n_samples),
            "Low": np.random.uniform(50, 150, n_samples),
            "Close": np.random.uniform(100, 200, n_samples),
            "Volume": np.random.uniform(1000000, 5000000, n_samples),
            "RET": np.random.choice([0, 1], n_samples),
        }
    )


@pytest.fixture
def dummy_parameters():
    parameters = {
        "model_options": {
            "test_size": 0.2,
            "random_state": 42,
            "target": "RET",
            "kfold": 5,
            "feature_importance_threshold": 0.01,
        }
    }
    return parameters


def test_split_data(dummy_data, dummy_parameters):
    X_train, X_test, y_train, y_test = split_data(
        dummy_data, dummy_parameters["model_options"]
    )
    assert len(X_train) == 80  # 80% of 100 samples
    assert len(y_train) == 80
    assert len(X_test) == 20  # 20% of 100 samples
    assert len(y_test) == 20
    assert "RET" not in X_train.columns
    assert "RET" not in X_test.columns


def test_train_decision_tree(dummy_data, dummy_parameters):
    X = dummy_data.drop("RET", axis=1)
    y = dummy_data["RET"]
    model = train_decision_tree(X, y)
    assert hasattr(model, "predict")
    assert hasattr(model, "score")

    # Test predictions
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(pred in [0, 1] for pred in predictions)


def test_tune_decision_tree(dummy_data, dummy_parameters):
    X = dummy_data.drop("RET", axis=1)
    y = dummy_data["RET"]
    result = tune_decision_tree(X, y, dummy_parameters["model_options"])

    assert "best_model" in result
    assert "best_params" in result
    assert "cv_results" in result
    assert hasattr(result["best_model"], "predict")
    assert isinstance(result["best_params"], dict)
    assert "max_depth" in result["best_params"]
    assert "criterion" in result["best_params"]


def test_data_science_pipeline(caplog, dummy_data, dummy_parameters):
    pipeline = create_ds_pipeline()
    catalog = DataCatalog()
    catalog.add_feed_dict(
        {
            "train_df_rm_duplicates": dummy_data,
            "params:model_options": dummy_parameters["model_options"],
        }
    )

    caplog.set_level(logging.DEBUG, logger="kedro")
    successful_run_msg = "Pipeline execution completed successfully."

    SequentialRunner().run(pipeline, catalog)

    assert successful_run_msg in caplog.text
