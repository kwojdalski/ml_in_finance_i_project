"""
Stock Market Movement Prediction Pipeline

This module contains the main pipeline for predicting stock market movements using machine learning.
It implements data preprocessing, feature engineering, model training and evaluation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from kedro.pipeline import Pipeline, node
from kedro.io import DataCatalog

from .GBClassifierGridSearch import HistGBClassifierGridSearch
from .nn import Net
from .ta_indicators import (
    calculate_all_ta_indicators,
    filter_infinity_values,
    remove_duplicated_columns,
)
from .utils import (
    ID_COLS,
    drop_missing_returns,
    model_fit,
    plot_correlation_matrix,
    plot_model_accuracy,
    plot_nan_percentages,
    plot_ret_and_vol,
    preprocess_data,
    simulate_strategy,
)

# Keep all original comments and markdown sections
# %% [markdown]
# # Stock Market Movement Prediction
# ### Using Machine Learning to Forecast Next-Day Stock Price Movements
# This project tackles a binary classification problem in financial markets...
# [Keep all original markdown sections]

def create_pipeline(**kwargs) -> Pipeline:
    """Creates the Kedro pipeline for stock market prediction.
    
    Returns:
        Pipeline: The complete model training pipeline
    """
    return Pipeline(
        [
            node(
            a    func=load_and_preprocess_data,
                inputs=["x_train_raw", "y_train_raw", "x_test_raw", "parameters"],
                outputs=["train_df", "test_df"],
                name="load_and_preprocess_data_node",
            ),
            node(
                func=calculate_technical_indicators,
                inputs=["train_df", "test_df"],
                outputs=["ta_indicators_train", "ta_indicators_test"],
                name="calculate_technical_indicators_node", 
            ),
            node(
                func=process_and_save_data,
                inputs=[
                    "train_df",
                    "test_df", 
                    "ta_indicators_train",
                    "ta_indicators_test"
                ],
                outputs=["processed_train_df", "processed_test_df"],
                name="process_and_save_data_node",
            ),
            node(
                func=train_decision_tree,
                inputs=["processed_train_df", "parameters"],
                outputs="dt_model",
                name="train_decision_tree_node",
            ),
            node(
                func=train_gradient_boosting,
                inputs=["processed_train_df", "parameters"],
                outputs="gb_model",
                name="train_gradient_boosting_node",
            ),
            node(
                func=train_neural_network,
                inputs=["processed_train_df", "parameters"],
                outputs="nn_model",
                name="train_neural_network_node",
            ),
            node(
                func=evaluate_models,
                inputs=[
                    "dt_model",
                    "gb_model", 
                    "nn_model",
                    "processed_test_df"
                ],
                outputs="model_evaluation",
                name="evaluate_models_node",
            ),
        ]
    )

def load_and_preprocess_data(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame, 
    x_test: pd.DataFrame,
    parameters: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the raw training and test data.
    
    Args:
        x_train: Raw training features
        y_train: Raw training target
        x_test: Raw test features
        parameters: Pipeline parameters
        
    Returns:
        Tuple containing preprocessed train and test dataframes
    """
    # Keep original data loading and preprocessing code
    train_df, test_df = load_data(x_train, y_train, x_test)
    
    # Keep original preprocessing steps
    train_df = drop_missing_returns(train_df)
    train_df, test_df = preprocess_data(train_df, test_df, remove_id_cols=False)
    
    return train_df, test_df

d

# Keep all original plotting and visualization code
