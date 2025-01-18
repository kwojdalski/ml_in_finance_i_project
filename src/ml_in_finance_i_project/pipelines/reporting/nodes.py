# This function uses plotly.express
import logging as log
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px  # noqa:  F401
import plotly.graph_objects as go
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path
path = Path(__file__).parent.parent.parent
sys.path.append(str(path))

from utils import CAT_COLS, ID_COLS


def plot_feature_importance(model, X_train: pd.DataFrame, threshold: float = 0.01):
    """
    Create a bar plot showing the importance of each feature in the model.

    Args:
        model: A fitted model object that has feature_importances_ attribute
        features (list[str]): List of feature names used in the model

    Returns:
        plotly.graph_objects.Figure: A plotly figure containing the feature importance plot
    """
    # Create dataframe with feature importances
    features = X_train.drop(columns=["RET"], errors="ignore").columns.tolist()
    feature_importances = pd.DataFrame(
        {"feature": features, "importance": model["model"].feature_importances_}
    )

    feature_importances = feature_importances.sort_values("importance", ascending=True)

    # Create bar plot
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=feature_importances["importance"],
            y=feature_importances["feature"],
            orientation="h",
            marker_color="steelblue",
        )
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", line_width=2)
    # Update layout
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Features",
        height=max(400, len(feature_importances) * 20),
        width=1200,
        showlegend=False,
    )
    return fig


def plot_nan_percentages(cleaned_train: pd.DataFrame) -> go.Figure:
    """Create a bar plot showing NaN percentages by category and subcategory.

    Args:
        cleaned_train: DataFrame containing the data to analyze

    Returns:
        plotly.graph_objects.Figure: A plotly figure containing the NaN percentage plot
    """
    # Calculate NaN percentages for each category and subcategory
    plot_df = pd.concat(
        [
            cleaned_train.groupby(category).apply(
                lambda x: pd.Series(
                    {
                        "Category": category,
                        "Subcategory": x.name,
                        "NaN_Percentage": (x.isna().any(axis=1).sum() / len(x) * 100),
                    }
                )
            )
            for category in CAT_COLS
        ]
    ).reset_index(drop=True)

    # Create subplot figure
    fig = go.Figure()

    # Add bars for each category
    for category in plot_df["Category"].unique():
        cat_data = plot_df[plot_df["Category"] == category]
        fig.add_trace(
            go.Bar(
                x=cat_data["Subcategory"],
                y=cat_data["NaN_Percentage"],
                name=category,
            )
        )

    # Update layout
    fig.update_layout(
        title="NaN Percentages by Category",
        xaxis_title="Subcategory",
        yaxis_title="Percentage (%)",
        barmode="group",
        height=600,
        width=1200,
        showlegend=True,
    )

    return fig


def plot_na(train_df):
    """Create plots showing missing value distributions."""
    na_counts = train_df.isna().sum()
    na_percentages = (na_counts / len(train_df)) * 100

    # Create summary DataFrame
    na_summary = pd.DataFrame(
        {"Missing Values": na_counts, "Percentage": na_percentages}
    )

    # Sort by percentage of missing values
    na_summary = na_summary.sort_values("Percentage", ascending=False)

    # Filter to only show variables with missing values
    na_summary = na_summary[na_summary["Missing Values"] > 0]

    if len(na_summary) > 0:
        log.info("\nVariables with missing values:")
        log.info(na_summary)
    else:
        log.info("\nNo missing values found in any variables")

    # Plot missing values distribution
    na_summary["is_ret"] = na_summary.index.str.contains("RET")

    # Create subplots
    fig = go.Figure()

    # Add bars for return and non-return variables
    for is_ret in [True, False]:
        subset = na_summary[na_summary["is_ret"] == is_ret]
        fig.add_trace(
            go.Bar(
                x=subset.index,
                y=subset["Percentage"],
                name="Return Variables" if is_ret else "Other Variables",
                marker_color="steelblue",
            )
        )

    # Update layout
    fig.update_layout(
        title="Missing Values Distribution",
        xaxis_title="Variables",
        yaxis_title="Percentage of Missing Values (%)",
        height=600,
        width=1000,
        showlegend=True,
        xaxis_tickangle=45,
    )

    return fig


def plot_model_accuracy(model_results: pd.DataFrame):
    """Plot accuracy comparison of different models.

    Args:
        model_results (dict): Dictionary containing model names and their accuracy scores

    Returns:
        plotly.graph_objects.Figure: A plotly figure showing model accuracy comparison
    """
    # Create faceted plot by Statistic
    model_results = model_results[model_results["Statistic"] == "accuracy"]
    fig = px.bar(
        model_results,
        x="Model",
        y="value",
        facet_col="Statistic",
        color="Model",
        facet_col_wrap=3,  # Show 3 plots per row
        height=1200,
        width=1200,
    )

    # Add benchmark line to each facet
    for annotation in fig.layout.annotations:
        if "accuracy" in annotation.text.lower():
            fig.add_hline(
                y=0.5131,
                line_dash="dash",
                line_color="red",
                annotation_text="Benchmark (0.5131)",
                annotation_position="bottom left",
                # row=annotation.row,
                # col=annotation.col,
            )

    # Update layout
    fig.update_layout(
        title="Model Performance Metrics",
        showlegend=True,
        yaxis_title="Value",
    )

    # Update facet formatting
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # Set free y-axis scales for each facet
    fig.update_yaxes(matches=None)

    return fig


def plot_ret_and_vol(train_df: pd.DataFrame, row_id: int = 24) -> go.Figure:
    """Create a plot showing returns and volume for a given row.

    Args:
        train_df: DataFrame containing the training data
        row_id: Row ID to plot

    Returns:
        plotly.graph_objects.Figure: A plotly figure with the visualization
    """
    # Prepare data
    days = np.arange(-20, 1)  # Include 0 for target point
    return_cols = [f"RET_{n}" for n in range(20, 0, -1)]
    volume_cols = [f"VOLUME_{n}" for n in range(20, 0, -1)]

    # Get returns and volumes
    returns = train_df[return_cols].iloc[row_id].values
    volumes = train_df[volume_cols].iloc[row_id].values / 10

    # Create figure
    fig = go.Figure()

    # Add historical returns
    fig.add_trace(
        go.Bar(
            x=days[:-1],  # Exclude last day (0) for historical data
            y=returns,
            name="Historical Returns",
            marker_color="steelblue",
            opacity=0.7,
        )
    )

    # Add target return
    target_return = 0.1 if train_df["RET"].iloc[row_id] else -0.1
    fig.add_trace(
        go.Bar(
            x=[0],
            y=[target_return],
            name="Target Return",
            marker_color="red",
            opacity=0.7,
        )
    )

    # Add volume line
    fig.add_trace(
        go.Scatter(
            x=days[:-1],  # Exclude last day (0) for volume data
            y=volumes,
            name="Volume (scaled down by 10)",
            line=dict(color="teal", width=2),
        )
    )

    # Update layout
    fig.update_layout(
        title="Returns and Volume Over Time",
        xaxis_title="Days to Target",
        yaxis_title="Return / Volume",
        height=800,
        width=1200,
        showlegend=True,
    )

    return fig


def plot_correlation_matrix(df):
    """
    Create and plot a correlation matrix heatmap for the given dataframe.

    Args:
        df (pd.DataFrame): Input dataframe to compute correlations for

    Returns:
        plotly.graph_objects.Figure: A plotly figure containing the correlation heatmap
    """
    # Compute correlation matrix
    corr_matrix = df.drop(columns=[ID_COLS], errors="ignore").corr()

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            zmin=-0.35,
            zmax=0.35,
            colorscale=[
                [0, "brown"],
                [0.5, "white"],
                [1, "green"],
            ],
        )
    )

    # Update layout
    fig.update_layout(
        title="Correlation Matrix",
        height=1200,
        width=1200,
        xaxis_tickangle=90,
    )

    return fig


def calculate_model_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    """Calculate comprehensive set of model metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "avg_precision": average_precision_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        # Additional statistical metrics
        "pred_pos_ratio": np.mean(y_pred),
        "true_pos_ratio": np.mean(y_true),
        "pred_std": np.std(y_proba),
    }

    # Calculate class-specific metrics
    for class_label in [0, 1]:
        mask = y_true == class_label
        if np.any(mask):
            metrics.update(
                {
                    f"class_{class_label}_precision": precision_score(
                        y_true, y_pred, pos_label=class_label
                    ),
                    f"class_{class_label}_recall": recall_score(
                        y_true, y_pred, pos_label=class_label
                    ),
                    f"class_{class_label}_f1": f1_score(
                        y_true, y_pred, pos_label=class_label
                    ),
                }
            )

    return metrics


def get_model_predictions(
    model: Any, X: np.ndarray, is_torch_model: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Get model predictions and probabilities.

    Args:
        model: Trained model
        X: Input features
        is_torch_model: Whether the model is a PyTorch model

    Returns:
        Tuple of (predictions, probabilities)
    """
    if is_torch_model:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values)
            outputs = model(X_tensor)
            y_proba = outputs.squeeze().numpy()
            y_pred = (outputs >= 0.5).squeeze().numpy()
    else:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

    return y_pred, y_proba


def calculate_benchmark_metrics(
    metrics: Dict[str, float], benchmark: float = 0.5131
) -> Dict[str, float]:
    """Calculate benchmark comparison metrics.

    Args:
        metrics: Dictionary of model metrics
        benchmark: Benchmark accuracy value

    Returns:
        Dictionary with added benchmark metrics
    """
    metrics["benchmark_accuracy"] = benchmark
    metrics["accuracy_vs_benchmark"] = metrics["accuracy"] - benchmark
    metrics["accuracy_pct_improvement"] = (
        (metrics["accuracy"] - benchmark) / benchmark * 100
    )
    metrics["relative_improvement"] = (metrics["accuracy"] / benchmark) - 1

    return metrics


def aggregate_model_results(
    base_dt: Optional[Any] = None,
    grid_dt: Optional[dict] = None,
    tuned_gb: Optional[dict] = None,
    nn_model: Optional[Any] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate comprehensive model results and metrics.

    Args:
        base_dt: Base decision tree model
        grid_dt: Tuned decision tree model dictionary
        tuned_gb: Tuned gradient boosting model dictionary
        nn_model: Neural network model
        X_test: Test features
        y_test: Test target
        X_test_selected: Selected test features

    Returns:
        Nested dictionary containing metrics for all models
    """
    all_metrics = {}

    # Base Decision Tree
    if base_dt is not None:
        y_pred, y_proba = get_model_predictions(
            base_dt["model"], X_test[base_dt["features"]]
        )
        base_metrics = calculate_model_metrics(y_test, y_pred, y_proba)
        all_metrics["base_decision_tree"] = calculate_benchmark_metrics(base_metrics)

    # Tuned Decision Tree
    if grid_dt is not None:
        y_pred, y_proba = get_model_predictions(
            grid_dt["model"], X_test[grid_dt["features"]]
        )
        tuned_dt_metrics = calculate_model_metrics(y_test, y_pred, y_proba)
        all_metrics["tuned_decision_tree"] = calculate_benchmark_metrics(
            tuned_dt_metrics
        )

    # Gradient Boosting stages
    if tuned_gb is not None:
        gb_stages = {
            "gb_n_estimators": tuned_gb["n_estimators_result"],
            "gb_tree_params": tuned_gb["tree_params_result"],
            "gb_leaf_params": tuned_gb["leaf_params_result"],
            "gb_max_features": tuned_gb["max_features_result"],
        }

        for stage_name, model in gb_stages.items():
            y_pred, y_proba = get_model_predictions(model, X_test[tuned_gb["features"]])
            stage_metrics = calculate_model_metrics(y_test, y_pred, y_proba)
            all_metrics[stage_name] = calculate_benchmark_metrics(stage_metrics)

    # Neural Network
    if nn_model is not None:
        y_pred, y_proba = get_model_predictions(
            nn_model["model"], X_test[nn_model["features"]], is_torch_model=True
        )
        nn_metrics = calculate_model_metrics(y_test, y_pred, y_proba)
        all_metrics["neural_network"] = calculate_benchmark_metrics(nn_metrics)

    # Add metadata
    all_metrics["metadata"] = {
        "n_test_samples": len(y_test),
        "n_features": X_test.shape[1],
        "n_features_selected": X_test[tuned_gb["features"]].shape[1],
        "class_distribution.positive": float(np.mean(y_test)),
        "class_distribution.negative": float(1 - np.mean(y_test)),
    }
    # metrics must be flattened
    all_metrics_flattened = {
        k + "." + k2: v2
        for k, v in all_metrics.items()
        for k2, v2 in (v.items() if isinstance(v, dict) else {k: v}.items())
    }
    model_results = (
        pd.DataFrame(all_metrics)
        .T.reset_index()
        .rename(columns={"index": "Model"})
        .melt(id_vars=["Model"], var_name="Statistic")
    )
    # return all_metrics
    return model_results, all_metrics_flattened
