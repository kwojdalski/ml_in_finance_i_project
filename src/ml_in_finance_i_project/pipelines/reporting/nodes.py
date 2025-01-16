# This function uses plotly.express
import logging as log
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px  # noqa:  F401
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score

sys.path
path = Path(__file__).parent.parent.parent
sys.path.append(str(path))

from utils import CAT_COLS, ID_COLS


def feature_importance(model, X_train_clean: pd.DataFrame, threshold: float = 0.01):
    """
    Create a bar plot showing the importance of each feature in the model.

    Args:
        model: A fitted model object that has feature_importances_ attribute
        features (list[str]): List of feature names used in the model

    Returns:
        plotly.graph_objects.Figure: A plotly figure containing the feature importance plot
    """
    # Create dataframe with feature importances
    features = X_train_clean.drop(columns=["RET"], errors="ignore").columns.tolist()
    feature_importances = pd.DataFrame(
        {"feature": features, "importance": model["best_model"].feature_importances_}
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
        width=800,
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
        width=1000,
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


def plot_model_accuracy(model_results):
    """Plot accuracy comparison of different models.

    Args:
        model_results (dict): Dictionary containing model names and their accuracy scores

    Returns:
        plotly.graph_objects.Figure: A plotly figure showing model accuracy comparison
    """
    plot_df = pd.DataFrame(
        {"Model": model_results.keys(), "Accuracy": model_results.values()}
    ).sort_values("Accuracy")

    fig = go.Figure()

    # Add bar trace
    fig.add_trace(
        go.Bar(
            x=plot_df["Model"],
            y=plot_df["Accuracy"],
            marker_color="steelblue",
        )
    )

    # Add benchmark line
    fig.add_hline(
        y=0.5131,
        line_dash="dash",
        line_color="red",
        annotation_text="Benchmark (0.5131)",
        annotation_position="bottom left",
    )

    # Update layout
    fig.update_layout(
        title="Model Accuracy Comparison",
        xaxis_title="Model",
        yaxis_title="Accuracy Score",
        height=600,
        width=800,
        showlegend=False,
        yaxis_range=[0.5, plot_df["Accuracy"].max() * 1.1],
    )

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
    returns = train_df.loc[row_id, return_cols].values
    volumes = train_df.loc[row_id, volume_cols].values / 10

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
    target_return = 0.1 if train_df.loc[row_id, "RET"] else -0.1
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
        height=600,
        width=800,
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
        height=800,
        width=800,
        xaxis_tickangle=90,
    )

    return fig


def simulate_strategy(y_test, y_predict, n_simulations=1000, n_days=100):
    """
    Simulate trading strategy performance based on model predictions.

    Args:
        y_test: True test labels
        y_predict: Model predictions
        n_simulations: Number of simulations to run
        n_days: Number of days to simulate

    Returns:
        plotly.graph_objects.Figure: A plotly figure showing the simulation results
    """
    # Get accuracy
    nn_accuracy = accuracy_score(y_test, y_predict)

    # Initialize strategy results
    nn_strategy = np.zeros(n_days)

    # Simulate
    for i in range(n_simulations):
        tmp_nn_strategy = []
        for d in range(n_days):
            tmp_nn_strategy.append(+1 if nn_accuracy > np.random.random() else -1)
        # Cumulative sum
        nn_strategy += np.cumsum(tmp_nn_strategy)

    # Compute average performance
    nn_strategy /= n_simulations

    # Create figure
    fig = go.Figure()

    # Add line trace
    fig.add_trace(
        go.Scatter(
            x=list(range(n_days)),
            y=nn_strategy,
            name=f"NN, acc = {nn_accuracy:1.4f}",
            line=dict(color="firebrick", width=2),
        )
    )

    # Update layout
    fig.update_layout(
        title="Avg Performance of Long-Short Strategy Simulation",
        xaxis_title="Days",
        yaxis_title="PnL",
        height=600,
        width=800,
        showlegend=True,
    )

    return fig


def evaluate_models(models, X_test, y_test):
    """Evaluate all trained models and return dictionary of accuracy scores.

    Args:
        models (dict): Dictionary mapping model names to model objects
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: Dictionary mapping model names to accuracy scores
    """

    results = {}

    for name, model in models.items():
        if hasattr(model, "best_estimator_"):
            # For grid search results
            score = model.best_estimator_.score(X_test, y_test)
        elif hasattr(model, "score"):
            # For sklearn models
            score = model.score(X_test, y_test)
        else:
            # For neural network or other models
            score = accuracy_score(y_test, model)

        results[name] = score

    return results


def aggregate_model_results(
    base_dt=None,
    grid_dt=None,
    tuned_gb=None,
    nn_model=None,
    X_test=None,
    y_test=None,
    X_test_selected=None,
) -> dict:
    """Aggregate model results into a dictionary.

    Args:
        base_dt: Base decision tree model
        grid_dt: Tuned decision tree model
        tuned_gb: Tuned gradient boosting model
        nn_model: Neural network model

    Returns:
        Dictionary containing the trained models
    """
    return {
        "Decision Tree (Base)": base_dt.score(X_test, y_test),
        "Decision Tree (Tuned)": grid_dt["best_model"].score(X_test, y_test),
        "GB (n_estimators)": tuned_gb["n_estimators_result"].score(X_test, y_test),
        "GB (+ tree params)": tuned_gb["tree_params_result"].score(X_test, y_test),
        "GB (+ leaf params)": tuned_gb["leaf_params_result"].score(X_test, y_test),
        "GB (+ max features)": tuned_gb["max_features_result"].score(X_test, y_test),
    }
