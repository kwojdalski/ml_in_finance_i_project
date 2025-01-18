import logging as log
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import cross_val_score

kfold = 2
ID_COLS = [
    "ID",
    "STOCK",
    "DATE",
    "INDUSTRY",
    "INDUSTRY_GROUP",
    "SECTOR",
    "SUB_INDUSTRY",
]
CAT_COLS = [
    "INDUSTRY",
    "INDUSTRY_GROUP",
    "SECTOR",
    "SUB_INDUSTRY",
    "STOCK",
    "DATE",
]


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
        "Cross Validation Accuracy: %0.5f (+/- %0.2f)" % (scores1.mean(), scores1.std())
    )
    log.info(
        "Cross Validation Precision: %0.5f (+/- %0.2f)"
        % (scores2.mean(), scores2.std())
    )
    log.info(
        "Cross Validation Recall: %0.5f (+/- %0.2f)" % (scores3.mean(), scores3.std())
    )


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
        plt.plot(fpr, tpr, color="blue", label="ROC curve (area = %0.2f)" % auc_score)
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


def load_data(
    x_train: str | Path,
    y_train: str | Path,
    x_test: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test data, preprocess by:
    - Loading CSV files
    - Dropping NA values and ID columns
    - Computing moving averages and mean returns
    - Converting target to binary

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - train_df: Preprocessed training dataframe
            - test_df: Preprocessed test dataframe
    """
    # Load data
    x_train: pd.DataFrame = pd.read_csv("./data/x_train.csv")
    y_train: pd.DataFrame = pd.read_csv("./data/y_train.csv")
    train_df: pd.DataFrame = pd.concat([x_train, y_train], axis=1)
    test_df: pd.DataFrame = pd.read_csv("./data/x_test.csv")

    return train_df, test_df


def drop_missing_returns(train_df: pd.DataFrame, n_days: int = 5) -> pd.DataFrame:
    """Drop rows where all return features for the last n_days are missing.

    Args:
        train_df: DataFrame containing return features named RET_1, RET_2, etc.
        n_days: Number of most recent days to check for missing returns

    Returns:
        DataFrame with rows dropped where all return features are missing
    """
    return_features = [f"RET_{day}" for day in range(1, n_days + 1)]

    # Calculate proportion of missing values for each row
    missing_prop = (
        train_df[return_features].isna().sum(axis=1)
        / train_df[return_features].shape[1]
    )

    # Drop rows where all return features are missing
    return train_df[missing_prop < 1].copy()


def calculate_statistics(data: pd.DataFrame) -> pd.Series:
    """Calculate statistical measures for RET_1 and VOLUME_1 columns."""
    stats: dict[str, float] = {}

    # Helper function to calculate stats for a column
    def get_column_stats(col_name: str) -> dict[str, float]:
        series: pd.Series = data[col_name]
        log_series: pd.Series = np.log(series.replace(0, np.nan)).dropna()
        log_squared: pd.Series = log_series**2
        base_stats: dict[str, float] = {
            f"Mean_{col_name}": np.mean(series),
            f"Skewness_{col_name}": skew(series.dropna()),
            f"Kurtosis_{col_name}": kurtosis(series.dropna()),
            f"Std_{col_name}": np.std(series),
        }

        log_stats: dict[str, float] = {
            f"Log_Mean_{col_name}": np.mean(log_series),
            f"Log_Skewness_{col_name}": skew(log_series.dropna()),
            f"Log_Kurtosis_{col_name}": kurtosis(log_series.dropna()),
            f"Log_Std_{col_name}": np.std(log_series),
        }

        squared_stats: dict[str, float] = {
            f"Log_Squared_Skewness_{col_name}": skew(log_squared.dropna()),
            f"Log_Squared_Kurtosis_{col_name}": kurtosis(log_squared.dropna()),
            f"Log_Squared_Std_{col_name}": np.std(log_squared),
        }

        return {**base_stats, **log_stats, **squared_stats}

    # Calculate stats for both RET_1 and VOLUME_1
    stats.update(get_column_stats("RET_1"))
    stats.update(get_column_stats("VOLUME_1"))

    return pd.Series(stats)


# %%


def get_node_idx(pipeline, node_name):
    """Get the index of a node in a pipeline by its name.

    Args:
        pipeline: The pipeline to search in
        node_name: Name of the node to find

    Returns:
        int: Index of the node in the pipeline
    """
    return next(i for i, node in enumerate(pipeline.nodes) if node.name == node_name)


def get_node_names(pipeline):
    """Get list of all node names in a pipeline.

    Args:
        pipeline: The pipeline to get node names from

    Returns:
        list: List of node names in the pipeline
    """
    return [node.name for node in pipeline.nodes]


def run_pipeline_node(pipeline_name: str, node_name: str, inputs: dict):
    """Run a specific node from a pipeline.

    Args:
        pipeline_name: Name of the pipeline
        node_name: Name of the node to run
        inputs: Dictionary of input parameters for the node

    Returns:
        Output from running the node
    """
    node_idx = get_node_idx(pipelines[pipeline_name], node_name)
    return pipelines[pipeline_name].nodes[node_idx].run(inputs)


def _handle_dataframe_io(filepath: str, df=None, mode="read"):
    """Handle reading/writing dataframes in different formats.

    Args:
        filepath: Path to the data file
        df: DataFrame to save (only needed for write mode)
        mode: Either 'read' or 'write'

    Returns:
        DataFrame if reading, None if writing
    """
    if filepath.endswith(".csv"):
        print(f"{'Loading' if mode=='read' else 'Saving'} CSV file...")
        return (
            pd.read_csv(filepath)
            if mode == "read"
            else df.to_csv(filepath, index=False)
        )
    elif filepath.endswith(".parquet"):
        print(f"{'Loading' if mode=='read' else 'Saving'} Parquet file...")
        return (
            pd.read_parquet(filepath)
            if mode == "read"
            else df.to_parquet(filepath, index=False)
        )
    elif filepath.endswith(".pickle") or filepath.endswith(".pkl"):
        print(f"{'Loading' if mode=='read' else 'Saving'} Pickle file...")
        return pd.read_pickle(filepath) if mode == "read" else df.to_pickle(filepath)
    else:
        raise ValueError(f"Unsupported file extension for {filepath}")


def get_node_inputs(node, catalog):
    """Get input paths for a pipeline node and load the data.

    Args:
        node: Pipeline node to get inputs for
        catalog: Data catalog containing dataset info

    Returns:
        dict: Dictionary mapping input names to their loaded dataframes
    """
    print(f"Getting inputs for node: {node.name}")
    print(f"Node inputs: {node.inputs}")

    inputs = {}
    for input_name in node.inputs:
        if isinstance(input_name, str):
            print(f"\nProcessing input: {input_name}")
            if not input_name.startswith("params:"):
                try:
                    filepath = str(catalog.datasets[input_name]._filepath)
                    print(f"Loading from filepath: {filepath}")

                    df = _handle_dataframe_io(filepath, mode="read")
                    print(f"Successfully loaded dataframe with shape: {df.shape}")
                    inputs[input_name] = df
                except Exception as e:
                    print(f"Error loading data for {input_name}: {str(e)}")
                    inputs[input_name] = None
            else:
                print(f"Skipping parameter input: {input_name}")

    print(f"\nFinished loading {len(inputs)} inputs")
    return inputs


def get_node_outputs(node, catalog):
    """Get output data from a pipeline node.

    Args:
        node: Pipeline node to get outputs for
        catalog: Data catalog containing dataset info
        outputs: Dictionary of output data to save

    Returns:
        dict: Dictionary mapping output names to their loaded dataframes
    """
    print(f"Getting outputs for node: {node.name}")
    print(f"Node outputs: {node.outputs}")

    output_dict = {}
    for output_name in node.outputs:
        if isinstance(output_name, str):
            print(f"\nProcessing output: {output_name}")
            try:
                filepath = str(catalog.datasets[output_name]._filepath)
                print(f"Loading from filepath: {filepath}")

                df = _handle_dataframe_io(filepath, mode="read")
                print(f"Successfully loaded dataframe with shape: {df.shape}")
                output_dict[output_name] = df
            except Exception as e:
                print(f"Error loading data for {output_name}: {str(e)}")
                output_dict[output_name] = None

    print(f"\nFinished loading {len(node.outputs)} outputs")
    return output_dict
