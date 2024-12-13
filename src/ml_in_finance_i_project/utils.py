import logging as log
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    annotate,
    coord_cartesian,
    coord_equal,
    coord_flip,
    element_text,
    facet_wrap,
    geom_bar,
    geom_hline,
    geom_line,
    geom_tile,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_gradient2,
    scale_x_continuous,
    theme,
)
from scipy.stats import kurtosis, skew
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

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


# %%
def feature_importance(model, features: list[str], threshold: float = 0.01):
    """
    Create a bar plot showing the importance of each feature in the model.

    Args:
        model: A fitted model object that has feature_importances_ attribute
        features (list[str]): List of feature names used in the model

    Returns:
        plotnine.ggplot: A plotnine plot object containing the feature importance plot
    """
    # Create dataframe with feature importances
    feature_importances = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    )

    # Sort by importance
    feature_importances = feature_importances.sort_values("importance", ascending=True)

    # Create plot
    plot = (
        ggplot(
            feature_importances, aes(x="reorder(feature, importance)", y="importance")
        )
        + geom_bar(stat="identity", fill="steelblue")
        + geom_hline(yintercept=threshold, linetype="dashed", color="red")
        + coord_flip()
        + labs(title="Feature Importance", x="Features", y="Importance")
        + theme(figure_size=(13, 12))
    )
    return plot


# %%
def model_fit(
    model,
    X: pd.DataFrame,
    Y: pd.Series,
    features: list[str],
    performCV: bool = True,
    roc: bool = False,
    printFeatureImportance: bool = False,
) -> None:
    """
    Fits a model, makes predictions, and evaluates performance using confusion matrix, accuracy score,
    cross-validation, ROC curve and feature importance analysis.
    """
    # Fitting the model on the data_set
    model.fit(X[features], Y)

    # Predict training set:
    predictions = model.predict(X[features])
    # predprob = model.predict_proba(X[features])[:, 1]

    # Create and print confusion matrix
    cfm = confusion_matrix(Y, predictions)
    log.info("\nModel Confusion matrix")
    log.info(cfm)

    # Print model report:
    log.info("\nModel Report")
    log.info("Accuracy : %.4g" % accuracy_score(Y.values, predictions))

    # Perform cross-validation: evaluate using 10-fold cross validation
    # kfold = StratifiedKFold(n_splits=10, shuffle=True)
    if performCV:
        evaluation(model, X[features], Y, kfold)
    if roc:
        compute_roc(Y, predictions, plot=True)

    # Print Feature Importance:
    if printFeatureImportance:
        if isinstance(model, HistGradientBoostingClassifier):
            warnings.warn(
                "Feature importance is only supported for GradientBoostingClassifier"
            )
        else:
            feature_importance(model, features)


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


def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    remove_id_cols: bool = True,
    sample_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Clean data
    n_before = len(train_df)
    train_df = train_df.dropna()
    n_after = len(train_df)
    log.debug(f"Dropped {n_before - n_after} rows with NA values from train_df")

    if remove_id_cols:
        train_df = train_df.drop(ID_COLS, axis=1)
        log.debug(f"Dropped ID columns {ID_COLS} from train_df")

    n_before = len(test_df)
    test_df = test_df.dropna()
    n_after = len(test_df)
    log.debug(f"Dropped {n_before - n_after} rows with NA values from test_df")

    if remove_id_cols:
        test_df = test_df.drop(ID_COLS, axis=1)
        log.debug(f"Dropped ID columns {ID_COLS} from test_df")

    # Convert target to binary
    sign_of_return: LabelEncoder = LabelEncoder()
    train_df["RET"] = sign_of_return.fit_transform(train_df["RET"])

    # Sample training data if sample_n is provided
    if sample_n is not None:
        train_df = train_df.sample(n=sample_n, random_state=42)
        log.debug(f"Sampled {sample_n} rows from train_df")

    return train_df, test_df


def plot_nan_percentages(cleaned_train: pd.DataFrame) -> ggplot:
    # Create a list of categories to analyze

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

    # Create plot
    p = (
        ggplot(plot_df, aes(x="Subcategory", y="NaN_Percentage", fill="Category"))
        + geom_bar(stat="identity")
        + facet_wrap("~Category", scales="free_x", ncol=2)
        + theme(
            figure_size=(15, 10),
            axis_text_x=element_text(rotation=90),
            subplots_adjust={"hspace": 0.5},
        )
        + labs(x="Sub-category", y="Percentage (%)")
    )

    return p


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


def plot_sector_industry_stats(
    sector_stats: pd.DataFrame, industry_stats: pd.DataFrame
) -> None:
    """Plot sector and industry statistics using plotnine.

    Args:
        sector_stats: DataFrame containing sector statistics
        industry_stats: DataFrame containing industry statistics
    """

    # Helper function to create plot for a group of metrics
    def create_stats_plot(
        data: pd.DataFrame, group_col: str, metrics: list[str]
    ) -> None:
        # Melt the dataframe to get metrics in long format
        plot_data = data.melt(
            id_vars=[group_col],
            value_vars=metrics,
            var_name="Metric",
            value_name="Value",
        )

        # Create plot
        p = (
            ggplot(plot_data, aes(x=group_col, y="Value", fill=group_col))
            + geom_bar(stat="identity")
            + facet_wrap("~Metric", scales="free_y", ncol=5)
            + theme(
                figure_size=(25, 10),
                axis_text_x=element_text(rotation=90),
                subplots_adjust={"hspace": 0.5, "wspace": 0.3},
            )
            + labs(x=group_col, y="Value")
        )
        print(p)

    # Define metrics to plot
    base_metrics = [
        "Mean_RET_1",
        "Skewness_RET_1",
        "Kurtosis_RET_1",
        "Std_RET_1",
    ]
    # log_metrics = [
    #     "Log_Mean_RET_1",
    #     "Log_Skewness_RET_1",
    #     "Log_Kurtosis_RET_1",
    #     "Log_Std_RET_1",
    # ]

    # Plot sector statistics
    create_stats_plot(sector_stats, "SECTOR", base_metrics)
    # create_stats_plot(sector_stats, "SECTOR", log_metrics)

    # Plot industry statistics
    create_stats_plot(industry_stats, "INDUSTRY", base_metrics)
    # create_stats_plot(industry_stats, "INDUSTRY", log_metrics)


def plot_na(train_df):
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
    p = (
        ggplot(
            na_summary.reset_index(),
            aes(x="reorder(index, -Percentage)", y="Percentage"),
        )
        + geom_bar(stat="identity", fill="steelblue")
        + facet_wrap("~is_ret", scales="free")
        + theme(figure_size=(12, 6), axis_text_x=element_text(rotation=45, hjust=1))
        + labs(
            title="Missing Values Distribution",
            x="Variables",
            y="Percentage of Missing Values (%)",
        )
    )

    print(p)


def plot_model_accuracy(model_results):
    """Plot accuracy comparison of different models.

    Args:
        model_results (dict): Dictionary containing model names and their accuracy scores

    Returns:
        plotnine plot object showing model accuracy comparison
    """
    plot_df = pd.DataFrame(
        {"Model": model_results.keys(), "Accuracy": model_results.values()}
    )

    return (
        ggplot(
            plot_df.sort_values("Accuracy"),
            aes(x="reorder(Model, Accuracy)", y="Accuracy"),
        )
        + geom_bar(stat="identity")
        + geom_hline(yintercept=0.5131, color="red", linetype="dashed")
        + annotate(
            "text",
            x=0,
            y=0.5131,
            label="Benchmark (0.5131)",
            va="bottom",
            ha="left",
            color="red",
        )
        + theme(figure_size=(10, 6), axis_text_x=element_text(rotation=45, ha="right"))
        + labs(title="Model Accuracy Comparison", x="Model", y="Accuracy Score")
        + coord_cartesian(ylim=(0.5, plot_df["Accuracy"].max() * 1.1))
    )


# plot one row of data
def plot_ret_and_vol(train_df: pd.DataFrame, row_id: int = 24) -> ggplot:
    """Create a plot showing returns and volume for a given row.

    Args:
        train_df: DataFrame containing the training data
        row_id: Row ID to plot

    Returns:
        ggplot object with the visualization
    """
    # Prepare return data
    x_range = np.arange(20, 0, -1)
    return_cols = [f"RET_{n}" for n in x_range]
    volume_cols = [f"VOLUME_{n}" for n in x_range]

    # Create return data frame
    return_data = pd.DataFrame(
        {
            "Days": x_range,
            "Return": train_df[return_cols].loc[row_id].values,
            "Type": "Historical",
        }
    )

    # Add target point
    target_data = pd.DataFrame(
        {
            "Days": [0],
            "Return": [0.1 if train_df["RET"].loc[row_id] else -0.1],
            "Type": "Target",
        }
    )

    plot_data = pd.concat([return_data, target_data])

    # Create volume data
    volume_data = pd.DataFrame(
        {"Days": x_range, "Volume": train_df[volume_cols].loc[row_id].values / 10}
    )
    # Create plot
    p = (
        ggplot()
        +
        # Return bars
        geom_bar(
            data=plot_data,
            mapping=aes(
                x="-Days", y="Return", fill="Type"
            ),  # Negate Days to invert x-axis
            stat="identity",
            alpha=0.7,
        )
        +
        # Volume line
        geom_line(
            data=volume_data,
            mapping=aes(x="-Days", y="Volume", group=1),  # Negate Days to invert x-axis
            color="teal",
            size=1,
        )
        + theme(figure_size=(10, 6))
        + labs(x="Days to", y="Return / Volume (Volume scaled down by 10)")
        + scale_x_continuous(labels=lambda x: [str(abs(int(i))) for i in x])
    )

    return p


# %%
def plot_correlation_matrix(df):
    """
    Create and plot a correlation matrix heatmap for the given dataframe.

    Args:
        df (pd.DataFrame): Input dataframe to compute correlations for

    Returns:
        None: Displays the correlation heatmap plot
    """
    # compute the correlation matrix
    corr_matrix = df.drop(
        columns=[
            ID_COLS,
        ],
        errors="ignore",
    ).corr()

    # Convert correlation matrix to long format for plotnine
    corr_df = corr_matrix.reset_index().melt(id_vars="index")
    corr_df.columns = ["Var1", "Var2", "value"]

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
    corr_df = corr_df[~mask.ravel()]

    # Create heatmap with plotnine
    p = (
        ggplot(corr_df, aes("Var1", "Var2", fill="value"))
        + geom_tile()
        + scale_fill_gradient2(
            low="brown", mid="white", high="green", limits=[-0.35, 0.35], midpoint=0
        )
        + theme(
            figure_size=(12, 10),
            axis_text_x=element_text(rotation=90, size=9),
            axis_text_y=element_text(size=9),
        )
        + coord_equal()
        + labs(title="Correlation Matrix")
    )
    print(p)


# simulation parameters
def simulate_strategy(y_test, y_predict, n_simulations=1000, n_days=100):
    """
    Simulate trading strategy performance based on model predictions.

    Args:
        y_test: True test labels
        y_predict: Model predictions
        n_simulations: Number of simulations to run
        n_days: Number of days to simulate

    Returns:
        np.array: Average cumulative strategy returns across simulations
    """

    # get accuracy
    nn_accuracy = accuracy_score(y_test, y_predict)

    # initialize strategy results
    nn_strategy = np.zeros(n_days)

    # simulate
    for i in range(n_simulations):
        tmp_nn_strategy = []
        for d in range(n_days):
            tmp_nn_strategy.append(+1 if nn_accuracy > np.random.random() else -1)
        # cumulative sum
        nn_strategy += np.cumsum(tmp_nn_strategy)

    # compute average performance
    nn_strategy /= n_simulations

    plot_data = pd.DataFrame(
        {
            "Days": range(n_days),
            "NN": nn_strategy,
        }
    ).melt(id_vars=["Days"], var_name="Model", value_name="PnL")

    # Add accuracy values to labels
    plot_data["Label"] = plot_data["Model"].map(
        {
            "NN": f"NN, acc = {nn_accuracy:1.4f}",
        }
    )

    # Create plot
    p = (
        ggplot(plot_data, aes(x="Days", y="PnL", color="Label"))
        + geom_line(size=1.2, alpha=0.8)
        + scale_color_manual(values=["firebrick", "seagreen", "darkcyan"])
        + theme(figure_size=(10, 6))
        + labs(
            title="Avg Performance of Long-Short Strategy Simulation",
            x="Days",
            y="PnL",
            color="Model",
        )
    )

    print(p)
