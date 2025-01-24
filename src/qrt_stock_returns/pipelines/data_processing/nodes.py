import logging as log
from pathlib import Path

import numpy as np
import pandas as pd
from kedro.config import OmegaConfigLoader
from pyspark.sql import DataFrame as SparkDataFrame
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ta_indicators import calculate_all_ta_indicators

conf_loader = OmegaConfigLoader(".", base_env="", default_run_env="")
# Read the configuration file
conf_params = conf_loader["parameters"]


def load_data(
    x_train: str | Path | pd.DataFrame,
    y_train: str | Path | pd.DataFrame,
    x_test: str | Path | pd.DataFrame,
    sample_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test data from files or DataFrames and optionally sample.

    Args:
        x_train: Training features data source
        y_train: Training target data source
        x_test: Test features data source
        sample_n: Optional number of samples to take from training data

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Raw training and test dataframes
    """
    # Load data
    if isinstance(x_train, str | Path):
        x_train: pd.DataFrame = pd.read_csv(x_train)
    if isinstance(y_train, str | Path):
        y_train: pd.DataFrame = pd.read_csv(y_train)
    train_df: pd.DataFrame = pd.concat([x_train, y_train], axis=1)
    if isinstance(x_test, str | Path):
        test_df: pd.DataFrame = pd.read_csv(x_test)
    else:
        test_df: pd.DataFrame = x_test

    # Sample training data if requested
    if sample_n is not None:
        train_df = train_df.sample(n=sample_n, random_state=42)
        test_df = test_df.sample(n=sample_n, random_state=42)
        log.debug(f"Sampled {sample_n} rows from train_df and test_df")

    return train_df, test_df


def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    remove_id_cols: bool = False,
    n_days: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Args:
        train_df: Raw training dataframe
        test_df: Raw test dataframe
        remove_id_cols: Whether to remove ID columns
        n_days: Number of most recent days to check for missing returns

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Preprocessed training and test dataframes
    """
    # Drop rows where all recent return features are missing
    return_features = [f"RET_{day}" for day in range(1, n_days + 1)]
    missing_prop = (
        train_df[return_features].isna().sum(axis=1)
        / train_df[return_features].shape[1]
    )
    train_df = train_df[missing_prop < 1].copy()

    # Clean data and drop NA values
    n_before = len(train_df)
    train_df = train_df.dropna()
    n_after = len(train_df)
    log.debug(f"Dropped {n_before - n_after} rows with NA values from train_df")

    # Set index and remove duplicates
    train_df = train_df.loc[:, ~train_df.columns.duplicated()]
    train_df = train_df.set_index("ID")
    test_df = test_df.loc[:, ~test_df.columns.duplicated()]
    test_df = test_df.set_index("ID")
    log.debug("Set index to ID column for train_df and test_df")

    # Drop ID columns if requested
    if remove_id_cols:
        train_df = train_df.drop(
            conf_params["raw_data"]["id_cols"], axis=1, errors="ignore"
        )
        test_df = test_df.drop(
            conf_params["raw_data"]["id_cols"], axis=1, errors="ignore"
        )
        log.debug(f"Dropped ID columns {conf_params['raw_data']['id_cols']}")

    # Convert target to binary
    sign_of_return: LabelEncoder = LabelEncoder()
    train_df["RET"] = sign_of_return.fit_transform(train_df["RET"])

    return train_df, test_df


def process_and_save_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ta_indicators_df: pd.DataFrame,
    test_ta_indicators_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process and save/load training and test data with technical indicators.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        ta_indicators_df: Technical indicators for training data
        test_ta_indicators_df: Technical indicators for test data
        train_df_path: Path to save/load processed training data
        test_df_path: Path to save/load processed test data

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Processed training and test dataframes
    """
    # Concatenate technical indicators
    train_df = pd.concat([train_df, ta_indicators_df], axis=1)
    test_df = pd.concat([test_df, test_ta_indicators_df], axis=1)

    return train_df, test_df


def create_model_input_table(
    shuttles: SparkDataFrame, companies: SparkDataFrame, reviews: SparkDataFrame
) -> SparkDataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    # Rename columns to prevent duplicates
    shuttles = shuttles.withColumnRenamed("id", "shuttle_id")
    companies = companies.withColumnRenamed("id", "company_id")

    rated_shuttles = shuttles.join(reviews, "shuttle_id", how="left")
    model_input_table = rated_shuttles.join(companies, "company_id", how="left")
    model_input_table = model_input_table.dropna()
    return model_input_table


def drop_id_cols(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_df.drop(
        conf_params["raw_data"]["id_cols"], axis=1, errors="ignore"
    ), test_df.drop(conf_params["raw_data"]["id_cols"], axis=1, errors="ignore")


def drop_obsolete_technical_indicators(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop technical indicators that are not useful for prediction.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        target: Target column name

    Returns:
        tuple containing:
        - Filtered training dataframe
        - List of remaining features
    """
    import re

    # Find columns to drop based on pattern matching
    cols_to_drop = [
        col
        for col in train_df.columns
        if re.search(
            r"\D(?:([1-2]{1}[0-9])|([8-9]{1})\_)",
            str(col),
        )
        and not col.startswith(("RET", "VOLUME"))  # don't drop RET and VOLUME
    ]

    # Drop identified columns
    train_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    test_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Get remaining features
    features = [col for col in train_df.columns if col != target]

    return train_df, test_df, features


def remove_duplicated_columns(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Remove duplicated columns from train and test datasets.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame

    Returns:
        Tuple containing:
        - Filtered train DataFrame
        - Filtered test DataFrame
        - Updated features list with duplicates removed
    """
    # Find duplicated features using set operations
    # Find duplicates in train dataset
    train_duplicates = train_df.columns[train_df.columns.duplicated()].tolist()

    # Find duplicates in test dataset
    test_duplicates = test_df.columns[test_df.columns.duplicated()].tolist()
    duplicated_cols = list(set(train_duplicates) | set(test_duplicates))
    # If no duplicates found, return original data
    if not duplicated_cols:
        return train_df, test_df, []

    # Remove duplicated columns from train and test dataframes
    train_filtered = train_df.loc[:, ~train_df.columns.duplicated()]
    test_filtered = test_df.loc[:, ~test_df.columns.duplicated()]

    return train_filtered, test_filtered, duplicated_cols


def filter_infinity_values(
    train: pd.DataFrame, test: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Check for infinity values in train and test datasets and filter out features containing infinities.

    Args:
        train: Training DataFrame
        test: Test DataFrame
        features: List of feature names to check
        target: Target column name

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Filtered train and test DataFrames with infinity values removed
    """
    # Check for infinity values in the train dataset
    cols = list(set(train.columns) | set(test.columns))

    # Get columns with infinity values in train dataset
    def _get_inf_cols(df: pd.DataFrame) -> pd.Index:
        return df.columns[df.isin([np.inf, -np.inf]).any()]

    inf_train = _get_inf_cols(train)
    inf_test = _get_inf_cols(test)
    inf_cols = list(set(inf_train) | set(inf_test))

    # Filter out features containing infinities from train and test
    filt_feats = [col for col in cols if col not in inf_cols]
    log.debug(f"Features after filtering infinities: {filt_feats + [target]}")
    # Filter to only include columns that exist in both dataframes
    train_feats = [f for f in filt_feats if f in train.columns]
    test_feats = [f for f in filt_feats if f in test.columns]
    if set(train_feats) != set(test_feats):
        log.warning("Train and test datasets have different available features")

    train_feats = train_feats + (target if isinstance(target, list) else [target])
    train_filt = train[train_feats]
    test_filt = test[test_feats]

    return train_filt, test_filt, filt_feats


def calculate_technical_indicators(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate or load technical indicators for train and test data.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Technical indicators for train and test data
    """
    log.info("Calculating technical indicators for train and test data")
    ta_indicators_df = calculate_all_ta_indicators(train_df, features)
    test_ta_indicators_df = calculate_all_ta_indicators(test_df, features)
    train_df = pd.concat([train_df, ta_indicators_df], axis=1)
    test_df = pd.concat([test_df, test_ta_indicators_df], axis=1)

    return train_df, test_df


def calculate_statistical_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Calculate statistical features by grouping data on different dimensions.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, list]: A tuple containing:
            - Modified training dataframe with new features
            - Modified test dataframe with new features
            - List of new feature names
    """
    new_features = []
    shifts = [1, 2, 3, 4, 5]  # Choose some different shifts
    statistics = ["median", "std"]  # the type of stat
    gb_features = [
        ["DATE", "INDUSTRY"],
        ["DATE", "INDUSTRY_GROUP"],
        ["DATE", "SECTOR"],
        ["DATE", "SUB_INDUSTRY"],
    ]
    target_feature = "RET"

    # Create a name by joining the last element of each gb_feature list
    for gb_feature in gb_features:
        for shift in shifts:
            for stat in statistics:
                name = f"{target_feature}_{shift}_{gb_feature[-1]}_{stat}"
                feat = f"{target_feature}_{shift}"
                new_features.append(name)
                for data in [train_df, test_df]:
                    data[name] = data.groupby(gb_feature)[feat].transform(stat)

    return train_df, test_df, new_features


def remove_nan_rows(train_df: pd.DataFrame) -> tuple:
    """Remove rows containing NaN values from training data.

    Args:
        train_df: Training dataframe

    Returns:
        Tuple containing cleaned X_train, X_test, y_train, y_test
    """
    nan_mask = train_df.isna().any(axis=1)
    train_clean = train_df[~nan_mask]

    return train_clean


def remove_duplicates_and_nans(
    train_df_filtered: pd.DataFrame,
    test_df_filtered: pd.DataFrame,
    imputing_method: str = "knn",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove duplicated columns and NaN values from filtered datasets.

    Args:
        train_df_filtered: Filtered training DataFrame
        test_df_filtered: Filtered test DataFrame
        y_train: Training target
        y_test: Test target

    Returns:
        Tuple containing:
        - Cleaned train DataFrame
        - Cleaned test DataFrame
    """
    # Remove duplicated columns
    train_filt = train_df_filtered.loc[:, ~train_df_filtered.columns.duplicated()]
    test_filt = test_df_filtered.loc[:, ~test_df_filtered.columns.duplicated()]

    # Then handle NaN values
    nan_mask = train_filt.isna().any(axis=1)
    train_clean = train_filt[~nan_mask]

    # Dictionary mapping imputing methods to their implementations
    imputing_methods = {
        "knn": lambda df: pd.DataFrame(
            KNNImputer(n_neighbors=5).fit_transform(df),
            columns=df.columns,
            index=df.index,
        ),
        "mean": lambda df: df.fillna(df.mean()),
        "median": lambda df: df.fillna(df.median()),
        "interpolate": lambda df: df.interpolate(),
        "iterative": lambda df: pd.DataFrame(
            IterativeImputer(
                estimator=HistGradientBoostingRegressor(),  # it can handle NaAns
                max_iter=10,
                random_state=0,
            ).fit_transform(df),
            columns=df.columns,
            index=df.index,
        ),
    }

    # Get the imputing function or raise error if invalid method
    impute_func = imputing_methods.get(imputing_method)
    if not impute_func:
        raise ValueError(f"Invalid imputing method: {imputing_method}")

    # Apply the selected imputing method
    test_clean = impute_func(test_filt)

    return train_clean, test_clean


def remove_collinear_features(df, threshold=0.95, mi_threshold=0.001, vif_threshold=10):
    """
    Remove collinear features from a dataframe based on correlation threshold, mutual information scores,
    and variance inflation factor (VIF). Keeps features with highest predictive power.
    """
    # Calculate correlation matrix
    correlation_matrix = df.corr().abs()
    upper = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    # Calculate mutual information scores
    mi_scores = mutual_info_regression(df, df["RET"])
    mi_scores = pd.Series(mi_scores, index=df.columns)

    # Calculate VIF scores
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]

    features_to_drop = []

    # Drop features with low MI scores
    low_mi_features = mi_scores[mi_scores < mi_threshold].index.tolist()
    features_to_drop.extend(low_mi_features)

    # Drop features with high VIF
    high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]["Feature"].tolist()
    features_to_drop.extend(high_vif_features)

    # Handle correlated features
    for column in upper.columns:
        highly_correlated = upper[column][upper[column] > threshold].index.tolist()
        if highly_correlated:
            # Keep feature with highest MI score
            corr_features_mi = mi_scores[highly_correlated + [column]]
            features_to_drop.extend(
                [f for f in corr_features_mi.index if f != corr_features_mi.idxmax()]
            )

    # Remove duplicates from drop list
    features_to_drop = list(set(features_to_drop))

    log.debug(
        f"Dropping {len(features_to_drop)} features based on correlation, MI, and VIF: {features_to_drop}"
    )
    return df.drop(columns=features_to_drop)
