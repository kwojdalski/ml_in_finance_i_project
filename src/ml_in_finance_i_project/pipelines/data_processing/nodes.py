import logging as log
from pathlib import Path

import numpy as np
import pandas as pd
from kedro.config import OmegaConfigLoader
from pyspark.sql import DataFrame as SparkDataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ta_indicators import calculate_all_ta_indicators

conf_loader = OmegaConfigLoader(".", base_env="", default_run_env="")
# Read the configuration file
conf_params = conf_loader["parameters"]


def load_data(
    x_train: str | Path | pd.DataFrame,
    y_train: str | Path | pd.DataFrame,
    x_test: str | Path | pd.DataFrame,
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
    if isinstance(x_train, str | Path):
        x_train: pd.DataFrame = pd.read_csv(x_train)
    if isinstance(y_train, str | Path):
        y_train: pd.DataFrame = pd.read_csv(y_train)
    train_df: pd.DataFrame = pd.concat([x_train, y_train], axis=1)
    if isinstance(x_test, str | Path):
        test_df: pd.DataFrame = pd.read_csv(x_test)
    else:
        test_df: pd.DataFrame = x_test

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
        train_df = train_df.drop(conf_params["raw_data"]["id_cols"], axis=1)
        log.debug(
            f"Dropped ID columns {conf_params['raw_data']['id_cols']} from train_df"
        )

    n_before = len(test_df)
    test_df = test_df.dropna()
    n_after = len(test_df)
    log.debug(f"Dropped {n_before - n_after} rows with NA values from test_df")

    if remove_id_cols:
        test_df = test_df.drop(conf_params["raw_data"]["id_cols"], axis=1)
        log.debug(
            f"Dropped ID columns {conf_params['raw_data']['id_cols']} from test_df"
        )

    # Convert target to binary
    sign_of_return: LabelEncoder = LabelEncoder()
    train_df["RET"] = sign_of_return.fit_transform(train_df["RET"])

    # Sample training data if sample_n is provided
    if sample_n is not None:
        train_df = train_df.sample(n=sample_n, random_state=42)
        log.debug(f"Sampled {sample_n} rows from train_df")

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
    return train_df.drop(conf_params["raw_data"]["id_cols"], axis=1), test_df.drop(
        conf_params["raw_data"]["id_cols"], axis=1
    )


def drop_obsolete_technical_indicators(
    train_df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, list[str]]:
    """Drop technical indicators that are not useful for prediction.

    Args:
        train_df: Training dataframe
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

    # Get remaining features
    features = [col for col in train_df.columns if col != target]

    return train_df, features


def remove_duplicated_columns(
    train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Remove duplicated columns from train and test datasets.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        features: List of feature names

    Returns:
        Tuple containing:
        - Filtered train DataFrame
        - Filtered test DataFrame
        - Updated features list with duplicates removed
    """
    # Find duplicated column names in train dataset
    duplicated_cols = []
    feature_names = set()
    for feature in features:
        if feature in feature_names:
            duplicated_cols.append(feature)
        else:
            feature_names.add(feature)

    # Remove duplicated columns
    if duplicated_cols:
        # Only drop duplicated columns that exist in each df
        train_duplicates = [col for col in duplicated_cols if col in train_df.columns]
        test_duplicates = [col for col in duplicated_cols if col in test_df.columns]
        filtered_features = [col for col in features if col not in duplicated_cols]
        train_filtered = train_df.drop(columns=train_duplicates)
        test_filtered = test_df.drop(columns=test_duplicates)
        return train_filtered, test_filtered, filtered_features

    return train_df, test_df, features


def filter_infinity_values(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str
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
    inf_counts_train = np.isinf(train).sum()
    inf_columns_train = inf_counts_train[inf_counts_train > 0]
    inf_columns_train_list = inf_columns_train.index.tolist()

    # Check for infinity values in the test dataset
    inf_counts_test = np.isinf(test).sum()
    inf_columns_test = inf_counts_test[inf_counts_test > 0]
    inf_columns_test_list = inf_columns_test.index.tolist()

    # Combine the lists for further use if needed
    inf_columns_combined_list = list(
        set(inf_columns_train_list + inf_columns_test_list)
    )

    # Filter out features containing infinities from train and test
    filtered_features = [
        col for col in features if col not in inf_columns_combined_list
    ]
    log.debug(f"Features after filtering infinities: {filtered_features + [target]}")
    # Filter to only include columns that exist in both dataframes
    train_available_features = [f for f in filtered_features if f in train.columns]
    test_available_features = [f for f in filtered_features if f in test.columns]
    if set(train_available_features) != set(test_available_features):
        log.warning("Train and test datasets have different available features")
    train_filtered = train[train_available_features + [target]]
    test_filtered = test[test_available_features]

    return train_filtered, test_filtered, filtered_features


def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["RET"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return X_train, X_test, y_train, y_test


def calculate_technical_indicators(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate or load technical indicators for train and test data.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Technical indicators for train and test data
    """
    log.info("Calculating technical indicators for train and test data")
    ta_indicators_df = calculate_all_ta_indicators(train_df)
    test_ta_indicators_df = calculate_all_ta_indicators(test_df)
    train_df = pd.concat([train_df, ta_indicators_df], axis=1)
    test_df = pd.concat([test_df, test_ta_indicators_df], axis=1)

    return ta_indicators_df, test_ta_indicators_df


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
