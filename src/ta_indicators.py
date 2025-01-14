import logging as log
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import talib
from scipy.stats import skew

# def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
#     """Calculate various technical indicators using TA-Lib."""

#     # Ensure we have OHLCV data in the expected format
#     high = df["high"].values
#     low = df["low"].values
#     close = df["close"].values
#     volume = df["volume"].values

#     # Volatility Measures
#     # Rolling Standard Deviation (20-day volatility)
#     df["rolling_std_20"] = talib.STDDEV(close, timeperiod=20)

#     # Volatility of Volatility (20-day rolling std of 20-day volatility)
#     df["vol_of_vol"] = talib.STDDEV(df["rolling_std_20"].values, timeperiod=20)

#     # Technical Indicators
#     # Money Flow Index (MFI)
#     # df["mfi"] = talib.MFI(high, low, close, volume, timeperiod=14)

#     # Relative Strength Index (RSI)
#     df["rsi"] = talib.RSI(close, timeperiod=14)

#     # Accumulation/Distribution Line (ADL)
#     # df["adl"] = talib.AD(high, low, close, volume)

#     # Average True Range (ATR)
#     # df["atr"] = talib.ATR(high, low, close, timeperiod=14)

#     # MACD
#     df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
#         close, fastperiod=12, slowperiod=26, signalperiod=9
#     )

#     return df


def calculate_rsi(
    data: pd.DataFrame, window: int = 14
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate RSI for each stock.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'RET_i' columns (i from 1 to 20).
    window: int, the lookback period for RSI calculation.

    Returns:
    Tuple containing:
        - pd.DataFrame with RSI values for each stock
        - List[str] of new feature names added
    """
    data = data.copy()

    # Calculate average gains and losses over the window period
    avg_gain = (
        data.groupby(["STOCK", "DATE"])[[f"RET_{day}" for day in range(1, window + 1)]]
        .mean()
        .agg(lambda x: x[x > 0].mean(), axis=1)
    )
    avg_loss = (
        data.groupby(["STOCK", "DATE"])[[f"RET_{day}" for day in range(1, window + 1)]]
        .mean()
        .agg(lambda x: x[x < 0].mean(), axis=1)
        .abs()
    )

    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    # Convert RSI Series to DataFrame
    rsi_df = rsi.reset_index()
    rsi_df.rename(columns={0: "RSI"}, inplace=True)

    # Join RSI values back to the original dataframe
    data = data.merge(rsi_df, on=["STOCK", "DATE"], how="left")

    # Fill NaNs in RSI with the median RSI value for each stock
    data["RSI"] = data.groupby("STOCK")["RSI"].transform(lambda x: x.fillna(x.median()))

    # Add overbought and oversold signals
    data["overbought_rsi"] = np.where(data["RSI"] > 70, 1, 0)
    data["oversold_rsi"] = np.where(data["RSI"] < 30, 1, 0)

    # List of new features added
    new_features = ["RSI", "overbought_rsi", "oversold_rsi"]

    return data, new_features


def calculate_rsi_per_sector(
    data: pd.DataFrame, window: int = 14
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate RSI for each sector.

    Args:
    data: pd.DataFrame, containing 'SECTOR', 'DATE', and 'RET_i' columns (i from 1 to 20).
    window: int, the lookback period for RSI calculation.

    Returns:
    Tuple containing:
        - pd.DataFrame with RSI values for each sector
        - List[str] of new feature names added
    """
    data = data.copy()

    # Calculate average gains and losses over the window period
    avg_gain = (
        data.groupby(["SECTOR", "DATE"])[[f"RET_{day}" for day in range(1, window + 1)]]
        .mean()
        .agg(lambda x: x[x > 0].mean(), axis=1)
    )
    avg_loss = (
        data.groupby(["SECTOR", "DATE"])[[f"RET_{day}" for day in range(1, window + 1)]]
        .mean()
        .agg(lambda x: x[x < 0].mean(), axis=1)
        .abs()
    )

    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi_sector = 100 - (100 / (1 + rs))

    # Convert RSI Series to DataFrame
    rsi_sector_df = rsi_sector.reset_index()
    rsi_sector_df.rename(columns={0: "RSI_SECTOR"}, inplace=True)

    # Join RSI values back to the original dataframe
    data = data.merge(rsi_sector_df, on=["SECTOR", "DATE"], how="left")

    # Add overbought and oversold signals
    data["overbought_rsi_sector"] = np.where(data["RSI_SECTOR"] > 70, 1, 0)
    data["oversold_rsi_sector"] = np.where(data["RSI_SECTOR"] < 30, 1, 0)

    # List of new features added
    new_features = ["RSI_SECTOR", "overbought_rsi_sector", "oversold_rsi_sector"]

    return data, new_features


def calculate_roc_past_rows(
    data: pd.DataFrame, window: int = 12
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate ROC for each stock for the columns RET_1 to RET_5 over a rolling window of past rows.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'RET_i' columns (i from 1 to 5).
    window: int, the lookback period for ROC calculation.

    Returns:
    pd.DataFrame with ROC values for the columns RET_1 to RET_5 for each stock.
    List[str] of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in [1, 5, 14]:
        ret_col = f"RET_{i}"
        roc_col = f"ROC_{i}"
        new_features.append(f"ROC_{i}")

        # Calculate ROC over past rows for each stock and date
        data[roc_col] = data.groupby(["STOCK"])[ret_col].transform(
            lambda x: 100 * (1 + x).pct_change(periods=window) * 100
        )

        # Fill NaNs with the median of the ROC values for each stock
        data[roc_col] = data.groupby("STOCK")[roc_col].transform(
            lambda x: x.fillna(x.median())
        )

        # Fillna for those stocks which have only 1 observation
        data[roc_col] = data.groupby("SECTOR")[roc_col].transform(
            lambda x: x.fillna(x.median())
        )

    return data, new_features


def calculate_momentum(
    data: pd.DataFrame, window: int = 10
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate Momentum for both RET and VOLUME for each stock encapsulating the past 20 days.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', 'RET_i' columns (i from 1 to 20),
    and 'VOLUME_i' columns (i from 1 to 20).
    window: int, the lookback period for Momentum calculation (default is 10).

    Returns:
    pd.DataFrame with Momentum values for RET and VOLUME for each stock.
    List[str] of new feature names.
    """
    data = data.copy()
    new_features = []

    for target in ["RET", "VOLUME"]:
        window_size = window
        momentum_col_name = f"{target}_{window_size}_day_momentum"
        new_features.append(momentum_col_name)

        # Calculate the rolling mean of target columns and mean of the first day
        rolling_mean_target = data.groupby(by=["STOCK", "DATE"])[
            [f"{target}_{day}" for day in range(2, window_size + 1)]
        ].mean()
        target_1_mean = data.groupby(by=["STOCK", "DATE"])[[f"{target}_1"]].mean()

        # Align the data for subtraction
        current_value_aligned, rolling_mean_value_aligned = target_1_mean.align(
            rolling_mean_target, axis=0, level="STOCK"
        )
        momentum_value = current_value_aligned.sub(
            rolling_mean_value_aligned.mean(axis=1), axis=0
        )

        # Rename the column to indicate momentum
        momentum_value.rename(columns={f"{target}_1": momentum_col_name}, inplace=True)

        # Join the momentum back to the original data
        placeholder = data.join(momentum_value, on=["STOCK", "DATE"], how="left")
        data[momentum_col_name] = placeholder[momentum_col_name]

    return data, new_features


def calculate_momentum_sector(
    data: pd.DataFrame, window: int = 10
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate Momentum for both RET and VOLUME for each sector encapsulating the past 20 days.

    Args:
    data: pd.DataFrame, containing 'SECTOR', 'DATE', 'RET_i' columns (i from 1 to 20),
    and 'VOLUME_i' columns (i from 1 to 20).
    window: int, the lookback period for Momentum calculation (default is 10).

    Returns:
    pd.DataFrame with Momentum values for RET and VOLUME for each sector.
    List[str] of new feature names.
    """
    data = data.copy()
    new_features = []

    for target in ["RET", "VOLUME"]:
        window_size = window
        momentum_col_name = f"{target}_{window_size}_day_momentum_sector"
        new_features.append(momentum_col_name)

        # Calculate the rolling mean of target columns and mean of the first day
        rolling_mean_target = data.groupby(by=["SECTOR", "DATE"])[
            [f"{target}_{day}" for day in range(2, window_size + 1)]
        ].mean()
        target_1_mean = data.groupby(by=["SECTOR", "DATE"])[[f"{target}_1"]].mean()

        # Align the data for subtraction
        current_value_aligned, rolling_mean_value_aligned = target_1_mean.align(
            rolling_mean_target, axis=0, level="SECTOR"
        )
        momentum_value = current_value_aligned.sub(
            rolling_mean_value_aligned.mean(axis=1), axis=0
        )

        # Rename the column to indicate momentum
        momentum_value.rename(columns={f"{target}_1": momentum_col_name}, inplace=True)

        # Join the momentum back to the original data
        placeholder = data.join(momentum_value, on=["SECTOR", "DATE"], how="left")
        data[momentum_col_name] = placeholder[momentum_col_name]

    return data, new_features


# Function to calculate stochastic oscillator
def calculate_stochastic_oscillator(
    data: pd.DataFrame, window: int = 14, smooth_window: int = 3
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate Stochastic Oscillator for each stock using RET_1, RET_3, RET_5, RET_10, and RET_20.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'RET_i' columns (i from 1 to 20).
    window: int, the lookback period for Stochastic Oscillator calculation.
    smooth_window: int, the smoothing window for %D calculation.

    Returns:
    pd.DataFrame with %K and %D values for each stock.
    List[str] of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in [1, 5, 14]:
        ret_col = f"RET_{i}"
        low_col = f"L14_{i}"
        high_col = f"H14_{i}"
        k_col = f"%K_{i}"
        d_col = f"%D_{i}"
        new_features.extend([k_col, d_col])

        # Calculate the lowest low and highest high over the lookback period
        data[low_col] = data.groupby("STOCK")[ret_col].transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
        data[high_col] = data.groupby("STOCK")[ret_col].transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )

        # Calculate %K
        data[k_col] = (
            (data[ret_col] - data[low_col]) / (data[high_col] - data[low_col])
        ) * 100

        # Calculate %D (SMA of %K)
        data[d_col] = data.groupby("STOCK")[k_col].transform(
            lambda x: x.rolling(smooth_window, min_periods=1).mean()
        )

        # Fill NaNs with the median for each stock group
        data[[k_col, d_col]] = data.groupby("STOCK")[[k_col, d_col]].transform(
            lambda x: x.fillna(x.median())
        )

        # Fill NaNs with the median for each sector group
        data[[k_col, d_col]] = data.groupby("SECTOR")[[k_col, d_col]].transform(
            lambda x: x.fillna(x.median())
        )

    return data, new_features


# Function to calculate MACD
def calculate_macd(
    data: pd.DataFrame,
    short_window: int = 12,
    long_window: int = 26,
    signal_window: int = 9,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate MACD and MACD Divergence for each stock for RET_1 to RET_5.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'RET_i' columns (i from 1 to 5).
    short_window: int, the short window period for EMA calculation.
    long_window: int, the long window period for EMA calculation.
    signal_window: int, the signal line period for EMA calculation.

    Returns:
    pd.DataFrame with MACD, Signal Line, and MACD Divergence for each stock for RET_1 to RET_5.
    List[str] of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in [1, 5, 14]:
        ret_col = f"RET_{i}"
        ema_12_col = f"EMA_12_{i}"
        ema_26_col = f"EMA_26_{i}"
        macd_col = f"MACD_{i}"
        signal_line_col = f"Signal_Line_{i}"
        macd_divergence_col = f"MACD_Divergence_{i}"

        new_features.extend([macd_col, signal_line_col, macd_divergence_col])

        # Calculate short-term EMA
        data[ema_12_col] = data.groupby("STOCK")[ret_col].transform(
            lambda x: x.ewm(span=short_window, adjust=False).mean()
        )

        # Calculate long-term EMA
        data[ema_26_col] = data.groupby("STOCK")[ret_col].transform(
            lambda x: x.ewm(span=long_window, adjust=False).mean()
        )

        # Calculate MACD
        data[macd_col] = data[ema_12_col] - data[ema_26_col]

        # Calculate Signal Line
        data[signal_line_col] = data.groupby("STOCK")[macd_col].transform(
            lambda x: x.ewm(span=signal_window, adjust=False).mean()
        )

        # Calculate MACD Divergence
        data[macd_divergence_col] = data[macd_col] - data[signal_line_col]

    return data, new_features


# Function to calculate golden cross
def calculate_golden_cross(
    data: pd.DataFrame, short_window: int = 10, long_window: int = 200
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate Golden Cross for each stock.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'Close' columns.
    short_window: int, the short window period for moving average calculation.
    long_window: int, the long window period for moving average calculation.

    Returns:
    pd.DataFrame with Golden Cross binary variable for each stock.
    List[str] of new feature names.
    """
    data = data.copy()

    # Calculate short-term SMA
    data[f"SMA_{short_window}_1"] = data.groupby(["STOCK"])["RET_1"].transform(
        lambda x: x.rolling(window=short_window, min_periods=1).mean()
    )

    # Calculate long-term SMA
    data[f"SMA_{long_window}_1"] = data.groupby(["STOCK"])["RET_1"].transform(
        lambda x: x.rolling(window=long_window, min_periods=1).mean()
    )

    # Determine Golden Cross
    data["Golden_Cross_1"] = np.where(
        data[f"SMA_{short_window}_1"] > data[f"SMA_{long_window}_1"], 1, 0
    )

    features = ["Golden_Cross_1"]

    return data, features


# Function to calculate conditional aggregated features
def calculate_conditional_aggregated_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    shifts: List[int] = [1, 2, 3, 4],
    statistics: List[str] = ["mean"],
    gb_features_list: List[List[str]] = [["SECTOR", "DATE"]],
    target_features: List[str] = ["RET"],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Calculate conditional aggregated features for the given shifts, statistics, and group-by features.

    Args:
    train: pd.DataFrame, training data containing target features and group-by features.
    test: pd.DataFrame, testing data containing target features and group-by features.
    shifts: list, list of shifts to calculate (default: [1, 2, 3, 4]).
    statistics: list, list of statistics to calculate (default: ['mean']).
    gb_features_list: list, list of group-by feature lists (default: [['SECTOR', 'DATE']]).
    target_features: list, list of target features to calculate (default: ['RET']).

    Returns:
    pd.DataFrame for train and test with added conditional aggregated features.
    List[str] of new feature names.
    """
    new_features = []

    for target_feature in target_features:
        for gb_features in gb_features_list:
            tmp_name = "_".join(gb_features)
            for shift in shifts:
                for stat in statistics:
                    name = f"{target_feature}_{shift}_{tmp_name}_{stat}"
                    feat = f"{target_feature}_{shift}"
                    new_features.append(name)
                    for data in [train, test]:
                        data[name] = data.groupby(gb_features)[feat].transform(stat)

    return train, test, new_features


# Function to compute statistical features
def compute_statistical_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    periods: int = 4,
    statistics: List[str] = ["mean", "std"],
    target_features: List[str] = ["RET", "VOLUME"],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Compute statistical features for the given target columns over specified periods.

    Args:
    train: pd.DataFrame, training data containing target columns.
    test: pd.DataFrame, testing data containing target columns.
    periods: int, number of periods to calculate (default: 4).
    statistics: list, list of statistics to calculate (default: ['mean', 'std']).
    target_features: list, list of target columns to calculate (default: ['RET', 'VOLUME']).

    Returns:
    pd.DataFrame for train and test with added statistical features.
    List[str] of new feature names.
    """
    new_features = []

    for target_feature in target_features:
        for stat in statistics:
            for period in range(periods):
                feature_name = f"{stat}_{target_feature}_STOCK_PERIOD_{period+1}"
                new_features.append(feature_name)
                for data in [train, test]:
                    if stat == "mean":
                        if target_feature == "VOLUME":
                            data[feature_name] = (
                                data[
                                    [
                                        f"{target_feature}_{period*5 + day}"
                                        for day in range(1, 6)
                                    ]
                                ]
                                .mean(axis=1)
                                .abs()
                            )
                        else:
                            data[feature_name] = data[
                                [
                                    f"{target_feature}_{period*5 + day}"
                                    for day in range(1, 6)
                                ]
                            ].mean(axis=1)
                    elif stat == "std":
                        data[feature_name] = data[
                            [
                                f"{target_feature}_{period*5 + day}"
                                for day in range(1, 6)
                            ]
                        ].std(axis=1)
                    elif stat == "min":
                        data[feature_name] = data[
                            [
                                f"{target_feature}_{period*5 + day}"
                                for day in range(1, 6)
                            ]
                        ].min(axis=1)
                    elif stat == "max":
                        data[feature_name] = data[
                            [
                                f"{target_feature}_{period*5 + day}"
                                for day in range(1, 6)
                            ]
                        ].max(axis=1)
                    elif stat == "median":
                        data[feature_name] = data[
                            [
                                f"{target_feature}_{period*5 + day}"
                                for day in range(1, 6)
                            ]
                        ].median(axis=1)

    return train, test, new_features


def fillna_with_sector_median(train: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Fill NaN values in the specified features with the median value of the SECTOR.

    Args:
    train: pd.DataFrame, training data containing features.
    features: list, list of feature columns to fill NaNs for.

    Returns:
    pd.DataFrame with NaNs filled.
    """
    for feature in features:
        train[feature] = train.groupby("SECTOR")[feature].transform(
            lambda x: x.fillna(x.median())
        )

    for feature in features:
        train[feature] = train[feature].transform(lambda x: x.fillna(x.median()))
    return train


# Function to compute group ratios
def compute_group_ratios(
    train: pd.DataFrame,
    test: pd.DataFrame,
    shifts: List[int] = [1, 2, 3, 4],
    statistics: List[str] = ["sum"],
    grouping_features: List[List[str]] = [["SECTOR", "DATE"]],
    target_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Compute group ratios of shifted target column sums to total sums for each group of specified features.

    Args:
    train: pd.DataFrame, training data containing target columns and grouping features.
    test: pd.DataFrame, testing data containing target columns and grouping features.
    shifts: list, list of shifts to calculate (default: [1, 2, 3, 4]).
    statistics: list, list of statistics to calculate (default: ['sum']).
    grouping_features: list, list of group-by feature lists (default: [['SECTOR', 'DATE']]).
    target_columns: list, list of target columns to calculate (default: None).

    Returns:
    pd.DataFrame for train and test with added group ratio features.
    List[str] of new feature names.
    """
    generated_features = []

    if target_columns is None:
        target_columns = [
            col
            for col in train.columns
            if any(stat in col for stat in ["mean", "std", "min", "max", "median"])
        ]

    # Create shifted columns
    for target_column in target_columns:
        for shift in shifts:
            shifted_column = f"{target_column}_SHIFT_{shift}"
            for df in [train, test]:
                df[shifted_column] = df.groupby("STOCK")[target_column].shift(shift)

    # Compute group ratios
    for target_column in target_columns:
        for group in grouping_features:
            group_name = "_".join(group)
            for shift in shifts:
                for stat in statistics:
                    feature_name = f"{target_column}_SHIFT_{shift}_to_total_{target_column}_of_{group_name}"
                    shifted_column = f"{target_column}_SHIFT_{shift}"
                    generated_features.append(feature_name)
                    for df in [train, test]:
                        df[feature_name] = df[shifted_column] / df.groupby(group)[
                            shifted_column
                        ].transform("sum")
                        # Fill NaNs with the median of the group
                        df[feature_name] = df.groupby("SECTOR")[feature_name].transform(
                            lambda x: x.fillna(x.median())
                        )

    return train, test, generated_features


# Function to compute volatility
def compute_volatility(
    train: pd.DataFrame,
    test: pd.DataFrame,
    periods: List[int] = [2],
    targets: List[str] = ["RET", "VOLUME"],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Compute volatility (standard deviation) for specified targets over given periods.

    Args:
    train: pd.DataFrame, training data containing target columns.
    test: pd.DataFrame, testing data containing target columns.
    periods: list, list of periods in weeks to calculate volatility for (default: [2]).
    targets: list, list of target columns to calculate volatility for (default: ['RET', 'VOLUME']).

    Returns:
    pd.DataFrame for train and test with added volatility features.
    List[str] of new feature names.
    """
    new_features = []

    for period in periods:
        window_size = 5 * period
        for target in targets:
            name = f"{window_size}_day_mean_{target}_VOLATILITY"
            new_features.append(name)
            for data in [train, test]:
                rolling_std_target = (
                    data.groupby(["SECTOR", "DATE"])[
                        [f"{target}_{day}" for day in range(1, window_size + 1)]
                    ]
                    .mean()
                    .std(axis=1)
                    .to_frame(name)
                )
                placeholder = data.join(
                    rolling_std_target, on=["SECTOR", "DATE"], how="left"
                )
                data[name] = placeholder[name]

    return train, test, new_features


# Function to compute advanced volatility features
def compute_advanced_volatility_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    targets: List[str] = ["RET_1", "RET_10", "VOLUME_1", "VOLUME_10"],
    min_window_size: int = 1,
    fill_method: str = "median",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Compute advanced volatility features.

    Args:
    train: pd.DataFrame, training data containing target columns.
    test: pd.DataFrame, testing data containing target columns.
    targets: list, list of target columns to calculate volatility for (default: ['RET_1', 'RET_10', 'RET_20']).
    periods: list, list of periods in weeks to calculate volatility for (default: [2]).
    min_window_size: int, minimum window size for rolling calculations to avoid NaNs.
    fill_method: str, method to fill NaN values ('zero' or 'median').

    Returns:
    pd.DataFrame for train and test with added volatility features.
    List[str] of new feature names.
    """
    new_features = []

    def calculate_features(
        data: pd.DataFrame, target: str, window_size: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        rolling_std = (
            data[target].rolling(window=window_size, min_periods=min_window_size).std()
        )
        vol_skew = (
            data[target]
            .rolling(window=window_size, min_periods=min_window_size)
            .apply(lambda x: skew(x) if len(set(x)) > 1 else np.nan, raw=True)
        )
        vol_of_vol = rolling_std.rolling(
            window=window_size, min_periods=min_window_size
        ).std()
        return rolling_std, vol_skew, vol_of_vol

    def fill_missing_values(
        data: pd.DataFrame, columns: List[str], fill_method: str
    ) -> pd.DataFrame:
        if fill_method == "median":
            for col in columns:
                data[col] = data.groupby("SECTOR")[col].transform(
                    lambda x: x.fillna(x.median())
                )
        elif fill_method == "zero":
            data[columns] = data[columns].fillna(0)
        return data

    window_size = 10
    for target in targets:
        rolling_std_name = f"{window_size}_day_rolling_std_{target}"
        vol_skew_name = f"{window_size}_day_vol_skew_{target}"  # just skew of volume and ret, not of volatility
        vol_of_vol_name = f"{window_size}_day_vol_of_vol_{target}"

        new_features.extend([rolling_std_name, vol_skew_name, vol_of_vol_name])

        for data in [train, test]:
            grouped = data.groupby(["SECTOR", "DATE"])
            rolling_std, vol_skew, vol_of_vol = zip(
                *grouped.apply(lambda x: calculate_features(x, target, window_size))
            )

            data[rolling_std_name] = np.concatenate(rolling_std)
            data[vol_skew_name] = np.concatenate(vol_skew)
            data[vol_of_vol_name] = np.concatenate(vol_of_vol)

            # Fill missing values
            data = fill_missing_values(
                data, [rolling_std_name, vol_skew_name, vol_of_vol_name], fill_method
            )

    return train, test, new_features


def filter_infinity_values(
    train: pd.DataFrame, test: pd.DataFrame, features: List[str], target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
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
    print(filtered_features + [target])
    # Filter to only include columns that exist in both dataframes
    train_available_features = [f for f in filtered_features if f in train.columns]
    test_available_features = [f for f in filtered_features if f in test.columns]
    if set(train_available_features) != set(test_available_features):
        log.warning("Train and test datasets have different available features")
    train_filtered = train[train_available_features + [target]]
    test_filtered = test[test_available_features]

    return train_filtered, test_filtered, filtered_features


def remove_duplicated_columns(
    train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
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


def calculate_moving_averages(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate mean returns and moving averages for different periods.

    Args:
        data: pd.DataFrame containing RET_i columns

    Returns:
        Tuple containing:
            - DataFrame with added moving average columns
            - List of new feature names
    """
    data = data.copy()
    new_features = []

    # Calculate overall mean of returns
    data["Mean"] = data[[f"RET_{i}" for i in range(1, 21)]].mean(axis=1)
    new_features.append("Mean")

    # Calculate moving averages for different periods
    ma_periods = [5, 10, 15]
    for period in ma_periods:
        col_name = f"MA{period}"
        data[col_name] = data[[f"RET_{i}" for i in range(1, period + 2)]].mean(axis=1)
        new_features.append(col_name)

    return data, new_features


def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    """
    Calculate Bollinger Bands for each stock for RET_1 to RET_5.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'RET_i' columns (i from 1 to 5).
    window: int, the lookback period for SMA calculation.
    num_std_dev: int, the number of standard deviations for band calculation.

    Returns:
    pd.DataFrame with Bollinger Bands and distance between bands for each stock for RET_1 to RET_5.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in [1, 5, 14]:
        ret_col = f"RET_{i}"
        sma_col = f"SMA_{i}"
        std_col = f"STD_{i}"
        upper_band_col = f"Upper_Band_{i}"
        lower_band_col = f"Lower_Band_{i}"
        band_distance_col = f"Band_Distance_{i}"

        # new_features.extend([sma_col, std_col, upper_band_col, lower_band_col, band_distance_col])
        new_features.extend([upper_band_col, lower_band_col, band_distance_col])

        # Calculate SMA
        data[sma_col] = data.groupby("STOCK")[ret_col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

        # Calculate Standard Deviation
        data[std_col] = data.groupby("STOCK")[ret_col].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        data = data.fillna(data.median())

        # Calculate Bollinger Bands
        data[upper_band_col] = data[sma_col] + (num_std_dev * data[std_col])
        data[lower_band_col] = data[sma_col] - (num_std_dev * data[std_col])

        # Calculate distance between bands
        data[band_distance_col] = data[upper_band_col] - data[lower_band_col]

    return data, new_features


def calculate_cumulative_returns(df):
    feat_ = []
    for day in range(1, 21):
        cum_return_col = f"CUM_RET_{day}"
        feat_.append(cum_return_col)
        df[cum_return_col] = df.groupby("STOCK")[f"RET_{day}"].transform(
            lambda x: (1 + x).cumprod() - 1
        )
    return df, feat_


def calculate_bollinger_bands_cum_ret(data, window=20, num_std_dev=2):
    """
    Calculate Bollinger Bands for each stock for CUM_RET_1 to CUM_RET_5.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'CUM_RET_i' columns (i from 1 to 5).
    window: int, the lookback period for SMA calculation.
    num_std_dev: int, the number of standard deviations for band calculation.

    Returns:
    pd.DataFrame with Bollinger Bands and distance between bands for each stock for CUM_RET_1 to CUM_RET_5.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in [1, 5, 14]:
        cum_ret_col = f"CUM_RET_{i}"
        sma_col = f"SMA_CUM_{i}"
        std_col = f"STD_CUM_{i}"
        upper_band_col = f"Upper_Band_CUM_{i}"
        lower_band_col = f"Lower_Band_CUM_{i}"
        band_distance_col = f"Band_Distance_CUM_{i}"

        # new_features.extend([sma_col, std_col, upper_band_col, lower_band_col, band_distance_col])
        new_features.extend([band_distance_col])

        # Calculate SMA
        data[sma_col] = data.groupby("STOCK")[cum_ret_col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

        # Calculate Standard Deviation
        data[std_col] = data.groupby("STOCK")[cum_ret_col].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

        # Fill NaNs with the median for each stock group
        data[[sma_col, std_col]] = data.groupby("STOCK")[[sma_col, std_col]].transform(
            lambda x: x.fillna(x.median())
        )

        # Calculate Bollinger Bands
        data[upper_band_col] = data[sma_col] + (num_std_dev * data[std_col])
        data[lower_band_col] = data[sma_col] - (num_std_dev * data[std_col])

        # Calculate distance between bands
        data[band_distance_col] = data[upper_band_col] - data[lower_band_col]

        # Fill NaNs with the median for each stock group
        data[[std_col, upper_band_col, lower_band_col, band_distance_col]] = (
            data.groupby(
                "STOCK"
            )[
                [std_col, upper_band_col, lower_band_col, band_distance_col]
            ].transform(lambda x: x.fillna(x.median()))
        )

        # Fill NaNs with the median for each sector group
        data[[std_col, upper_band_col, lower_band_col, band_distance_col]] = (
            data.groupby(
                "SECTOR"
            )[
                [std_col, upper_band_col, lower_band_col, band_distance_col]
            ].transform(lambda x: x.fillna(x.median()))
        )

    return data, new_features


def calculate_mfi(data, window=14):
    """
    Calculate MFI for each stock for RET_1, RET_10, and RET_20.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', 'RET_i' and 'VOLUME_i' columns (i being 1, 10, and 20).
    window: int, the lookback period for MFI calculation.

    Returns:
    pd.DataFrame with MFI values for each stock.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in [1, 10, 20]:
        ret_col = f"RET_{i}"
        volume_col = f"VOLUME_{i}"
        typical_price_col = f"Typical_Price_{i}"
        money_flow_col = f"Money_Flow_{i}"
        positive_money_flow_col = f"Positive_Money_Flow_{i}"
        negative_money_flow_col = f"Negative_Money_Flow_{i}"
        money_flow_ratio_col = f"Money_Flow_Ratio_{i}"
        mfi_col = f"MFI_{i}"
        overbought_col = f"overbought_mfi_{i}"
        oversold_col = f"oversold_mfi_{i}"

        # new_features.extend([typical_price_col, money_flow_col, positive_money_flow_col, negative_money_flow_col, money_flow_ratio_col, mfi_col, overbought_col, oversold_col])
        new_features.extend([mfi_col, overbought_col, oversold_col])

        # Calculate Typical Price using RET_i
        data[typical_price_col] = data[ret_col]

        # Calculate Money Flow
        data[money_flow_col] = data[typical_price_col] * data[volume_col]

        # Calculate Positive and Negative Money Flow
        data[positive_money_flow_col] = np.where(
            data[typical_price_col].diff(1) > 0, data[money_flow_col], 0
        )
        data[negative_money_flow_col] = np.where(
            data[typical_price_col].diff(1) < 0, data[money_flow_col], 0
        )

        # Calculate Money Flow Ratio
        data[money_flow_ratio_col] = data.groupby("STOCK")[
            positive_money_flow_col
        ].transform(lambda x: x.rolling(window, min_periods=1).sum()) / data.groupby(
            "STOCK"
        )[negative_money_flow_col].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )

        # Calculate MFI
        data[mfi_col] = 100 - (100 / (1 + data[money_flow_ratio_col]))

        # Add overbought and oversold signals
        data[overbought_col] = np.where(data[mfi_col] > 80, 1, 0)
        data[oversold_col] = np.where(data[mfi_col] < 20, 1, 0)

        # Fill NaNs with the median for each stock group
        data[[money_flow_ratio_col, mfi_col]] = data.groupby("STOCK")[
            [money_flow_ratio_col, mfi_col]
        ].transform(lambda x: x.fillna(x.median()))

        # Fill NaNs with the median for each sector group
        data[[money_flow_ratio_col, mfi_col]] = data.groupby("SECTOR")[
            [money_flow_ratio_col, mfi_col]
        ].transform(lambda x: x.fillna(x.median()))

    return data, new_features


def calculate_ema(data, window=20):
    """
    Calculate EMA for each stock for RET_1 to RET_5 and CUM_RET_1 to CUM_RET_5.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', 'RET_i' and 'CUM_RET_i' columns (i from 1 to 5).
    window: int, the lookback period for EMA calculation.

    Returns:
    pd.DataFrame with EMA values for each stock.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in range(1, 6):
        ret_col = f"RET_{i}"
        cum_ret_col = f"CUM_RET_{i}"
        ema_ret_col = f"EMA_{ret_col}"
        ema_cum_ret_col = f"EMA_{cum_ret_col}"

        new_features.extend([ema_ret_col, ema_cum_ret_col])

        # Calculate EMA of daily returns
        data[ema_ret_col] = data.groupby("STOCK")[ret_col].transform(
            lambda x: x.ewm(span=window, adjust=False).mean()
        )

        # Calculate EMA of cumulative returns
        data[ema_cum_ret_col] = data.groupby("STOCK")[cum_ret_col].transform(
            lambda x: x.ewm(span=window, adjust=False).mean()
        )

    return data, new_features


def calculate_sma(data, window=20):
    """
    Calculate SMA for each stock for RET_1 to RET_5 and CUM_RET_1 to CUM_RET_5.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', 'RET_i' and 'CUM_RET_i' columns (i from 1 to 5).
    window: int, the lookback period for SMA calculation.

    Returns:
    pd.DataFrame with SMA values for each stock.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in range(1, 6):
        ret_col = f"RET_{i}"
        cum_ret_col = f"CUM_RET_{i}"
        sma_ret_col = f"SMA_{ret_col}"
        sma_cum_ret_col = f"SMA_{cum_ret_col}"

        new_features.extend([sma_ret_col, sma_cum_ret_col])

        # Calculate SMA of daily returns
        data[sma_ret_col] = data.groupby("STOCK")[ret_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

        # Calculate SMA of cumulative returns
        data[sma_cum_ret_col] = data.groupby("STOCK")[cum_ret_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    return data, new_features


def detect_volume_spikes(df, threshold=2):
    new_features_volume = []

    for day in [1, 5, 14]:  #  range(1, 21):
        deviation_rvol_col = f"DEVIATION_RVOL_{day}"
        spike_col = f"SPIKE_{day}"
        new_features_volume.append(spike_col)

        # Detect volume spikes
        df[spike_col] = (
            np.abs(df[deviation_rvol_col])
            > threshold
            * df.groupby("STOCK")[deviation_rvol_col].transform(
                lambda x: x.rolling(window=20).std()
            )
        ).astype(int)
    return df, new_features_volume


def calculate_adl(df, window=14):
    """
    Calculate ADL for each stock.

    Args:
    df: pd.DataFrame, containing 'STOCK', 'DATE', 'RET_i' (i from 1 to 5) and 'VOLUME_i' columns.

    Returns:
    pd.DataFrame with ADL values for each stock.
    list of new feature names.
    """
    df = df.copy()
    new_features = []
    epsilon = 1e-10  # Small value to prevent division by zero

    for i in [1, 3, 5, 10]:
        high_col = f"HIGH_{i}"
        low_col = f"LOW_{i}"
        close_col = f"RET_{i}"
        volume_col = f"VOLUME_{i}"
        money_flow_multiplier_col = f"Money_Flow_Multiplier_{i}"
        money_flow_volume_col = f"Money_Flow_Volume_{i}"
        adl_col = f"ADL_{i}"

        # new_features.extend([money_flow_multiplier_col, money_flow_volume_col, adl_col])
        new_features.extend([adl_col])

        # Calculate High, Low, and Close as the rolling max, min, and close
        df[high_col] = df.groupby("STOCK")[close_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).max()
        )
        df[low_col] = df.groupby("STOCK")[close_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).min()
        )

        # Calculate Money Flow Multiplier
        df[money_flow_multiplier_col] = (
            (df[close_col] - df[low_col]) - (df[high_col] - df[close_col])
        ) / (df[high_col] - df[low_col] + epsilon)

        # Calculate Money Flow Volume
        df[money_flow_volume_col] = df[money_flow_multiplier_col] * df[volume_col]

        # Fill NaNs with the median of the Money Flow Volume column
        df[money_flow_volume_col] = df.groupby("STOCK")[
            money_flow_volume_col
        ].transform(lambda x: x.fillna(x.median()))

        # Calculate ADL
        df[adl_col] = df.groupby("STOCK")[money_flow_volume_col].cumsum()

    return df, new_features


def calculate_relative_volume(data, window=20):
    """
    Calculate Relative Volume for each stock for VOLUME_1 to VOLUME_5.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', and 'VOLUME_i' columns (i from 1 to 5).
    window: int, the lookback period for average volume calculation.

    Returns:
    pd.DataFrame with Relative Volume values for each stock.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in range(1, 6):
        volume_col = f"VOLUME_{i}"
        avg_volume_col = f"Average_Volume_{i}"
        rel_volume_col = f"Relative_Volume_{i}"

        new_features.extend([avg_volume_col, rel_volume_col])

        # Calculate average volume over the specified period
        data[avg_volume_col] = data.groupby("STOCK")[volume_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

        # Calculate relative volume
        data[rel_volume_col] = data[volume_col] / data[avg_volume_col]

    return data, new_features


def calculate_obv(data):
    """
    Calculate OBV for each stock for RET_1 to RET_5 and VOLUME_1 to VOLUME_5.

    Args:
    data: pd.DataFrame, containing 'STOCK', 'DATE', 'RET_i' (i from 1 to 5) and 'VOLUME_i' columns.

    Returns:
    pd.DataFrame with OBV values for each stock.
    list of new feature names.
    """
    data = data.copy()
    new_features = []

    for i in [1, 3, 5, 14]:
        ret_col = f"RET_{i}"
        volume_col = f"VOLUME_{i}"
        obv_col = f"OBV_{i}"

        new_features.append(obv_col)

        # Calculate OBV
        data[obv_col] = (
            data.groupby("STOCK")
            .apply(
                lambda x: (
                    (x[ret_col].diff() > 0) * x[volume_col]
                    - (x[ret_col].diff() < 0) * x[volume_col]
                ).cumsum()
            )
            .reset_index(level=0, drop=True)
        )

    return data, new_features


def calculate_atr(df, window=14):
    """
    Calculate ATR per stock.

    Args:
    df: pd.DataFrame, containing 'STOCK', 'DATE', and 'RET_i' columns.
    window: int, the lookback period for ATR calculation.

    Returns:
    pd.DataFrame with ATR values for each stock.
    list of new feature names.
    """
    df = df.copy()
    new_features = []

    for day in [1, 5, 14]:
        # Calculate High, Low, and Close
        df[f"HIGH_{day}"] = df.groupby("STOCK")[f"RET_{day}"].transform(
            lambda x: x.rolling(window=window, min_periods=1).max()
        )
        df[f"LOW_{day}"] = df.groupby("STOCK")[f"RET_{day}"].transform(
            lambda x: x.rolling(window=window, min_periods=1).min()
        )
        df[f"CLOSE_{day}"] = df[f"RET_{day}"]

        # Calculate True Range (TR)
        df[f"TR_{day}"] = (
            df.groupby("STOCK")
            .apply(
                lambda x: np.maximum(
                    x[f"HIGH_{day}"] - x[f"LOW_{day}"],
                    np.abs(x[f"HIGH_{day}"] - x[f"CLOSE_{day}"].shift(1)),
                    np.abs(x[f"LOW_{day}"] - x[f"CLOSE_{day}"].shift(1)),
                )
            )
            .reset_index(level=0, drop=True)
        )

        # Fill NaNs with the median of the TR column
        df[f"TR_{day}"] = df.groupby("STOCK")[f"TR_{day}"].transform(
            lambda x: x.fillna(x.median())
        )

        # Calculate Average True Range (ATR)
        df[f"ATR_{day}"] = df.groupby("STOCK")[f"TR_{day}"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        new_features.append(f"ATR_{day}")

        # Fill NaNs with the median for each stock group
        df[[f"ATR_{day}"]] = df.groupby("STOCK")[[f"ATR_{day}"]].transform(
            lambda x: x.fillna(x.median())
        )

        # Fill NaNs with the median for each sector group
        df[[f"ATR_{day}"]] = df.groupby("SECTOR")[[f"ATR_{day}"]].transform(
            lambda x: x.fillna(x.median())
        )

    return df, new_features


# Example usage:
# df_with_atr, new_features_atr = calculate_atr(df)


def calculate_technical_features(train_df, test_df):
    """Calculate all technical indicators for both train and test dataframes."""
    technical_functions = [
        calculate_rsi,
        calculate_rsi_per_sector,
        calculate_roc_past_rows,
        calculate_momentum,
        calculate_momentum_sector,
        calculate_stochastic_oscillator,
        calculate_macd,
        calculate_golden_cross,
        calculate_moving_averages,
        calculate_adl,
        calculate_relative_volume,
        calculate_obv,
        calculate_mfi,
        calculate_ema,
        calculate_sma,
        detect_volume_spikes,
    ]

    for func in technical_functions:
        log.info(f"Calculating {func.__name__} for train and test dataframes")
        train_df, _ = func(train_df)
        test_df, _ = func(test_df)

    return train_df, test_df


def calculate_ta_indicators(
    df: pd.DataFrame,
    periods: list[int] = [2, 5, 14],
    remove_nan: bool = True,
    ta_func: callable = talib.RSI,
    ta_args: dict = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Calculate technical indicators for different periods.

    Args:
        df: Input DataFrame containing RET_ and/or VOL_ columns
        periods: List of periods to calculate indicators for, defaults to [2, 5, 14]
        remove_nan: Whether to remove columns containing NaN values, defaults to True
        ta_func: TA-Lib function to use for calculation, defaults to talib.RSI
        ta_args: Dictionary of arguments to pass to ta_func including data_type, defaults to None

    Returns:
        Tuple containing:
            - DataFrame containing calculated technical indicators
            - List of new feature column names
    """
    # Check that ta_args is provided and contains data_type
    if ta_args is None:
        raise ValueError("ta_args must be provided with data_type")
    if "data_type" not in ta_args:
        raise ValueError("data_type must be specified in ta_args")

    data_type = ta_args.pop(
        "data_type"
    )  # Remove data_type from ta_args after getting it
    if data_type not in ["ret", "vol", "both"]:
        raise ValueError("data_type must be one of: 'ret', 'vol', 'both'")

    # Get both RET_ and VOL_ columns
    ret_columns = [col for col in df.columns if col.startswith("RET_")]
    vol_columns = [col for col in df.columns if col.startswith("VOLUME_")]

    # Process RET_ and VOL_ columns based on data_type
    ret_df = pd.DataFrame()
    vol_df = pd.DataFrame()

    if data_type in ["ret", "both"] and ret_columns:
        ret_df = df[ret_columns[::-1]].transpose()
    if data_type in ["vol", "both"] and vol_columns:
        vol_df = df[vol_columns[::-1]].transpose()

    results = pd.DataFrame()
    new_features = []

    # Get function name from ta_func for column naming
    func_name = ta_func.__name__

    # Process based on data_type
    def process_indicator(df_input, is_combined=False):
        if is_combined:
            indicator = (
                df_input[0]
                .reset_index(drop=True)
                .combine(
                    df_input[1].reset_index(drop=True),
                    lambda x, y: ta_func(x.ffill(), y.ffill(), **ta_args),
                )
            )
        else:
            indicator = df_input.apply(lambda x: ta_func(x.ffill(), **ta_args))

        indicator = indicator.ffill().T
        new_cols = [
            f"{func_name}_{i+1}_{period}" for i in range(len(indicator.columns))
        ]
        indicator.columns = new_cols
        new_features.extend(new_cols)
        return indicator

    if data_type == "ret" and not ret_df.empty:
        for period in periods:
            results = pd.concat([results, process_indicator(ret_df)], axis=1)

    elif data_type == "vol" and not vol_df.empty:
        for period in periods:
            results = pd.concat([results, process_indicator(vol_df)], axis=1)

    elif data_type == "both" and not ret_df.empty and not vol_df.empty:
        for period in periods:
            results = pd.concat(
                [results, process_indicator((ret_df, vol_df), True)], axis=1
            )

    if remove_nan:
        # Remove columns that contain any NaN values
        nan_columns = results.columns[results.isna().all()].tolist()
        if nan_columns:
            results = results.drop(columns=nan_columns)
            new_features = [f for f in new_features if f not in nan_columns]

    return results, new_features


def calculate_all_ta_indicators(df, features=None):
    """Calculate all technical indicators for the given dataframe.

    Args:
        df (pd.DataFrame): Input dataframe with price/volume data
        features (list, optional): List of technical indicators to calculate. If None, calculates all.

    Returns:
        pd.DataFrame: DataFrame containing all calculated technical indicators
    """
    all_features = [
        (talib.OBV, {"data_type": "both"}),
        (talib.RSI, {"data_type": "ret"}),
        (talib.MOM, {"timeperiod": 5, "data_type": "ret"}),
        (talib.ROCR, {"timeperiod": 5, "data_type": "ret"}),
        (talib.CMO, {"timeperiod": 14, "data_type": "ret"}),
        (talib.EMA, {"timeperiod": 5, "data_type": "ret"}),
        (talib.SMA, {"timeperiod": 5, "data_type": "ret"}),
        (talib.WMA, {"timeperiod": 5, "data_type": "ret"}),
        (talib.MIDPOINT, {"timeperiod": 10, "data_type": "ret"}),
        # (talib.MIDPRICE, {"timeperiod": 10, "data_type": "ret"}),
    ]

    # if features is not None:
    #     all_features = [(ta_func, ta_args) for ta_func, ta_args in all_features
    #                    if ta_func.__name__ in features]

    ta_indicators_df = pd.concat(
        [
            result[0]  # Get first element of tuple returned by calculate_ta_indicators
            for result in [
                calculate_ta_indicators(
                    df[features],
                    periods=[2, 5, 14],
                    ta_func=ta_func,
                    ta_args=ta_args,
                )
                for ta_func, ta_args in all_features
            ]
        ],
        axis=1,
    )
    return ta_indicators_df
