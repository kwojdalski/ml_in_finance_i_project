import pandas as pd
import talib


def calculate_ta_indicators(  # noqa
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
    ]

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
