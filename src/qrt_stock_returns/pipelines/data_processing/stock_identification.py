# %%
"""Functions for identifying and mapping anonymous stock IDs to actual stock symbols."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def estimate_tick_size(
    df: pd.DataFrame,
    bid_col: str = "bid_price",
    ask_col: str = "ask_price",
    stock_id_col: str = "stock_id",
    date_id_col: str = "date_id",
) -> pd.DataFrame:
    """
    Estimate tick size from bid/ask price differences.

    Args:
        df: DataFrame containing bid/ask prices
        bid_col: Name of bid price column
        ask_col: Name of ask price column
        stock_id_col: Name of stock ID column
        date_id_col: Name of date ID column

    Returns:
        DataFrame with tick size estimates and inferred prices
    """
    # Calculate price differences
    df = df.copy()
    df["bid_price_diff"] = df.groupby([date_id_col, stock_id_col])[bid_col].diff().abs()
    df["ask_price_diff"] = df.groupby([date_id_col, stock_id_col])[ask_col].diff().abs()

    # Get minimum non-zero price difference per day/stock
    tick_bid = (
        df[df.bid_price_diff > 0]
        .groupby([date_id_col, stock_id_col])
        .bid_price_diff.min()
    )
    tick_ask = (
        df[df.ask_price_diff > 0]
        .groupby([date_id_col, stock_id_col])
        .ask_price_diff.min()
    )

    tick_est = pd.DataFrame({"tick_est_bid": tick_bid, "tick_est_ask": tick_ask})

    tick_est["tick_size_min"] = tick_est.min(axis=1)
    tick_est["price_est"] = 0.01 / tick_est.tick_size_min

    return tick_est


def align_dates(
    dates: pd.Series, reference_date: str
) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Create mappings between date_ids and actual dates.

    Args:
        dates: Series of dates
        reference_date: Reference date corresponding to date_id=0

    Returns:
        Tuple of (date_id_to_date, date_to_date_id) mappings
    """
    index = dates[dates == reference_date].index[0]
    date_to_date_id = {d: i - index for i, d in enumerate(sorted(dates))}
    date_id_to_date = {i - index: d for i, d in enumerate(sorted(dates))}
    return date_id_to_date, date_to_date_id


def find_stock_matches(
    estimated_prices: pd.DataFrame,
    actual_prices: pd.DataFrame,
    min_date_id: int = 450,
    max_date_id: int = 481,
) -> Dict[int, str]:
    """
    Find matches between anonymous stock IDs and actual stock symbols.

    Args:
        estimated_prices: DataFrame with estimated prices from tick sizes
        actual_prices: DataFrame with actual NASDAQ prices
        min_date_id: Start date_id for comparison
        max_date_id: End date_id for comparison

    Returns:
        Dictionary mapping stock_ids to ticker symbols
    """
    all_scores = []
    size = max_date_id - min_date_id

    for symbol in actual_prices.ticker.unique():
        symbol_data = actual_prices[actual_prices.ticker == symbol]
        symbol_prices = symbol_data.close.values[0:size]

        if len(symbol_prices) < size:
            continue

        for stock_id in estimated_prices.columns:
            est_prices = (
                estimated_prices[stock_id].fillna(0).values[min_date_id:max_date_id]
            )
            distance = (
                np.linalg.norm(symbol_prices - est_prices, 2) / symbol_prices.mean()
            )
            if np.isfinite(distance):
                all_scores.append([symbol, stock_id, distance])

    # Find best matches
    score_df = pd.DataFrame(all_scores, columns=["ticker", "stock_id", "distance"])
    best_matches = score_df.loc[score_df.groupby("stock_id").distance.idxmin()]

    # Log any duplicates
    duplicates = best_matches[best_matches.ticker.duplicated(False)]
    if not duplicates.empty:
        log.warning(f"Found duplicate matches: {duplicates.to_dict('records')}")

    return best_matches.set_index("stock_id").ticker.to_dict()


def identify_stocks(
    df: pd.DataFrame, nasdaq_data: pd.DataFrame, reference_date: Optional[str] = None
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Main function to identify anonymous stocks using NASDAQ data.

    Args:
        df: DataFrame with anonymous stock data
        nasdaq_data: DataFrame with NASDAQ price data
        reference_date: Optional reference date for alignment

    Returns:
        Tuple of (stock_id_to_symbol, date_id_to_date) mappings
    """
    # Estimate prices from tick sizes
    tick_estimates = estimate_tick_size(df)
    price_estimates = pd.pivot(
        tick_estimates.reset_index(),
        index="date_id",
        values="price_est",
        columns="stock_id",
    ).fillna(method="ffill")

    # If no reference date provided, try to find MSFT first
    if reference_date is None:
        msft_data = nasdaq_data[nasdaq_data.ticker == "MSFT"]
        # TODO: Implement MSFT-based date alignment
        reference_date = msft_data.date.iloc[0]  # Placeholder

    # Create date mappings
    date_id_to_date, date_to_date_id = align_dates(nasdaq_data.date, reference_date)

    # Find stock matches
    stock_id_to_symbol = find_stock_matches(price_estimates, nasdaq_data)

    return stock_id_to_symbol, date_id_to_date
