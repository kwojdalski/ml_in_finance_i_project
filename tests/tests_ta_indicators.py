import logging
import sys
import unittest

import numpy as np
import pandas as pd
import talib

from src.qrt_stock_returns.pipelines.data_processing.ta_indicators import (
    calculate_all_ta_indicators,
    calculate_ta_indicators,
)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
log.addHandler(stream_handler)


class TestTAIndicators(unittest.TestCase):
    def setUp(self):
        # Create test data for unit tests
        n_stocks = 10
        n_dates = 30  # 30 time periods
        n_rows = n_stocks * n_dates

        # Generate sample data
        np.random.seed(42)
        self.test_data = pd.DataFrame(
            {
                "STOCK": np.random.randint(1, 100, n_rows),
                "INDUSTRY": np.random.randint(1, 20, n_rows),
                "INDUSTRY_GROUP": np.random.randint(1, 15, n_rows),
                "SECTOR": np.random.randint(1, 10, n_rows),
                "SUB_INDUSTRY": np.random.randint(1, 30, n_rows),
                "DATE": np.repeat(range(n_dates), n_stocks),
                "RET": np.random.randn(n_rows) / 100,
                "VOLUME": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_1": np.random.randn(n_rows) / 100,
                "VOLUME_1": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_2": np.random.randn(n_rows) / 100,
                "VOLUME_2": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_3": np.random.randn(n_rows) / 100,
                "VOLUME_3": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_4": np.random.randn(n_rows) / 100,
                "VOLUME_4": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_5": np.random.randn(n_rows) / 100,
                "VOLUME_5": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_6": np.random.randn(n_rows) / 100,
                "VOLUME_6": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_7": np.random.randn(n_rows) / 100,
                "VOLUME_7": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_8": np.random.randn(n_rows) / 100,
                "VOLUME_8": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_9": np.random.randn(n_rows) / 100,
                "VOLUME_9": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_10": np.random.randn(n_rows) / 100,
                "VOLUME_10": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_11": np.random.randn(n_rows) / 100,
                "VOLUME_11": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_12": np.random.randn(n_rows) / 100,
                "VOLUME_12": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_13": np.random.randn(n_rows) / 100,
                "VOLUME_13": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_14": np.random.randn(n_rows) / 100,
                "VOLUME_14": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_15": np.random.randn(n_rows) / 100,
                "VOLUME_15": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_16": np.random.randn(n_rows) / 100,
                "VOLUME_16": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_17": np.random.randn(n_rows) / 100,
                "VOLUME_17": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_18": np.random.randn(n_rows) / 100,
                "VOLUME_18": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_19": np.random.randn(n_rows) / 100,
                "VOLUME_19": np.random.randint(1000, 10000, n_rows).astype(float),
                "RET_20": np.random.randn(n_rows) / 100,
                "VOLUME_20": np.random.randint(1000, 10000, n_rows).astype(float),
            }
        )

        # Generate return and volume columns
        for i in range(1, 21):
            self.test_data[f"RET_{i}"] = np.random.randn(n_rows) / 100
            self.test_data[f"VOLUME_{i}"] = np.random.randint(
                1000, 10000, n_rows
            ).astype(float)

        # Add target
        self.test_data["RET"] = np.random.choice([True, False], size=n_rows)

    def test_calculate_ta_indicators(self):
        """Test calculate_ta_indicators with different configurations"""
        # Test RSI calculation with returns
        result, features = calculate_ta_indicators(
            self.test_data,
            periods=[2, 5, 14],
            ta_func=talib.RSI,
            ta_args={"data_type": "ret"},
        )

        # Check basic properties
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsInstance(features, list)

        # Test with volume data
        result, features = calculate_ta_indicators(
            self.test_data,
            periods=[2, 5],
            ta_func=talib.SMA,
            ta_args={"data_type": "vol", "timeperiod": 5},
        )
        # self.assertTrue(all("SMA" in col for col in result.columns))

        # Test with both return and volume
        result, features = calculate_ta_indicators(
            self.test_data,
            periods=[2],
            ta_func=talib.OBV,
            ta_args={"data_type": "both"},
        )
        # self.assertTrue(all("OBV" in col for col in result.columns))

        # Test error cases
        with self.assertRaises(ValueError):
            calculate_ta_indicators(
                self.test_data,
                ta_func=talib.RSI,
                ta_args={"data_type": "invalid"},
            )

        with self.assertRaises(ValueError):
            calculate_ta_indicators(
                self.test_data,
                ta_func=talib.RSI,
                ta_args={},  # Missing data_type
            )

    def test_calculate_all_ta_indicators(self):
        """Test calculate_all_ta_indicators functionality"""
        # Test with default parameters
        result = calculate_all_ta_indicators(
            self.test_data,
            features=[
                col
                for col in self.test_data.columns
                if col.startswith(("RET_", "VOLUME_"))
            ],
        )

        # Check that result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check that the result has the same number of rows as input
        self.assertEqual(len(result), len(self.test_data))
        # Log min/max values for each column

        log.info("Checking min/max values for each column in result DataFrame:")
        log.debug(result.columns)
        for col in result.columns:
            min_val = result[col].min()
            max_val = result[col].max()
            log.info(f"Column {col}: min={min_val:.4f}, max={max_val:.4f}")

        # Test with specific features
        subset_features = ["RET_1", "RET_2", "VOLUME_1", "VOLUME_2"]
        result_subset = calculate_all_ta_indicators(
            self.test_data, features=subset_features
        )
        self.assertIsInstance(result_subset, pd.DataFrame)


if __name__ == "__main__":
    unittest.main(verbosity=2)
