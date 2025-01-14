import unittest

import numpy as np
import pandas as pd

from ml_in_finance_i_project.ta_indicators import (
    calculate_bollinger_bands,
    calculate_cumulative_returns,
    calculate_momentum,
    calculate_momentum_sector,
    calculate_moving_averages,
    calculate_roc_past_rows,
    calculate_rsi,
    calculate_stochastic_oscillator,
)


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

    def test_rsi(self):
        result, features = calculate_rsi(self.test_data)

        # Check RSI columns exist
        self.assertTrue(any("RSI" in col for col in result.columns))
        self.assertTrue(all(isinstance(x, str) for x in features))

        # Check RSI values are within valid range (0-100)
        rsi_cols = [col for col in result.columns if "RSI" in col]
        for col in rsi_cols:
            self.assertTrue((result[col] >= 0).all())
            self.assertTrue((result[col] <= 100).all())

    def test_roc_past_rows(self):
        result, features = calculate_roc_past_rows(self.test_data)
        self.assertTrue(any("ROC" in col for col in result.columns))
        self.assertTrue(all(isinstance(x, str) for x in features))

    def test_momentum(self):
        result, features = calculate_momentum(self.test_data)
        self.assertTrue(any("momentum" in col.lower() for col in result.columns))
        self.assertTrue(all(isinstance(x, str) for x in features))

    def test_momentum_sector(self):
        result, features = calculate_momentum_sector(self.test_data)
        self.assertTrue(any("momentum_sector" in col.lower() for col in result.columns))
        self.assertTrue(all(isinstance(x, str) for x in features))

    def test_stochastic_oscillator(self):
        result, features = calculate_stochastic_oscillator(self.test_data)
        self.assertTrue(any("%K" in col for col in result.columns))
        self.assertTrue(any("%D" in col for col in result.columns))
        self.assertTrue(all(isinstance(x, str) for x in features))

    def test_moving_averages(self):
        result, features = calculate_moving_averages(self.test_data)
        self.assertIn("Mean", result.columns)
        self.assertTrue(any("MA" in col for col in result.columns))
        self.assertTrue(all(isinstance(x, str) for x in features))

    def test_bollinger_bands(self):
        result, features = calculate_bollinger_bands(self.test_data)
        self.assertTrue(any("Upper_Band" in col for col in result.columns))
        self.assertTrue(any("Lower_Band" in col for col in result.columns))
        self.assertTrue(all(isinstance(x, str) for x in features))

    def test_cumulative_returns(self):
        result, features = calculate_cumulative_returns(self.test_data)
        self.assertTrue(any("CUM_RET" in col for col in result.columns))
        self.assertTrue(all(isinstance(x, str) for x in features))


# a = TestTAIndicators()
# a.setUp()
# a, b = calculate_rsi(a.test_data)
# a["RSI"].range()
if __name__ == "__main__":
    unittest.main()
