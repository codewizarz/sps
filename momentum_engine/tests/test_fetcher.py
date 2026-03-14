import os
import tempfile
import unittest
from src.data_fetcher import IndianDataFetcher


class TestIndianDataFetcher(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for tests
        self.test_dir = tempfile.TemporaryDirectory()
        self.fetcher = IndianDataFetcher(data_dir=self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_fetch_nifty_symbols(self):
        symbols = self.fetcher.fetch_nifty_symbols()
        self.assertIsInstance(symbols, list)
        self.assertTrue(len(symbols) > 40)
        self.assertTrue(all(sym.endswith(".NS") for sym in symbols))

    def test_fetch_historical(self):
        # Fetching for 5 days of RELIANCE.NS to ensure quick response
        filepath = self.fetcher.fetch_historical(
            "RELIANCE.NS", "2023-01-01", "2023-01-10"
        )
        self.assertIsNotNone(filepath)
        self.assertTrue(os.path.exists(filepath))

        import pandas as pd

        df = pd.read_csv(filepath)
        self.assertFalse(df.empty)
        self.assertIn("Close", df.columns)

    def test_fetch_nifty_index_data(self):
        filepath = self.fetcher.fetch_nifty_index_data("2023-01-01", "2023-01-10")
        self.assertIsNotNone(filepath)
        self.assertTrue(os.path.exists(filepath))
        # Ensure filename was cleaned (e.g. ^NSEI -> NSEI.csv)
        self.assertTrue("NSEI.csv" in filepath)


if __name__ == "__main__":
    unittest.main()
