import unittest
import pandas as pd
import numpy as np
from src.strategies.vcp_detector import detect_vcp


class TestVCPDetector(unittest.TestCase):
    def test_vcp_detection_positive(self):
        # Create a synthetic VCP pattern
        dates = pd.date_range("2023-01-01", periods=100)
        df = pd.DataFrame(index=dates)

        # Base prices at 100
        # Base prices at 50 to avoid flatline issues with min/max
        df["Open"] = 50.0
        df["Low"] = 50.0
        df["High"] = 50.0
        df["Close"] = 50.0
        df["Volume"] = 10000.0

        # We need the 20MA volume to be higher than trough 3 volume
        df.iloc[-20:, df.columns.get_loc("Volume")] = 10000.0

        # Peak 1 at day 60
        df.iloc[60, df.columns.get_loc("High")] = 100.0

        # Trough 1 at day 70
        # (Drop > 15%, so 100 to 80 is 20%)
        df.iloc[70, df.columns.get_loc("Low")] = 80.0

        # Peak 2 at day 75
        df.iloc[75, df.columns.get_loc("High")] = 95.0

        # Trough 2 at day 85
        # (Drop < P1 * 0.6 = 20 * 0.6 = 12%, so 95 to 85 is 10.5%)
        df.iloc[85, df.columns.get_loc("Low")] = 85.0

        # Peak 3 at day 90
        df.iloc[90, df.columns.get_loc("High")] = 92.0

        # Trough 3 at day 95
        # (Drop < 5%, so 92 to 89 is 3.2%)
        df.iloc[95, df.columns.get_loc("Low")] = 89.0
        df.iloc[95, df.columns.get_loc("Volume")] = (
            3000.0  # Vol contract < 60% of 10000
        )

        # Last close
        df.iloc[99, df.columns.get_loc("Close")] = 91.0

        result = detect_vcp(df)

        self.assertEqual(result["signal"], "VCP")
        self.assertTrue(result["quality_score"] > 0)
        self.assertEqual(result["stop_loss"], 89.0)
        self.assertEqual(result["entry_price"], 91.0)

    def test_vcp_detection_negative_vol(self):
        # Same as above but volume doesn't dry up
        dates = pd.date_range("2023-01-01", periods=100)
        df = pd.DataFrame(index=dates)
        df["Open"] = 100.0
        df["High"] = 100.0
        df["Low"] = 100.0
        df["Close"] = 100.0
        df["Volume"] = 10000.0
        df.iloc[60, df.columns.get_loc("High")] = 100.0
        df.iloc[70, df.columns.get_loc("Low")] = 80.0
        df.iloc[75, df.columns.get_loc("High")] = 95.0
        df.iloc[85, df.columns.get_loc("Low")] = 85.0
        df.iloc[90, df.columns.get_loc("High")] = 92.0
        df.iloc[95, df.columns.get_loc("Low")] = 89.0
        # Volume stays high
        df.iloc[95, df.columns.get_loc("Volume")] = 9000.0
        df.iloc[-1, df.columns.get_loc("Close")] = 91.0

        result = detect_vcp(df)
        self.assertEqual(result["signal"], "NO")


if __name__ == "__main__":
    unittest.main()
