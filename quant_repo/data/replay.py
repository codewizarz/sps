import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import time
import os


class MarketReplay:
    """
    Lightweight Options Market Replay Tool.
    Visualizes option chains and Greeks step-by-step.
    """

    def __init__(self, data: Optional[pl.DataFrame] = None):
        if data is not None:
            self.tape = data.sort("timestamp")
        else:
            self.tape = self._generate_mock_tape()

        self.timestamps = self.tape["timestamp"].unique().sort()
        self.current_idx = 0
        self.max_idx = len(self.timestamps) - 1

    def _generate_mock_tape(self) -> pl.DataFrame:
        """Generates a synthetic day of options data for demonstration."""
        base_time = datetime(2023, 1, 5, 9, 15)
        times = [
            base_time + timedelta(minutes=i) for i in range(0, 375, 15)
        ]  # 15 min steps

        frames = []
        spot_price = 18000.0

        for t in times:
            # Random walk spot
            spot_price += np.random.normal(0, 5)

            # Strikes relative to spot
            strikes = [int(spot_price // 100 * 100) + i * 100 for i in range(-5, 6)]

            for k in strikes:
                # Basic pricing logic (Mock)
                moneyness = k - spot_price
                iv = 15.0 + (abs(moneyness) / 1000.0) * 2.0  # Skew

                # Call Price (Approx)
                if k < spot_price:  # ITM Call
                    call_px = (spot_price - k) + (iv / 2)
                    delta = 0.5 + (0.5 * (1 - abs(moneyness) / 1000))
                else:  # OTM Call
                    call_px = max(1.0, (1000 - abs(moneyness)) / 20) * (iv / 10)
                    delta = 0.5 - (0.5 * (abs(moneyness) / 1000))

                frames.append(
                    {
                        "timestamp": t,
                        "symbol": "NIFTY",
                        "spot": spot_price,
                        "strike": k,
                        "type": "CE",
                        "bid": round(call_px * 0.99, 1),
                        "ask": round(call_px * 1.01, 1),
                        "iv": round(iv, 1),
                        "delta": round(delta, 2),
                        "oi": np.random.randint(1000, 50000),
                    }
                )

        return pl.DataFrame(frames)

    def next_frame(self):
        if self.current_idx < self.max_idx:
            self.current_idx += 1
        return self.get_current_frame()

    def prev_frame(self):
        if self.current_idx > 0:
            self.current_idx -= 1
        return self.get_current_frame()

    def get_current_frame(self) -> pl.DataFrame:
        ts = self.timestamps[self.current_idx]
        return self.tape.filter(pl.col("timestamp") == ts)

    def render(self):
        """Renders the current frame to the terminal."""
        df = self.get_current_frame()
        ts = self.timestamps[self.current_idx]
        spot = df["spot"][0]

        # Clear screen (simulated for simplicity in non-interactive environments)
        # os.system('cls' if os.name == 'nt' else 'clear')

        print(f"\n--- MARKET REPLAY: {ts} ---")
        print(f"Index Spot: {spot:.2f}")
        print("-" * 65)
        print(
            f"{'Strike':^8} | {'Bid':^8} | {'Ask':^8} | {'IV':^6} | {'Delta':^6} | {'OI':^8}"
        )
        print("-" * 65)

        # Sort by strike descending
        df = df.sort("strike", descending=True)

        for row in df.iter_rows(named=True):
            is_atm = abs(row["strike"] - spot) < 50
            marker = "*" if is_atm else " "

            line = (
                f"{row['strike']:^8} | {row['bid']:^8.1f} | {row['ask']:^8.1f} | "
                f"{row['iv']:^6.1f} | {row['delta']:^6.2f} | {row['oi']:^8}"
            )

            if is_atm:
                print(f"\033[1m{line} < ATM\033[0m")  # Bold for ATM
            else:
                print(line)
        print("-" * 65)
        print(f"Frame {self.current_idx + 1} / {self.max_idx + 1}")

    def run_interactive(self):
        """Simple interactive loop."""
        print("Starting Replay. Controls: [n]ext, [p]rev, [q]uit")
        self.render()

        while True:
            cmd = input("> ").strip().lower()
            if cmd == "q":
                break
            elif cmd == "n":
                self.next_frame()
                self.render()
            elif cmd == "p":
                self.prev_frame()
                self.render()
            else:
                pass


if __name__ == "__main__":
    replay = MarketReplay()  # Uses mock data by default
    replay.run_interactive()
