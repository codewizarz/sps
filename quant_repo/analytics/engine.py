import polars as pl
from pathlib import Path
from typing import Dict
from quant_repo.analytics.metrics import calc_edge_decay
from quant_repo.analytics.attribution import AttributionEngine


class AnalyticsEngine:
    def __init__(self):
        self.attributor = AttributionEngine()

    def analyze(self, trade_log_path: str, output_dir: str):
        path = Path(trade_log_path)
        if not path.exists():
            print(f"[Analytics] File not found: {path}")
            return

        print(f"[Analytics] Loading from {path}...")
        try:
            df = pl.read_parquet(path)
        except Exception:
            # Fallback for CSV
            try:
                df = pl.read_csv(path)
            except Exception as e:
                print(f"[Analytics] Load failed: {e}")
                return

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Edge Decay Analysis
        df_decay = calc_edge_decay(df)
        df_decay.write_parquet(out_path / "trades_with_decay.parquet")

        # 2. Attribution
        print("[Analytics] Running Attribution...")
        by_signal = self.attributor.analyze_by_signal(df)
        by_signal.write_parquet(out_path / "attribution_by_signal.parquet")
        print("\n=== Signal Attribution ===")
        with pl.Config(tbl_formatting="ASCII_MARKDOWN"):
            print(by_signal)

        if "regime" in df.columns:
            by_regime = self.attributor.analyze_by_regime(df)
            by_regime.write_parquet(out_path / "attribution_by_regime.parquet")
            print("\n=== Regime Attribution ===")
            with pl.Config(tbl_formatting="ASCII_MARKDOWN"):
                print(by_regime)

        print(f"[Analytics] Reports saved to {out_path}")
