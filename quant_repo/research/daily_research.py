import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import argparse
from datetime import date, datetime
import time
import polars as pl

# Import our Zero-Budget Tools
from quant_repo.data.nse_collector import NSECollector
from quant_repo.data.bhavcopy import BhavcopyLoader
from quant_repo.data.pipeline import DataPipeline, Catalog
from quant_repo.data.audit import DataAuditor


def run_daily_cycle(dry_run: bool = False):
    """
    Orchestrates the Daily 'Free Lunch' Research Cycle.
    1. Ingest Intra-day Snapshots (Self-Collected).
    2. Download Official Closing Data (Bhavcopy).
    3. Audit & Update Research Lake.
    """
    today = date.today()
    print(f"=== ZERO-BUDGET RESEARCH CYCLE: {today} ===")

    # Paths
    lake_path = "./data/history"
    live_snap_path = Path("./data/live_snapshots") / today.strftime("%Y-%m-%d")

    # Tools
    pipeline = DataPipeline(root_path=lake_path)
    bhav_loader = BhavcopyLoader(root_path=lake_path)
    auditor = DataAuditor()

    # --- STEP 1: Process Live Intra-day Data ---
    print("\n[Step 1] Processing Live Snapshots...")

    if live_snap_path.exists():
        # Load all parquet snapshots for today
        try:
            # Polars Glob Pattern
            lazy_snap = pl.scan_parquet(str(live_snap_path / "*.parquet"))
            live_df = lazy_snap.collect()

            print(f"Loaded {len(live_df)} intra-day records.")

            if not dry_run:
                # Ingest into 'Intraday' Partition?
                # Our main pipeline partitions by Symbol/Year mostly for daily bars.
                # For intraday, we might want a separate lake or just keep raw snapshots?
                # Let's verify health but keep snapshots as 'Official Raw'.

                report = auditor.audit_dataframe(live_df)
                print(f"Live Data Health: {report.health_score:.1f}%")

                # We could aggregate to 1-min bars here if needed
                # For now, we trust the snapshot collector to persist raw.
        except Exception as e:
            print(f"Error processing snapshots: {e}")
            if not dry_run:
                # If no snapshots, maybe we forgot to run collector?
                print("WARNING: No live data found or readable.")
    else:
        print(f"No live snapshots found at {live_snap_path}")
        print("Tip: Did you run 'NSECollector.run_scheduler()' during market hours?")

    # --- STEP 2: Download Official Close (Bhavcopy) ---
    print("\n[Step 2] Downloading Official Bhavcopy...")

    if dry_run:
        print("[Dry Run] Skipping Download.")
    else:
        success = bhav_loader.download_and_process_date(today)
        if success:
            print("Bhavcopy Downloaded & Ingested Successfuly.")
        else:
            print(
                "Bhavcopy Download Failed (Market Closed / Holiday / Not Published Yet?)"
            )

    # --- STEP 3: Research / Validation ---
    print("\n[Step 3] Updating Research Lake & Validating...")

    # Check if we have data for today in Lake
    catalog = Catalog(root_path=lake_path)
    try:
        # Check NIFTY
        df_today = catalog.load_range("NIFTY", today, today)
        if len(df_today) > 0:
            print(f"Research Lake Updated: {len(df_today)} rows for NIFTY.")

            # Run simple stats
            close_prices = df_today["close"]
            print(f"Market Close Range: {close_prices.min()} - {close_prices.max()}")
        else:
            print("Research Lake: No data for today yet.")
    except Exception as e:
        print(f"Catalog Check: {e}")

    print("\n=== CYCLE COMPLETE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily Research Automation")
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate without heavy IO/Download"
    )
    args = parser.parse_args()

    run_daily_cycle(dry_run=args.dry_run)
