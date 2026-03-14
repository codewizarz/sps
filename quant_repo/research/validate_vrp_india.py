import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add repo root to sys.path to easily import research.data_normalizer
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from research.data_normalizer import load_normalized_lake


def main():
    lake_dir = repo_root.parent / "data" / "master_fo_lake"

    print(f"Loading normalized dataset from: {lake_dir}")
    df = load_normalized_lake(lake_dir)

    print("\nFiltering data...")
    # Keep only Underlying == "NIFTY"
    df = df[df["symbol"] == "NIFTY"].copy()

    # Keep near-ATM options only
    df = df[np.abs(df["strike"] - df["underlying_price"]) < 200].copy()

    # Check for Implied Volatility column explicitly since standard schema
    # does not strictly define 'implied_vol', but previous dataset
    # may have 'IMPLIED_VOL', 'IV', 'implied_vol', or 'IVOL'.
    iv_col = None
    for possible_iv in ["IMPLIED_VOL", "IV", "implied_vol", "IVOL", "IMPL_VOL"]:
        if possible_iv in df.columns:
            iv_col = possible_iv
            break

    # True IV Calculation using Black-Scholes Newton-Raphson
    # Inputs: S=UNDERLYING_PRICE, K=STRIKE_PRICE, T=days_to_expiry/365, r=0.06, price=option_price, type=CE/PE

    # Pre-calculate inputs
    S = df["underlying_price"].values
    K = df["strike"].values
    T = (df["expiry"] - df["timestamp"]).dt.days.values / 365.0
    T = np.maximum(T, 1e-5)  # Prevent /0
    r = 0.06
    P = df["option_price"].values

    # We need to distinguish Calls and Puts based on standard NSE OptnTp formats (CE/PE or Call/Put)
    # The normalizer captures 'option_type'. Typically 'CE' or 'PE'
    # Fill NA option types with empty string to avoid errors
    option_types = df["option_type"].fillna("").astype(str)
    is_call = option_types.str.upper().str.startswith("C").values

    import math

    def norm_cdf(x):
        return 0.5 * (1 + v_erf(x / np.sqrt(2)))

    def norm_pdf(x):
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    v_erf = np.vectorize(math.erf)

    def bs_price(S, K, T, r, sigma, is_call):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
        put_price = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

        return np.where(is_call, call_price, put_price)

    def bs_vega(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * norm_pdf(d1) * np.sqrt(T)

    print("\nCalculating TRUE_IV using Newton-Raphson Black-Scholes inversion...")

    # Vectorized Newton-Raphson
    sigma = np.full(len(S), 0.20)  # Initial guess = 0.20
    tol = 1e-5
    max_iter = 50

    for i in range(max_iter):
        price_est = bs_price(S, K, T, r, sigma, is_call)
        diff = price_est - P

        # If all diffs are within tolerance, we can stop early
        if np.all(np.abs(diff) < tol):
            break

        vega = bs_vega(S, K, T, r, sigma)

        # Avoid division by zero where vega is too small
        # Replace 0 vega with a tiny number to avoid inf, or skip update
        vega = np.where(vega < 1e-8, 1e-8, vega)

        step = diff / vega
        sigma -= step

        # Prevent sigma from going negative or insanely high
        sigma = np.clip(sigma, 1e-5, 5.0)

    # Filter out failures where the pricing diff never converged
    final_diff = np.abs(bs_price(S, K, T, r, sigma, is_call) - P)
    sigma = np.where(final_diff > 1e-2, np.nan, sigma)

    df["TRUE_IV"] = sigma
    iv_col = "TRUE_IV"

    df = df.dropna(subset=[iv_col, "underlying_price"])

    print("\nComputing VRP statistics...")
    # IV_t = mean implied vol per timestamp
    iv_t = df.groupby("timestamp")[iv_col].mean()

    # SPOT_t = mean underlying price per timestamp
    spot_t = df.groupby("timestamp")["underlying_price"].mean()

    returns = np.log(spot_t).diff()
    rv_t = returns.rolling(20).std() * np.sqrt(252)

    aligned = pd.DataFrame({"TRUE_IV": iv_t, "RV": rv_t, "SPOT": spot_t}).dropna()

    if len(aligned) == 0:
        print("Error: No overlapping days for IV and RV. Dataset might be too small.")
        sys.exit(1)

    aligned["VRP"] = aligned["TRUE_IV"] - aligned["RV"]

    print("\n====== VRP RESULTS ======")
    print(f"Mean IV                 : {aligned['TRUE_IV'].mean():.4f}")
    print(f"Mean RV                 : {aligned['RV'].mean():.4f}")
    print(f"Mean VRP gap            : {aligned['VRP'].mean():.4f}")

    pct = (aligned["VRP"] > 0).mean()
    print(f"Percent of days IV > RV : {pct:.2%}")

    print("\n====== INTERPRETATION ======")
    if pct > 0.60:
        print("VOLATILITY RISK PREMIUM CONFIRMED IN INDIAN MARKET")
    else:
        print("NO STRONG VRP EDGE FOUND")

    # Save output
    output_dir = repo_root / "research_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "vrp_nifty.parquet"

    try:
        aligned.to_parquet(out_file)
        print(f"\nSaved aligned VRP dataframe to: {out_file}")
    except Exception as e:
        print(f"Error saving output dataframe: {e}")


if __name__ == "__main__":
    main()
