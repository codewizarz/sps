import polars as pl
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ForecastCone:
    horizon_days: int
    p10: float
    p50: float
    p90: float
    spike_prob: float  # Probability of > 20% increase
    similar_dates: List[str]  # For interpretability


class VolForecastEngine:
    """
    Predicts future volatility distribution using Regime-Conditioned Historical Simulation.
    """

    def __init__(self, history: Optional[pl.DataFrame] = None):
        self.history = history
        self.feature_cols = ["iv_rank", "vol_momentum", "term_slope", "vvix"]
        self.features_normalized = None

    def train(self, history: pl.DataFrame):
        """
        Ingests historical data and pre-calculates normalized features.
        Expected cols: date, iv, iv_rank, vol_momentum, term_slope, vvix
        """
        self.history = history

        # Normalize features (Z-Score) to ensure Euclidean distance helps
        # We store mean/std for normalizing the query vector later
        self.stats = {}
        for col in self.feature_cols:
            if col in history.columns:
                mean = history[col].mean()
                std = history[col].std()
                if std == 0:
                    std = 1.0
                self.stats[col] = (mean, std)
            else:
                self.stats[col] = (0.0, 1.0)  # Default

        # Create a matrix of normalized features
        # We can use Polars expressions
        exprs = []
        for col in self.feature_cols:
            if col in history.columns:
                mu, sigma = self.stats[col]
                exprs.append(((pl.col(col) - mu) / sigma).alias(f"{col}_norm"))
            else:
                exprs.append(pl.lit(0.0).alias(f"{col}_norm"))

        self.features_normalized = history.with_columns(exprs)

    def forecast(
        self, current_state: dict, horizon_days: int = 5, k_neighbors: int = 50
    ) -> ForecastCone:
        """
        Finds K nearest neighbors to current_state and constructs forecast cone.
        current_state: dict with keys matching feature_cols
        """
        if self.features_normalized is None or self.history is None:
            return ForecastCone(horizon_days, 0, 0, 0, 0, [])

        # 1. Normalize Query Vector
        query_vec = []
        for col in self.feature_cols:
            val = current_state.get(col, 0.0)
            mu, sigma = self.stats[col]
            query_vec.append((val - mu) / sigma)

        query_vec_np = np.array(query_vec)

        # 2. Calculate Distances (Vectorized)
        # Extract feature matrix as numpy
        # Note: In prod for huge datasets, use Sklearn KDTree. For <10k rows, brute force is instant.
        feat_matrix = self.features_normalized.select(
            [f"{c}_norm" for c in self.feature_cols]
        ).to_numpy()

        # Euclidean Distance: sqrt(sum((x - y)^2))
        dists = np.linalg.norm(feat_matrix - query_vec_np, axis=1)

        # 3. Find Indices of K Nearest
        # Sort distances
        nearest_indices = np.argsort(dists)[:k_neighbors]

        # 4. Retrieve Outcomes
        # We want to know how IV changed over the next 'horizon_days' for these indices.
        # We need to look ahead in the history.
        # Assuming dataframe is sorted by date? We should rely on date or index.
        # Let's assume input history is time-sorted.

        # Get current IV from history at those indices
        # And future IV at index + horizon

        # Filter valid indices (must have room for horizon)
        max_idx = len(self.history) - horizon_days - 1
        valid_indices = [i for i in nearest_indices if i <= max_idx]

        if not valid_indices:
            return ForecastCone(horizon_days, 0, 0, 0, 0, [])

        future_changes = []
        similar_dates = []

        # Optimization: use Polars slicing if possible, but list comprehension is fine for K=50
        iv_series = self.history["iv"]
        date_series = self.history["date"]

        current_iv_est = current_state.get("iv", 0.0)
        if current_iv_est == 0:
            # If not provided, we can't project absolute levels well, but we can project % change.
            # Ideally current_state includes 'iv'.
            current_iv_est = 1.0  # Fallback to prevent div/0, assumes percentage output

        forecast_values = []

        for idx in valid_indices:
            start_iv = iv_series[idx]
            end_iv = iv_series[idx + horizon_days]

            # Multiplicative Change
            if start_iv > 0:
                change = end_iv / start_iv
                forecast_values.append(current_iv_est * change)

            similar_dates.append(str(date_series[idx]))

        if not forecast_values:
            return ForecastCone(
                horizon_days, current_iv_est, current_iv_est, current_iv_est, 0, []
            )

        # 5. Build Distribution
        forecasts = np.array(forecast_values)
        p10 = np.percentile(forecasts, 10)
        p50 = np.percentile(forecasts, 50)
        p90 = np.percentile(forecasts, 90)

        # Spike Prob: > 20% increase
        spike_threshold = current_iv_est * 1.20
        spike_prob = np.mean(forecasts > spike_threshold)

        return ForecastCone(
            horizon_days=horizon_days,
            p10=float(p10),
            p50=float(p50),
            p90=float(p90),
            spike_prob=float(spike_prob),
            similar_dates=similar_dates[:5],  # Return top 5 matches
        )
