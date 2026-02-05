import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from typing import Dict, List, Optional


class PortfolioOptimizer:
    """
    Implements Hierarchical Risk Parity (HRP) and Convexity Boosting.
    """

    def optimize_weights(
        self, returns_df: pd.DataFrame, use_convexity: bool = True
    ) -> Dict[str, float]:
        """
        Calculates optimal weights for a set of strategies given their historical returns.

        Args:
            returns_df: DataFrame where columns are Strategy Names, index is Date.
            use_convexity: If True, boosts weights of positively skewed strategies.

        Returns:
            Dictionary of {StrategyName: Weight}
        """
        if returns_df.empty:
            return {}

        columns = returns_df.columns.tolist()

        # 1. HRP Allocation

        # Correlation Matrix
        corr = returns_df.corr().fillna(0)
        cov = returns_df.cov().fillna(0)

        # Distance Matrix
        # d(i,j) = sqrt(0.5 * (1 - rho(i,j)))
        dist = np.sqrt(0.5 * (1 - corr))
        link = sch.linkage(squareform(dist), "single")

        # Quasi-Diagonalization (Sort indices)
        ordered_indices = self._get_quasi_diag(link)
        ordered_columns = [columns[i] for i in ordered_indices]

        # Recursively Bisect (Weight Allocation)
        hrp_weights = self._get_rec_bisection(cov, ordered_columns, ordered_indices)

        # Convert to Dict
        final_weights = hrp_weights.to_dict()

        # 2. Convexity Boosting
        if use_convexity:
            skewness = returns_df.skew()
            # Boost = (1 + Skew/2).
            # E.g. Skew=1.0 -> Boost=1.5. Skew=-1.0 -> Boost=0.5.
            # We cap the boost/penalty to avoid extreme behavior.

            for strat in final_weights:
                skew_val = skewness.get(strat, 0.0)
                # Dampen skew impact
                boost_factor = 1 + (skew_val * 0.5)
                boost_factor = max(0.5, min(1.5, boost_factor))  # Clamp 0.5 to 1.5

                final_weights[strat] *= boost_factor

            # Re-normalize
            total_w = sum(final_weights.values())
            for strat in final_weights:
                final_weights[strat] /= total_w

        return final_weights

    # --- HRP Helpers ---

    def _get_quasi_diag(self, link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return sort_ix.tolist()

    def _get_rec_bisection(self, cov, sort_ix, sort_ix_int):
        w = pd.Series(1, index=sort_ix)
        c_items = [sort_ix]

        while len(c_items) > 0:
            c_items = [
                i[j:k]
                for i in c_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]  # Cluster 1
                c_items1 = c_items[i + 1]  # Cluster 2

                c_var0 = self._get_cluster_var(cov, c_items0)
                c_var1 = self._get_cluster_var(cov, c_items1)

                alpha = 1 - c_var0 / (c_var0 + c_var1)
                w[c_items0] *= alpha
                w[c_items1] *= 1 - alpha

        return w

    def _get_cluster_var(self, cov, c_items):
        cov_sub = cov.loc[c_items, c_items]
        w = np.linalg.inv(cov_sub).sum(axis=1)  # Min Variance weights for cluster
        w = pd.Series(w, index=cov_sub.index)
        w = w / w.sum()  # Normalize

        # V_cluster = w' * Cov * w
        # Often approximated as 1/variance for HRP, but let's use IVP (Inverse Variance) logic simplifed
        # Standard HRP uses IVP variance not full MinVar
        # Re-simplifying to Standard HRP:
        # Var = Sum(Cov) / N^2 ?? No.
        # Standard HRP uses specific variance formula.

        # Let's use Inverse Variance Allocation for the cluster split
        # Var_cluster = (TR(Inverse(Cov)))^-1

        # Actually, simpler:
        # returns variance of a hierarchy based on inverse variance weighting
        # w = 1/diag(cov)
        ivp = 1.0 / np.diag(cov_sub)
        ivp /= ivp.sum()
        w = pd.Series(ivp, index=cov_sub.index)

        cluster_var = np.dot(np.dot(w.T, cov_sub), w)
        return cluster_var
