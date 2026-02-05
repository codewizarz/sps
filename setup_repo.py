import os
from pathlib import Path


def create_structure():
    root = Path("quant_repo")

    structure = {
        "data": ["loaders", "normalization", "contract_metadata"],
        "features": [
            "realized_volatility",
            "implied_volatility_surface",
            "skew_metrics",
            "term_structure",
            "liquidity_metrics",
        ],
        "signals": ["mispricing_signals", "volatility_arbitrage", "parity_checks"],
        "portfolio": ["position_sizing", "risk_constraints", "capital_allocation"],
        "execution": ["nautilus_strategies", "adapters"],
        "research": ["scripts"],
        "analytics": ["performance", "trade_analysis", "drawdowns", "exposure"],
        "testing": ["deterministic_backtests"],
        "config": [],
    }

    print(f"Creating repository at: {root.absolute()}")

    for main_module, sub_modules in structure.items():
        # Create module dir
        mod_path = root / main_module
        mod_path.mkdir(parents=True, exist_ok=True)
        (mod_path / "__init__.py").touch()

        for sub in sub_modules:
            # Create sub-module dir
            sub_path = mod_path / sub
            sub_path.mkdir(parents=True, exist_ok=True)
            (sub_path / "__init__.py").touch()
            print(f"  + {sub_path}")

    # Create root README
    with open(root / "README.md", "w") as f:
        f.write(
            "# Quantitative Research Repository\n\nModular NautilusTrader architecture.\n"
        )


if __name__ == "__main__":
    create_structure()
