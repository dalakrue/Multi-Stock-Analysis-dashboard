from __future__ import annotations

import numpy as np
import pandas as pd

from common import engineer_data


def run_background_processing(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create stable processed + engineered copies without changing the source frame."""
    processed = df.copy()

    numeric = [
        c for c in [
            "Avg Regime Candle Rank",
            "Candle After Regime Start",
            "Regime Maturity Score",
            "Avg Regime Candle Count Before Regime Change",
            "Completed Regime Samples",
            "Completed Candle",
        ] if c in processed.columns
    ]
    for c in numeric:
        processed[c] = pd.to_numeric(processed[c], errors="coerce")
        processed[c] = processed[c].replace([np.inf, -np.inf], np.nan)
        if processed[c].notna().any():
            processed[c] = processed[c].fillna(processed[c].median())

    # Keep categorical missingness explicit for stable model/analysis behavior.
    for c in processed.select_dtypes(include=["object", "string", "category"]).columns:
        processed[c] = processed[c].astype("string").fillna("UNKNOWN")

    engineered = engineer_data(processed, "Sequential forecasting")
    return processed, engineered
