# value_model.py
import numpy as np
import pandas as pd


def compute_value_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe of cars with columns:
      - price_sgd
      - mileage_km
      - depreciation_per_year
      - year
    add a 'value_score' (0–100) and rank them.
    """

    if df.empty:
        df["value_score"] = []
        df["value_rank"] = []
        return df

    df = df.copy()

    # Helper: normalize a column to 0–1, ignoring NaNs
    def norm_col(col, higher_is_better: bool = False):
        x = df[col].astype("float64")
        mask = x.notna()
        vals = x[mask]
        if vals.empty:
            return pd.Series([np.nan] * len(df), index=df.index)

        vmin = vals.min()
        vmax = vals.max()
        if vmax == vmin:
            # all same
            return pd.Series([0.5 if mask[i] else np.nan for i in range(len(df))], index=df.index)

        if higher_is_better:
            norm = (x - vmin) / (vmax - vmin)
        else:
            norm = (vmax - x) / (vmax - vmin)
        return norm

    # define derived "age"
    current_year = 2025  # or datetime.now().year
    df["car_age"] = current_year - df["year"] if "year" in df else np.nan

    # Normalize:
    # - lower price is better
    # - lower depreciation is better
    # - lower mileage is better
    # - lower age is better (newer car)
    df["price_norm"] = norm_col("price_sgd", higher_is_better=False)
    df["depr_norm"] = norm_col("depreciation_per_year", higher_is_better=False)
    df["mileage_norm"] = norm_col("mileage_km", higher_is_better=False)
    df["age_norm"] = norm_col("car_age", higher_is_better=False)

    # Weighted score (tune weights as you like)
    # We treat missing as 0.5 (neutral) in the score
    def safe_val(x):
        return 0.5 if pd.isna(x) else x

    scores = []
    for _, row in df.iterrows():
        price_n = safe_val(row["price_norm"])
        depr_n = safe_val(row["depr_norm"])
        mile_n = safe_val(row["mileage_norm"])
        age_n  = safe_val(row["age_norm"])
        # weights must sum to 1
        score = (
            0.35 * price_n +
            0.30 * depr_n +
            0.20 * mile_n +
            0.15 * age_n
        )
        scores.append(score)

    df["value_score"] = np.round(np.array(scores) * 100, 1)

    # Rank (1 = best)
    df = df.sort_values(by="value_score", ascending=False).reset_index(drop=True)
    df["value_rank"] = df.index + 1

    return df
