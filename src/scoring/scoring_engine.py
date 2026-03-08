"""
scoring_engine.py
Computes a Composite Score (0–100) for each fund in master_scheme_table.csv.
Ranks funds within each sub-category for three risk profiles.
Outputs: data/processed/ranked_funds.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..", "..")
MASTER     = os.path.join(BASE_DIR, "data", "processed", "master_scheme_table.csv")
OUT_PATH   = os.path.join(BASE_DIR, "data", "processed", "ranked_funds.csv")

# ── Default component weights (must sum to 1.0) ─────────────────────────────
# TieUp is FIXED at 5% for all profiles — the remaining 95% is split across
# Return, Alpha, Brokerage, and AUM depending on the risk profile.
DEFAULT_WEIGHTS = {
    "conservative": {
        "return":   0.40,  # Strong emphasis on track record
        "alpha":    0.00,  # Not relevant for conservative
        "brokerage":0.40,  # High brokerage focus
        "aum":      0.15,  # AUM reliability
        "tieup":    0.05,  # Fixed across all profiles
    },
    "moderate": {
        "return":   0.35,
        "alpha":    0.10,
        "brokerage":0.30,
        "aum":      0.20,
        "tieup":    0.05,
    },
    "aggressive": {
        "return":   0.40,
        "alpha":    0.15,
        "brokerage":0.20,
        "aum":      0.20,
        "tieup":    0.05,
    },
}

# Return period weights (how much 1Y / 3Y / 5Y contribute to the return score)
RETURN_PERIOD_WEIGHTS = {"1y": 0.40, "3y": 0.35, "5y": 0.25}

TIEUP_POINTS = {"A": 10.0, "B": 5.0, "None": 0.0}


def minmax_scale_series(s: pd.Series) -> pd.Series:
    """Min-max normalise a series to [0, 10]. NaN-safe."""
    not_null = s.notna()
    result   = s.copy().astype(float)
    if not_null.sum() < 2:
        result[not_null] = 5.0   # Assign neutral if too few values
        return result
    scaler = MinMaxScaler(feature_range=(0, 10))
    result[not_null] = scaler.fit_transform(s[not_null].values.reshape(-1, 1)).flatten()
    return result


def compute_return_score(df: pd.DataFrame) -> pd.Series:
    """
    Weighted average of 1Y, 3Y, 5Y Regular returns (normalised within sub-category).
    Falls back gracefully when a period is unavailable.
    """
    cols = {
        "1y": "return_1y_regular",
        "3y": "return_3y_regular",
        "5y": "return_5y_regular",
    }
    scores = pd.DataFrame(index=df.index)
    for period, col in cols.items():
        if col in df.columns:
            scores[period] = minmax_scale_series(df[col])
        else:
            scores[period] = np.nan

    # Weighted average ignoring missing periods (re-normalise weights)
    def weighted_avg_row(row):
        total_w, total_s = 0.0, 0.0
        for p, w in RETURN_PERIOD_WEIGHTS.items():
            if not np.isnan(row.get(p, np.nan)):
                total_s += w * row[p]
                total_w += w
        return (total_s / total_w) if total_w > 0 else np.nan

    return scores.apply(weighted_avg_row, axis=1)


def compute_alpha_score(df: pd.DataFrame) -> pd.Series:
    """1-Year Information Ratio (Regular), normalised within sub-category."""
    col = "info_ratio_1y_regular"
    if col in df.columns:
        return minmax_scale_series(df[col])
    return pd.Series(np.nan, index=df.index)


def compute_brokerage_score(df: pd.DataFrame) -> pd.Series:
    """Trail brokerage rate normalised within sub-category."""
    col = "trail_brokerage_incl_gst"
    if col in df.columns:
        return minmax_scale_series(df[col])
    return pd.Series(0.0, index=df.index)


def compute_tieup_score(df: pd.DataFrame) -> pd.Series:
    """Flat bonus: A=10, B=5, None/NaN=0. Weight is fixed at 5% in all profiles."""
    return df["tieup_category"].map(TIEUP_POINTS).fillna(0.0)


def compute_aum_score(df: pd.DataFrame) -> pd.Series:
    """AUM (Cr.) normalised within sub-category. Higher AUM = more reliable fund."""
    col = "aum_cr"
    if col in df.columns:
        return minmax_scale_series(df[col])
    return pd.Series(5.0, index=df.index)  # neutral if missing


def score_group(group: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Compute composite score for all funds in a single sub-category."""
    g = group.copy()

    g["_return_score"]    = compute_return_score(g)
    g["_alpha_score"]     = compute_alpha_score(g)
    g["_brokerage_score"] = compute_brokerage_score(g)
    g["_tieup_score"]     = compute_tieup_score(g)
    g["_aum_score"]       = compute_aum_score(g)

    # Composite (all components already in [0,10] range)
    g["composite_score"] = (
        weights["return"]    * g["_return_score"].fillna(5) +
        weights["alpha"]     * g["_alpha_score"].fillna(0) +
        weights["brokerage"] * g["_brokerage_score"].fillna(0) +
        weights["aum"]       * g["_aum_score"].fillna(5) +
        weights["tieup"]     * g["_tieup_score"]
    )
    # Scale to 0–100
    g["composite_score"] = (g["composite_score"] / 10) * 100
    g["composite_score"] = g["composite_score"].round(2)

    # Expose individual component scores for the dashboard
    g["score_return"]    = g["_return_score"].round(3)
    g["score_alpha"]     = g["_alpha_score"].round(3)
    g["score_brokerage"] = g["_brokerage_score"].round(3)
    g["score_tieup"]     = g["_tieup_score"].round(3)
    g["score_aum"]       = g["_aum_score"].round(3)

    g = g.drop(columns=["_return_score", "_alpha_score", "_brokerage_score",
                         "_tieup_score", "_aum_score"])
    return g


def rank_all(df: pd.DataFrame, risk_profile: str, weights: dict) -> pd.DataFrame:
    """Score and rank all funds, grouped by sub-category."""
    frames = []
    for sub_cat, grp in df.groupby("sub_category"):
        scored = score_group(grp, weights)
        scored["rank"] = scored["composite_score"].rank(
            method="min", ascending=False, na_option="bottom"
        ).astype(int)
        frames.append(scored)
    result = pd.concat(frames, ignore_index=True)
    result["risk_profile"] = risk_profile
    return result


def main(weights: dict = None):
    print(f"Loading master table: {MASTER}")
    df = pd.read_csv(MASTER)
    print(f"  Funds loaded: {len(df)}")

    all_ranked = []
    effective_weights = weights or DEFAULT_WEIGHTS

    for profile, w in effective_weights.items():
        print(f"  Scoring [{profile}]...")
        ranked = rank_all(df, profile, w)
        all_ranked.append(ranked)

    result = pd.concat(all_ranked, ignore_index=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    result.to_csv(OUT_PATH, index=False)

    print(f"\n[OK] ranked_funds.csv saved -> {OUT_PATH}")
    print(f"   Rows: {len(result)}  (= {len(df)} funds × {len(effective_weights)} risk profiles)")
    print(f"   Composite score range: {result['composite_score'].min():.1f} – {result['composite_score'].max():.1f}")

    # Quick sanity: top 5 Large Cap funds under Moderate
    top5 = (
        result[(result["sub_category"] == "Large Cap Fund") & (result["risk_profile"] == "moderate")]
        .nsmallest(5, "rank")[["scheme_name", "sub_category", "composite_score",
                                "trail_brokerage_incl_gst", "tieup_category", "rank"]]
    )
    if not top5.empty:
        print(f"\n   Top 5 Large Cap (Moderate):\n{top5.to_string(index=False)}")

    return result


if __name__ == "__main__":
    main()
