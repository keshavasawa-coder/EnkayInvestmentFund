"""
fund_shift.py
Given a selected fund, suggests top-3 alternatives in the same sub-category
that offer equal or better brokerage with similar composite performance.
Used by the dashboard Fund Shift Advisor tab.
"""
import os
import pandas as pd

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..", "..")
RANKED_FILE = os.path.join(BASE_DIR, "data", "processed", "ranked_funds.csv")


def suggest_alternatives(
    scheme_name: str,
    risk_profile: str = "moderate",
    n: int = 3,
    df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Finds alternative funds in the same sub-category that:
      1. Have brokerage >= selected fund's brokerage (primary filter)
      2. Are in the top-50% of the sub-category by composite score
      3. Are ranked by composite score descending

    Parameters
    ----------
    scheme_name  : str   – Exact name of the selected fund.
    risk_profile : str   – 'conservative', 'moderate', or 'aggressive'.
    n            : int   – Number of alternatives to return (default 3).
    df           : DataFrame – Pre-loaded ranked_funds df (optional).

    Returns
    -------
    pd.DataFrame with alternatives + delta columns vs the selected fund.
    Returns empty DataFrame if the fund is not found or no alternatives exist.
    """
    if df is None:
        df = pd.read_csv(RANKED_FILE)

    df = df[df["risk_profile"] == risk_profile].copy()

    # Locate selected fund
    selected_rows = df[df["scheme_name"] == scheme_name]
    if selected_rows.empty:
        print(f"[WARN] Fund not found in ranked data: '{scheme_name}'")
        return pd.DataFrame()

    selected = selected_rows.iloc[0]
    sub_cat  = selected["sub_category"]
    sel_brok = selected.get("trail_brokerage_incl_gst", None)
    sel_score= selected["composite_score"]

    # Peers in same sub-category (excluding the selected fund)
    peers = df[
        (df["sub_category"] == sub_cat) &
        (df["scheme_name"] != scheme_name)
    ].copy()

    if peers.empty:
        return pd.DataFrame()

    # ── Filter 1: brokerage ≥ selected fund's brokerage ───────────────────
    if pd.notna(sel_brok) and sel_brok > 0:
        peers = peers[peers["trail_brokerage_incl_gst"] >= sel_brok]

    # ── Filter 2: top-50% of sub-category by composite score ──────────────
    score_median = df[df["sub_category"] == sub_cat]["composite_score"].median()
    peers = peers[peers["composite_score"] >= score_median]

    if peers.empty:
        # Relax: drop the top-50% filter and just return best brokerage
        peers = df[
            (df["sub_category"] == sub_cat) &
            (df["scheme_name"] != scheme_name)
        ].copy()
        if pd.notna(sel_brok) and sel_brok > 0:
            peers = peers[peers["trail_brokerage_incl_gst"] >= sel_brok]

    # ── Sort by composite score, take top-n ───────────────────────────────
    peers = peers.nlargest(n, "composite_score")

    # ── Compute delta columns ─────────────────────────────────────────────
    peers = peers.copy()
    peers["delta_brokerage"]  = (peers["trail_brokerage_incl_gst"] - sel_brok).round(3)
    peers["delta_return_1y"]  = (peers["return_1y_regular"] - selected.get("return_1y_regular", None)).round(2)
    peers["delta_return_3y"]  = (peers["return_3y_regular"] - selected.get("return_3y_regular", None)).round(2)
    peers["delta_composite"]  = (peers["composite_score"] - sel_score).round(2)
    peers["selected_fund"]    = scheme_name

    cols = [
        "selected_fund", "scheme_name", "sub_category", "tieup_category",
        "return_1y_regular", "return_3y_regular", "return_5y_regular",
        "trail_brokerage_incl_gst", "composite_score", "rank",
        "delta_brokerage", "delta_return_1y", "delta_return_3y", "delta_composite",
    ]
    available = [c for c in cols if c in peers.columns]
    return peers[available].reset_index(drop=True)


if __name__ == "__main__":
    result = suggest_alternatives("Axis Large Cap Fund", risk_profile="moderate")
    print("Alternatives for 'Axis Large Cap Fund' (Moderate):")
    print(result.to_string(index=False) if not result.empty else "No alternatives found.")
