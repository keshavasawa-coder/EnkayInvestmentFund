"""
peer_comparison.py
Returns a structured comparison DataFrame for all funds in the same sub-category
as one or more selected funds. Used by the dashboard Peer Comparison tab.
"""
import os
import pandas as pd

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..", "..")
RANKED_FILE = os.path.join(BASE_DIR, "data", "processed", "ranked_funds.csv")

DISPLAY_COLS = [
    "scheme_name", "sub_category", "tieup_category",
    "nav_regular", "return_1y_regular", "return_3y_regular", "return_5y_regular",
    "trail_brokerage_incl_gst", "aum_cr",
    "score_return", "score_brokerage", "score_aum", "score_tieup",
    "composite_score", "rank",
]


def get_peer_comparison(
    scheme_names: list,
    risk_profile: str = "moderate",
    df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    scheme_names : list of str  – Funds to include; all peers in the same sub-category are returned.
    risk_profile : str          – 'conservative', 'moderate', or 'aggressive'.
    df           : DataFrame    – Optionally pass pre-loaded ranked_funds df (avoids re-reading CSV).

    Returns
    -------
    pd.DataFrame with all columns in DISPLAY_COLS, sorted by rank ascending.
    """
    if df is None:
        df = pd.read_csv(RANKED_FILE)

    df = df[df["risk_profile"] == risk_profile]

    # Identify sub-categories of the selected funds
    selected = df[df["scheme_name"].isin(scheme_names)]
    if selected.empty:
        return pd.DataFrame(columns=DISPLAY_COLS)

    sub_cats = selected["sub_category"].unique().tolist()

    # Return all peers in those sub-categories
    peers = df[df["sub_category"].isin(sub_cats)].copy()

    # Flag which funds were explicitly selected
    peers["is_selected"] = peers["scheme_name"].isin(scheme_names)

    # Keep only available display columns
    available = [c for c in DISPLAY_COLS if c in peers.columns] + ["is_selected"]
    peers = peers[available].sort_values("rank")

    return peers.reset_index(drop=True)


if __name__ == "__main__":
    # Quick test
    result = get_peer_comparison(
        ["HDFC Large Cap Fund", "Axis Large Cap Fund"],
        risk_profile="moderate",
    )
    print(result.to_string(index=False))
