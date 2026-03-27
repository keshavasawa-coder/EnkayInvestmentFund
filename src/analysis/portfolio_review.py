"""
portfolio_review.py
Analyzes the firm's current AUM holdings (from Scheme_wise_AUM.xls) and flags
schemes with high AUM but poor performance/brokerage. Suggests better alternatives.
Used by the dashboard Portfolio Exposure Review tab.
"""
import os
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
RANKED_FILE = os.path.join(BASE_DIR, "data", "processed", "ranked_funds.csv")
AUM_FILE = os.path.join(BASE_DIR, "Scheme_wise_AUM.xls")

FUZZY_THRESHOLD = 75

AUM_THRESHOLDS = {
    "25 Lakh": 2500000,
    "50 Lakh": 5000000,
    "75 Lakh": 7500000,
    "1 Cr": 10000000,
    "1.5 Cr": 15000000,
    "2 Cr": 20000000,
}


def load_aum_data(uploaded_file=None):
    """
    Load and clean the AUM data from file.
    
    Parameters:
    -----------
    uploaded_file : UploadedFile, optional
        Streamlit uploaded file object. If None, loads from default Scheme_wise_AUM.xls
    
    Returns:
    --------
    DataFrame with columns: sr_no, amc, scheme, nav, equity, debt, hybrid, physical_assets, others, total
    """
    import io
    
    if uploaded_file is not None:
        # Read uploaded file - auto-detect sheet name
        file_bytes = uploaded_file.read()
        
        # Try to find the AUM sheet
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        
        # Try common sheet names
        sheet_names_to_try = ["AUM Report", "AUM", "Sheet1", "Sheet0", xl.sheet_names[0]]
        df = None
        
        for sheet_name in sheet_names_to_try:
            try:
                # Try reading with header=1 (like the default)
                temp_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=1)
                # Check if it has the expected columns
                if any('scheme' in str(c).lower() for c in temp_df.columns):
                    df = temp_df
                    break
            except:
                continue
        
        if df is None:
            # Last resort: try without specifying header
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
    else:
        # Load from default file
        df = pd.read_excel(AUM_FILE, sheet_name="AUM Report", header=1)
    
    # Standardize column names
    df.columns = ["sr_no", "amc", "scheme", "nav", "equity", "debt", "hybrid", "physical_assets", "others", "total"]
    
    # Clean data
    df = df.dropna(subset=["scheme"])
    df["scheme"] = df["scheme"].astype(str).str.strip()
    df["amc"] = df["amc"].astype(str).str.strip()
    df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0)
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce").fillna(0)
    df["debt"] = pd.to_numeric(df["debt"], errors="coerce").fillna(0)
    df["hybrid"] = pd.to_numeric(df["hybrid"], errors="coerce").fillna(0)
    
    return df


def get_asset_class(row):
    """Determine asset class based on which column has the AUM."""
    if row["equity"] > 0:
        return "Equity"
    elif row["debt"] > 0:
        return "Debt"
    elif row["hybrid"] > 0:
        return "Hybrid"
    elif row["physical_assets"] > 0:
        return "Physical Assets"
    elif row["others"] > 0:
        return "Others"
    return "Unknown"


def load_ranked_funds():
    """Load the ranked funds data."""
    return pd.read_csv(RANKED_FILE)


def fuzzy_match_schemes(aum_schemes, ranked_schemes, threshold=FUZZY_THRESHOLD):
    """
    Match scheme names from AUM data to ranked funds using fuzzy matching.
    Returns dict: aum_scheme_name -> matched ranked fund name
    """
    matching = {}
    if hasattr(ranked_schemes, 'tolist'):
        ranked_list = ranked_schemes.tolist()
    else:
        ranked_list = list(ranked_schemes)
    
    for aum_scheme in aum_schemes:
        result = process.extractOne(
            aum_scheme, ranked_list,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result:
            matching[aum_scheme] = result[0]
    return matching


def flag_underperforming_schemes(
    aum_df,
    ranked_df,
    risk_profile="moderate",
    aum_threshold=2500000,
    score_percentile=50,
    include_brokerage_flag=True,
):
    """
    Flag schemes with high AUM but poor metrics.
    
    Parameters:
    -----------
    aum_df : DataFrame - AUM holdings data
    ranked_df : DataFrame - ranked funds data (already filtered by risk_profile)
    risk_profile : str - 'conservative', 'moderate', or 'aggressive'
    aum_threshold : int - minimum AUM to consider (default 25 lakh)
    score_percentile : int - flag if score is below this percentile (default 50 = bottom 50%)
    include_brokerage_flag : bool - also flag schemes with below-median brokerage
    
    Returns:
    --------
    dict with:
      - 'flagged': DataFrame of flagged schemes with alternatives
      - 'summary': Summary statistics
      - 'aum_by_asset': AUM breakdown by asset class
    """
    ranked = ranked_df[ranked_df["risk_profile"] == risk_profile].copy()
    
    aum_with_asset = aum_df.copy()
    aum_with_asset["asset_class"] = aum_with_asset.apply(get_asset_class, axis=1)
    
    match_map = fuzzy_match_schemes(
        aum_with_asset["scheme"].tolist(),
        ranked["scheme_name"].tolist()
    )
    
    aum_with_asset["matched_scheme"] = aum_with_asset["scheme"].map(match_map)
    
    aum_matched = aum_with_asset[aum_with_asset["matched_scheme"].notna()].copy()
    
    merged = aum_matched.merge(
        ranked[["scheme_name", "composite_score", "score_return", "score_brokerage", 
                "score_aum", "trail_brokerage_incl_gst", "nav_regular", "return_1y_regular", 
                "return_3y_regular", "return_5y_regular", "rank", "sub_category", "category"]],
        left_on="matched_scheme",
        right_on="scheme_name",
        how="left",
        suffixes=("_aum", "_ranked")
    )
    
    merged = merged[merged["total"] >= aum_threshold].copy()
    
    if merged.empty:
        return {
            "flagged": pd.DataFrame(),
            "summary": {
                "total_aum": aum_df["total"].sum(),
                "total_schemes": len(aum_df),
                "matched_schemes": len(aum_matched),
                "flagged_count": 0,
            },
            "aum_by_asset": aum_with_asset.groupby("asset_class")["total"].sum().to_dict(),
        }
    
    for sub_cat, group in merged.groupby("sub_category"):
        if group["composite_score"].notna().any():
            median_score = group["composite_score"].median()
            merged.loc[merged["sub_category"] == sub_cat, "score_median"] = median_score
    
    merged["score_flag"] = merged["composite_score"] < merged["score_median"]
    
    if include_brokerage_flag:
        for sub_cat, group in merged.groupby("sub_category"):
            if group["trail_brokerage_incl_gst"].notna().any():
                median_brok = group["trail_brokerage_incl_gst"].median()
                merged.loc[merged["sub_category"] == sub_cat, "brokerage_median"] = median_brok
        
        merged["brokerage_flag"] = (
            merged["trail_brokerage_incl_gst"] < merged["brokerage_median"]
        ) & merged["trail_brokerage_incl_gst"].notna()
    else:
        merged["brokerage_flag"] = False
    
    merged["is_flagged"] = merged["score_flag"] | merged["brokerage_flag"]
    
    flagged = merged[merged["is_flagged"]].copy()
    
    summary = {
        "total_aum": aum_df["total"].sum(),
        "total_schemes": len(aum_df),
        "matched_schemes": len(aum_matched),
        "schemes_above_threshold": len(merged),
        "flagged_count": len(flagged),
    }
    
    aum_by_asset = aum_with_asset.groupby("asset_class")["total"].sum().to_dict()
    
    return {
        "flagged": flagged,
        "summary": summary,
        "aum_by_asset": aum_by_asset,
        "all_holdings": merged,
    }


def get_alternatives_for_flagged(flagged_df, ranked_df, risk_profile="moderate", n=3):
    """
    Get better alternatives for each flagged scheme.
    """
    ranked = ranked_df[ranked_df["risk_profile"] == risk_profile].copy()
    
    results = []
    
    for _, row in flagged_df.iterrows():
        sub_cat = row.get("sub_category")
        if pd.isna(sub_cat):
            continue
        
        peers = ranked[ranked["sub_category"] == sub_cat].copy()
        
        if peers.empty:
            continue
        
        current_brok = row.get("trail_brokerage_incl_gst", 0)
        if pd.notna(current_brok) and current_brok > 0:
            peers = peers[peers["trail_brokerage_incl_gst"] >= current_brok]
        
        peers = peers.nlargest(n, "composite_score")
        
        for _, alt in peers.iterrows():
            results.append({
                "flagged_scheme": row.get("scheme", row.get("scheme_aum", "")),
                "flagged_aum": row["total"],
                "flagged_score": row["composite_score"],
                "flagged_brokerage": row["trail_brokerage_incl_gst"],
                "alternative_scheme": alt["scheme_name"],
                "alternative_score": alt["composite_score"],
                "alternative_brokerage": alt.get("trail_brokerage_incl_gst"),
                "alternative_return_1y": alt.get("return_1y_regular"),
                "alternative_return_3y": alt.get("return_3y_regular"),
                "alternative_return_5y": alt.get("return_5y_regular"),
                "alternative_aum": alt.get("aum_cr"),
                "alternative_tieup": alt.get("tieup_category"),
                "alternative_rank": alt["rank"],
                "sub_category": sub_cat,
            })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Loading AUM data...")
    aum_df = load_aum_data()
    print(f"  Loaded {len(aum_df)} schemes from AUM report")
    
    print("\nLoading ranked funds...")
    ranked_df = load_ranked_funds()
    print(f"  Loaded {len(ranked_df)} funds")
    
    print("\nAnalyzing portfolio exposure...")
    result = flag_underperforming_schemes(
        aum_df, ranked_df,
        risk_profile="moderate",
        aum_threshold=2500000,
        score_percentile=50,
        include_brokerage_flag=True,
    )
    
    print(f"\n=== Summary ===")
    print(f"Total AUM: ₹{result['summary']['total_aum']:,.0f}")
    print(f"Total Schemes: {result['summary']['total_schemes']}")
    print(f"Matched Schemes: {result['summary']['matched_schemes']}")
    print(f"Schemes above threshold: {result['summary']['schemes_above_threshold']}")
    print(f"Flagged Schemes: {result['summary']['flagged_count']}")
    
    print(f"\n=== AUM by Asset Class ===")
    for asset, aum in result["aum_by_asset"].items():
        print(f"  {asset}: ₹{aum:,.0f}")
    
    if not result["flagged"].empty:
        print(f"\n=== Flagged Schemes (Top 10) ===")
        display_cols = ["scheme_aum", "amc_aum", "total", "composite_score", "trail_brokerage_incl_gst", "score_flag", "brokerage_flag"]
        available_cols = [c for c in display_cols if c in result["flagged"].columns]
        print(result["flagged"][available_cols].head(10).to_string())
