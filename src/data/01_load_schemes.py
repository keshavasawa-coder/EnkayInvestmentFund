"""
01_load_schemes.py
Fetches the master mutual fund scheme list from the AMFI India portal.
Filters to Open-Ended, Regular Plan, Growth schemes only.
Outputs: data/processed/scheme_master.csv
"""
import os
import io
import requests
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "scheme_master.csv")

AMFI_URL = "https://portal.amfiindia.com/DownloadSchemeData_Po.aspx?mf=0"

# ── SEBI category → (category, sub_category) mapping ────────────────────────
# The AMFI portal stores categories like "Equity Scheme - Large Cap Fund"
# We split on " - " to get the broad category and specific sub-category.
CATEGORY_MAP = {
    # Equity
    "Equity Scheme - Large Cap Fund":                "Equity",
    "Equity Scheme - Large & Mid Cap Fund":          "Equity",
    "Equity Scheme - Flexi Cap Fund":                "Equity",
    "Equity Scheme - Multi Cap Fund":                "Equity",
    "Equity Scheme - Mid Cap Fund":                  "Equity",
    "Equity Scheme - Small Cap Fund":                "Equity",
    "Equity Scheme - Value Fund/ Contra Fund":       "Equity",
    "Equity Scheme - ELSS":                          "Equity",
    "Equity Scheme - Dividend Yield Fund":           "Equity",
    "Equity Scheme - Focused Fund":                  "Equity",
    "Equity Scheme - Sectoral/ Thematic":            "Equity",
    # Debt
    "Debt Scheme - Long Duration Fund":              "Debt",
    "Debt Scheme - Medium to Long Duration Fund":    "Debt",
    "Debt Scheme - Medium Duration Fund":            "Debt",
    "Debt Scheme - Short Duration Fund":             "Debt",
    "Debt Scheme - Low Duration Fund":               "Debt",
    "Debt Scheme - Ultra Short Duration Fund":       "Debt",
    "Debt Scheme - Money Market Fund":               "Debt",
    "Debt Scheme - Liquid Fund":                     "Debt",
    "Debt Scheme - Overnight Fund":                  "Debt",
    "Debt Scheme - Dynamic Bond":                    "Debt",
    "Debt Scheme - Corporate Bond Fund":             "Debt",
    "Debt Scheme - Credit Risk Fund":                "Debt",
    "Debt Scheme - Banking and PSU Fund":            "Debt",
    "Debt Scheme - Floater Fund":                    "Debt",
    "Debt Scheme - Gilt Fund":                       "Debt",
    "Debt Scheme - Gilt Fund with 10 year constant duration": "Debt",
    # Hybrid
    "Hybrid Scheme - Conservative Hybrid Fund":      "Hybrid",
    "Hybrid Scheme - Balanced Hybrid Fund":          "Hybrid",
    "Hybrid Scheme - Aggressive Hybrid Fund":        "Hybrid",
    "Hybrid Scheme - Dynamic Asset Allocation":      "Hybrid",
    "Hybrid Scheme - Multi Asset Allocation":        "Hybrid",
    "Hybrid Scheme - Arbitrage Fund":                "Hybrid",
    "Hybrid Scheme - Equity Savings":                "Hybrid",
    # Solution Oriented
    "Solution Oriented Scheme - Retirement Fund":    "Solution Oriented",
    "Solution Oriented Scheme - Children's Fund":    "Solution Oriented",
    # Other
    "Other Scheme - Index Funds":                    "Other",
    "Other Scheme - Other  ETFs":                    "Other",
    "Other Scheme - Gold ETF":                       "Other",
    "Other Scheme - FoF Overseas":                   "Other",
    "Other Scheme - FoF Domestic":                   "Other",
}


def extract_sub_category(scheme_category: str) -> str:
    """Extract sub-category from the full SEBI category string.
    E.g. 'Equity Scheme - Large Cap Fund' → 'Large Cap Fund'
    """
    if " - " in scheme_category:
        return scheme_category.split(" - ", 1)[1].strip()
    return scheme_category


def extract_category(scheme_category: str) -> str:
    """Extract broad category from the full SEBI category string."""
    return CATEGORY_MAP.get(scheme_category, "Unknown")


def is_regular_growth(scheme_nav_name: str) -> bool:
    """Check if a scheme variant is Regular Plan + Growth."""
    if not isinstance(scheme_nav_name, str):
        return False
    name_lower = scheme_nav_name.lower()
    # Must be Regular (not Direct)
    if "direct" in name_lower:
        return False
    # Must be Growth (not IDCW, Dividend, Bonus)
    if "growth" not in name_lower:
        return False
    # Exclude IDCW, Dividend, Bonus explicitly
    exclude = ["idcw", "dividend", "bonus", "payout"]
    for ex in exclude:
        if ex in name_lower:
            return False
    return True


def main():
    print(f"Fetching scheme list from AMFI portal...")
    resp = requests.get(AMFI_URL, timeout=30)
    resp.raise_for_status()

    # The portal returns CSV-like data
    # Columns: AMC, Code, Scheme Name, Scheme Type, Scheme Category,
    #          Scheme NAV Name, Scheme Minimum Amount, Launch Date,
    #          Closure Date, ISIN Div Payout/ ISIN Growth, ISIN Div Reinvestment
    raw_text = resp.text
    df = pd.read_csv(io.StringIO(raw_text), sep=",", dtype=str)

    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    print(f"  Total raw schemes: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Filter: Open Ended only
    df = df[df["Scheme Type"].str.strip() == "Open Ended"].copy()
    print(f"  After Open Ended filter: {len(df)}")

    # Filter: Regular Plan + Growth only
    df["_is_reg_growth"] = df["Scheme NAV Name"].apply(is_regular_growth)
    df = df[df["_is_reg_growth"]].copy()
    df = df.drop(columns=["_is_reg_growth"])
    print(f"  After Regular+Growth filter: {len(df)}")

    # Map categories
    df["Scheme Category"] = df["Scheme Category"].str.strip()
    df["category"] = df["Scheme Category"].apply(extract_category)
    df["sub_category"] = df["Scheme Category"].apply(extract_sub_category)

    # Drop unknowns
    unknown_mask = df["category"] == "Unknown"
    if unknown_mask.any():
        unknown_cats = df.loc[unknown_mask, "Scheme Category"].unique()
        print(f"  [WARN] Unknown categories ({len(unknown_cats)}): {list(unknown_cats)}")
    df = df[~unknown_mask].copy()
    print(f"  After removing unknown categories: {len(df)}")

    # Build output
    out = pd.DataFrame({
        "scheme_code": df["Code"].astype(int),
        "scheme_name": df["Scheme NAV Name"].str.strip(),
        "amc": df["AMC"].str.strip(),
        "scheme_type": df["Scheme Type"].str.strip(),
        "scheme_category_full": df["Scheme Category"],
        "category": df["category"],
        "sub_category": df["sub_category"],
    })

    # Deduplicate (some schemes may have multiple rows)
    out = out.drop_duplicates(subset=["scheme_code"], keep="first")
    print(f"  After dedup: {len(out)}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"\n[OK] scheme_master.csv saved -> {OUT_PATH}")
    print(f"   Total schemes: {len(out)}")
    print(f"   Categories: {out['category'].value_counts().to_dict()}")
    print(f"   Sub-categories ({out['sub_category'].nunique()}): "
          f"{sorted(out['sub_category'].unique().tolist())}")


if __name__ == "__main__":
    main()
