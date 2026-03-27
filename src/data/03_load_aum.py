"""
03_load_aum.py
Parses the AMFI Average AUM Excel file (average-aum.xlsx).
The file has a hierarchical structure:
  - Row 0 (header): Quarter label
  - Data row 0: column headers (AMFI Code, Scheme NAV Name, AUM excl FoF, FoF AUM)
  - AMC name rows: section headers for each mutual fund house
  - Category rows: sub-section headers (e.g., "Equity Scheme - Large Cap Fund")
  - Data rows: numeric AMFI Code, Scheme Name, AUM values (in Lakhs)
  - "Total" rows: aggregation rows to skip

Logic:
  1. Filter to only "Regular Plan" rows (exclude "Direct Plan")
  2. For each fund, sum the AUM across all payout options (Growth, IDCW, etc.)
  3. Convert from Lakhs to Crores (÷ 100)
  4. Output column: aum_cr (displayed as "Average AUM for the quarter")

Outputs: data/processed/scheme_aum.csv
"""
import os
import re
import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), "..", "..")
IN_FILE   = os.path.join(BASE_DIR, "average-aum.xlsx")
OUT_PATH  = os.path.join(BASE_DIR, "data", "processed", "scheme_aum.csv")

# Patterns to strip from the scheme name to get the base fund name
# Order matters: longer/more specific patterns first
PAYOUT_SUFFIXES = [
    # Explicit "Plan - Option" combinations
    r'\s+-\s+Growth\s+Plan\s+-\s+Growth\s+Option.*$',
    r'\s+-\s+Growth\s+Plan\s+-\s+Bonus\s+Option.*$',
    r'\s*[-–]?\s*(Regular|Growth|IDCW|Dividend|Bonus)\s+Plan\s*[-–]?\s*(Growth|IDCW|Dividend|Bonus)(\s+Option|\s+Payout)?.*$',
    # Just the options
    r'\s*[-–]?\s*(Growth|IDCW|Dividend|Bonus)\s+Option.*$',
    # Space-separated variants
    r'\s+(Regular|Growth|IDCW|Dividend|Bonus)\s+Plan\s+(Growth|IDCW|Dividend|Bonus)(\s+Option|\s+Payout)?.*$',
    r'\s+(Regular|Growth|IDCW|Dividend|Bonus)\s+Plan\s+(Half\s+Yearly|Quarterly|Monthly|Weekly|Annual|Daily)\s+(Dividend|IDCW).*$',
    # Just the Plan types
    r'\s*[-–]\s*(Regular|Growth|IDCW|Dividend|Bonus)\s+Plan.*$',
    r'\s+(Regular|Growth|IDCW|Dividend|Bonus)\s+Plan.*$',
    # Floating keywords without "Plan" or "Option" (e.g. "- Regular - Growth")
    r'\s*[-–]\s*(Regular|Growth)\s*[-–]\s*(Growth|IDCW|Dividend|Bonus).*$',
    r'\s*[-–]\s*(Regular|Growth|IDCW|Dividend|Bonus).*$',
]

def extract_base_fund_name(scheme_name: str) -> str:
    """
    Strip the plan type and payout option suffix to get the base fund name.
    E.g. '360 ONE Focused Fund - Regular Plan - Growth' → '360 ONE Focused Fund'
         '360 ONE Dynamic Bond Fund Regular Plan Quarterly Dividend' → '360 ONE Dynamic Bond Fund'
    """
    if not isinstance(scheme_name, str):
        return str(scheme_name)

    cleaned = scheme_name.strip()
    for pattern in PAYOUT_SUFFIXES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def parse_aum_excel(filepath: str) -> pd.DataFrame:
    """
    Parse the AMFI AUM Excel file with its hierarchical structure.
    - Removes ANY row containing "Direct Plan"
    - Treats remaining rows as Regular (includes "Growth Plan" or unspecified)
    - Sums AUM across payout options (Growth + IDCW + Bonus + others)
    - Converts from Lakhs to Crores
    Returns a DataFrame with columns: scheme_code, scheme_name, aum_cr
    """
    # Read raw — the file has a single header row followed by mixed data
    raw = pd.read_excel(filepath, engine="openpyxl", header=None)

    # Col 0: AMFI Code (or AMC name / category name / "Total")
    # Col 1: Scheme NAV Name
    # Col 2: AUM excl FoF (Rs in Lakhs)
    # Col 3: FoF AUM (Rs in Lakhs)
    raw.columns = ["col0", "col1", "col2", "col3"]

    # Skip the first two metadata rows (quarter title + column headers)
    raw = raw.iloc[2:].reset_index(drop=True)

    # Identify data rows: those where col0 is a valid numeric AMFI code
    raw["col0_numeric"] = pd.to_numeric(raw["col0"], errors="coerce")
    data_rows = raw[raw["col0_numeric"].notna()].copy()

    # Filter: ONLY exclude "Direct Plan"
    data_rows["name_str"] = data_rows["col1"].astype(str)
    data_rows = data_rows[
        ~data_rows["name_str"].str.contains("Direct Plan", case=False, na=False)
    ].copy()

    print(f"  Non-Direct Plan rows found: {len(data_rows)}")

    # Parse AUM values (both columns contribute to total AUM)
    data_rows["aum_excl_fof"] = pd.to_numeric(data_rows["col2"], errors="coerce").fillna(0)
    data_rows["aum_fof"] = pd.to_numeric(data_rows["col3"], errors="coerce").fillna(0)
    data_rows["aum_lakhs"] = data_rows["aum_excl_fof"] + data_rows["aum_fof"]

    # Extract base fund name for grouping
    data_rows["base_fund_name"] = data_rows["name_str"].apply(extract_base_fund_name)

    # Keep one scheme_code per fund (use the first/smallest code)
    data_rows["scheme_code"] = data_rows["col0_numeric"].astype(int)

    # Group by base fund name: sum AUM, keep first scheme_code
    grouped = data_rows.groupby("base_fund_name", as_index=False).agg(
        scheme_code=("scheme_code", "first"),
        aum_lakhs=("aum_lakhs", "sum"),
        variant_count=("scheme_code", "count"),
    )

    # Convert Lakhs → Crores (÷ 100)
    grouped["aum_cr"] = (grouped["aum_lakhs"] / 100.0).round(2)

    result = pd.DataFrame({
        "scheme_code": grouped["scheme_code"],
        "scheme_name": grouped["base_fund_name"].str.strip(),
        "aum_cr": grouped["aum_cr"],
    })

    # Drop rows with zero or negative AUM
    result = result[result["aum_cr"] > 0].copy()

    multi_variant = grouped[grouped["variant_count"] > 1]
    print(f"  Funds with multiple payout options aggregated: {len(multi_variant)}")
    print(f"  Total unique Regular Plan funds: {len(result)}")

    return result.reset_index(drop=True)


def main(filepath: str = None):
    aum_file = filepath or IN_FILE

    if not os.path.exists(aum_file):
        print(f"[WARN] AUM file not found: {aum_file}")
        print("  Upload the latest average-aum.xlsx from AMFI website.")
        return

    print(f"Reading AUM file: {aum_file}")
    df = parse_aum_excel(aum_file)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\n[OK] scheme_aum.csv saved -> {OUT_PATH}")
    print(f"   Total Non-Direct Plan funds: {len(df)}")
    print(f"   AUM range: {df['aum_cr'].min():.2f} Cr – {df['aum_cr'].max():.2f} Cr")
    print(f"   Median AUM: {df['aum_cr'].median():.2f} Cr")
    print(f"   Total AUM: {df['aum_cr'].sum():,.0f} Cr")


if __name__ == "__main__":
    main()
