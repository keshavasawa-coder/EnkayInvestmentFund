"""
01_load_performance.py
Aggregates all fund performance Excel files from CRISIL Intelligence.
Assigns Category and Sub-Category to each fund.
Outputs: data/processed/scheme_performance.csv
"""
import os
import pandas as pd

# ── Base paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
PERF_FOLDERS = {
    "Equity":            "Fund_Performance_Equity",
    "Debt":              "Fund_Performance_Debt",
    "Hybrid":            "Fund_Performance_Hybrid",
    "Solution Oriented": "Fund_Performance_Solution_Oriented",
    "Other":             "Fund_Performance_Other",
}
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "scheme_performance.csv")

# ── Sub-category mapping: filename → SEBI sub-category label ────────────────
# Keys are partial filename patterns (matched by 'in'); order matters.
SUBCATEGORY_MAP = {
    # Equity
    "1322":             "Large Cap Fund",
    "1323 (1)":         "Flexi Cap Fund",
    "1323 (2)":         "Multi Cap Fund",
    "1323 (3)":         "Mid Cap Fund",
    "1323 (4)":         "Small Cap Fund",
    "1323 (5)":         "Value Fund",
    "1323 (6)":         "ELSS",
    "1323 (7)":         "Contra Fund",
    "1323.":            "Large & Mid Cap Fund",
    "1324 (1)":         "Focused Fund",
    "1324 (2)":         "Thematic / Sectoral Fund",
    "1324.":            "Dividend Yield Fund",
    # Debt
    "1331 (1)":         "Medium to Long Duration Fund",
    "1331.":            "Long Duration Fund",
    "1332 (1)":         "Medium Duration Fund",
    "1332 (2)":         "Money Market Fund",
    "1332 (3)":         "Low Duration Fund",
    "1332 (4)":         "Ultra Short Duration Fund",
    "1332 (5)":         "Liquid Fund",
    "1332 (6)":         "Overnight Fund",
    "1332 (7)":         "Dynamic Bond Fund",
    "1332.":            "Short Duration Fund",
    "1333 (1)":         "Credit Risk Fund",
    "1333 (2)":         "Banking & PSU Fund",
    "1333 (3)":         "Floater Fund",
    "1333 (4)":         "Gilt Fund",
    "1333 (5)":         "Gilt Fund with 10Y Constant Duration",
    "1333.":            "Corporate Bond Fund",
    # Hybrid
    "1336 (1)":         "Conservative Hybrid Fund",
    "1336 (2)":         "Equity Savings Fund",
    "1336 (3)":         "Arbitrage Fund",
    "1336 (4)":         "Multi Asset Allocation Fund",
    "1336 (5)":         "Dynamic Asset Allocation / BAF",
    "1336 (6)":         "Balanced Hybrid Fund",
    "1336.":            "Aggressive Hybrid Fund",
    # Solution Oriented
    "1336 (7)":         "Children's Fund",
    "1336 (8)":         "Retirement Fund",
    # Other
    "1340":             "Index / ETF Fund",
    "1341":             "Fund of Funds",
}


def get_subcategory(filename: str) -> str:
    """Match filename to sub-category using the map above."""
    for key, label in SUBCATEGORY_MAP.items():
        if key in filename:
            return label
    return "Unknown"


def load_folder(category: str, folder_name: str) -> pd.DataFrame:
    folder_path = os.path.join(BASE_DIR, folder_name)
    if not os.path.isdir(folder_path):
        print(f"  [WARN] Folder not found: {folder_path}")
        return pd.DataFrame()

    frames = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".xlsx"):
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            df = pd.read_excel(fpath, header=4, engine="openpyxl")
            # Drop completely empty rows
            df = df.dropna(subset=["Scheme Name"])
            df["category"] = category
            df["sub_category"] = get_subcategory(fname)
            df["source_file"] = fname
            frames.append(df)
            print(f"  [OK] {fname}: {len(df)} funds -> [{df['sub_category'].iloc[0]}]")
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names to snake_case."""
    rename_map = {
        "Scheme Name":                          "scheme_name",
        "Benchmark":                            "benchmark",
        "Riskometer Scheme":                    "riskometer",
        "Riskometer Benchmark":                 "riskometer_benchmark",
        "NAV Date":                             "nav_date",
        "NAV Regular":                          "nav_regular",
        "NAV Direct":                           "nav_direct",
        "Return 1 Year (%) Regular":            "return_1y_regular",
        "Return 1 Year (%) Direct":             "return_1y_direct",
        "Return 1 Year (%) Benchmark":          "return_1y_benchmark",
        "Information Ratio* 1 Year (Regular)":  "info_ratio_1y_regular",
        "Information Ratio* 1 Year (Direct)":   "info_ratio_1y_direct",
        "Return 3 Year (%) Regular":            "return_3y_regular",
        "Return 3 Year (%) Direct":             "return_3y_direct",
        "Return 3 Year (%) Benchmark":          "return_3y_benchmark",
        "Information Ratio* 3 Year (Regular)":  "info_ratio_3y_regular",
        "Information Ratio* 3 Year (Direct)":   "info_ratio_3y_direct",
        "Return 5 Year (%) Regular":            "return_5y_regular",
        "Return 5 Year (%) Direct":             "return_5y_direct",
        "Return 5 Year (%) Benchmark":          "return_5y_benchmark",
        "Information Ratio* 5 Year (Regular)":  "info_ratio_5y_regular",
        "Information Ratio* 5 Year (Direct)":   "info_ratio_5y_direct",
        "Return 10 Year (%) Regular":           "return_10y_regular",
        "Return 10 Year (%) Direct":            "return_10y_direct",
        "Return 10 Year (%) Benchmark":         "return_10y_benchmark",
        "Information Ratio* 10 Year (Regular)": "info_ratio_10y_regular",
        "Information Ratio* 10 Year (Direct)":  "info_ratio_10y_direct",
        "Return Since Launch Regular":          "return_since_launch_regular",
        "Return Since Launch Direct":           "return_since_launch_direct",
        "Return Since Launch  Benchmark":       "return_since_launch_benchmark",
        "Return Since Launch Direct Benchmark": "return_since_launch_direct_benchmark",
        "Daily AUM (Cr.)":                      "aum_cr",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def main():
    all_frames = []
    for category, folder_name in PERF_FOLDERS.items():
        print(f"\n--- {category} ---")
        df = load_folder(category, folder_name)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        print("[ERROR] No data loaded.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined = normalise_columns(combined)

    # Filter out invalid scheme names (non-fund entries)
    invalid_patterns = [
        "For detailed understanding regarding Information Ratio",
        "amfiindia.com/otherdata/fund-performance/information-ratio",
        "As mandated by SEBI",
        "closing AUM of the previous calendar month",
    ]
    mask = ~combined["scheme_name"].str.contains("|".join(invalid_patterns), case=False, na=False)
    removed_count = len(combined) - mask.sum()
    combined = combined[mask].reset_index(drop=True)
    
    if removed_count > 0:
        print(f"   Removed {removed_count} non-fund entries (metadata rows)")

    # Extract AMC name from scheme name (first 1-3 words heuristic; refined later in merge step)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"\n[OK] scheme_performance.csv saved -> {OUT_PATH}")
    print(f"   Total funds: {len(combined)}")
    print(f"   Categories: {combined['category'].value_counts().to_dict()}")
    print(f"   Sub-categories ({combined['sub_category'].nunique()}): {sorted(combined['sub_category'].unique().tolist())}")


if __name__ == "__main__":
    main()
