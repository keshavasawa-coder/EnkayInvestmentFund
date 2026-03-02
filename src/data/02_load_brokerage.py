"""
02_load_brokerage.py
Loads Enkay's trail brokerage rates from NJ Partner Desk data.
No filtering on transaction period or investment amount slabs.
Where a scheme has multiple rows, the MAX brokerage rate (Incl. GST) is kept.
Outputs: data/processed/scheme_brokerage.csv
"""
import os
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), "..", "..")
IN_FILE   = os.path.join(BASE_DIR, "Brokerage_Rates_Enkay.xlsx")
OUT_PATH  = os.path.join(BASE_DIR, "data", "processed", "scheme_brokerage.csv")


def normalise_amc_name(name: str) -> str:
    """Strip common legal suffixes for easier matching."""
    if not isinstance(name, str):
        return ""
    replacements = [
        "Asset Management Company Limited",
        "Asset Management Company Ltd.",
        "Asset Management Company Ltd",
        "Asset Management (India) Private Limited",
        "Asset Management (India) Pvt. Ltd.",
        "Asset Management Private Limited",
        "Asset Management Limited",
        "Asset Managers Private Limited",
        "Investment Managers (India) Pvt. Ltd.",
        "Investment Management Private Limited",
        "Funds Management Limited",
        "Funds Management Ltd",
        "AMC Limited",
        "AMC Ltd.",
        "AMC Ltd",
        "Private Limited",
        "Pvt. Ltd.",
        "Limited",
        "Ltd.",
        "Ltd",
    ]
    result = name.strip()
    for r in replacements:
        result = result.replace(r, "").strip()
    # Remove trailing punctuation
    result = result.strip(" ,-.")
    return result


def main():
    print(f"Reading: {IN_FILE}")
    df = pd.read_excel(IN_FILE, header=0, engine="openpyxl")

    # Keep only the columns we care about
    keep_cols = [
        "AMC",
        "Scheme Type",
        "Scheme Sub Type",
        "Scheme Name",
        "Trail Brokerage Rate (Inclusive of GST)",
    ]
    df = df[keep_cols].copy()
    df.columns = ["amc", "scheme_type", "scheme_sub_type", "scheme_name", "trail_brokerage_incl_gst"]

    # Drop rows where scheme name or brokerage is null
    df = df.dropna(subset=["scheme_name", "trail_brokerage_incl_gst"])
    df["scheme_name"] = df["scheme_name"].str.strip()
    df["amc"]         = df["amc"].str.strip()

    print(f"  Raw rows after dropping nulls: {len(df)}")

    # For schemes with multiple rows, keep the MAX brokerage rate
    df = (
        df.groupby("scheme_name", as_index=False)
          .agg(
              amc                    = ("amc",                    "first"),
              scheme_type            = ("scheme_type",            "first"),
              scheme_sub_type        = ("scheme_sub_type",        "first"),
              trail_brokerage_incl_gst = ("trail_brokerage_incl_gst", "max"),
          )
    )

    # Add normalised AMC name for join
    df["amc_normalised"] = df["amc"].apply(normalise_amc_name)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\n✅ scheme_brokerage.csv saved → {OUT_PATH}")
    print(f"   Unique schemes: {len(df)}")
    print(f"   AMCs covered:   {df['amc'].nunique()}")
    print(f"   Brokerage range (Incl. GST): "
          f"{df['trail_brokerage_incl_gst'].min():.2f}% – "
          f"{df['trail_brokerage_incl_gst'].max():.2f}%")
    print(f"\n   By Scheme Type:\n{df['scheme_type'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
