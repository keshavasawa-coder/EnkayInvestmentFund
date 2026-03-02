"""
03_load_tieup.py
Parses TieUp_AMCs_List.xlsx and creates a mapping of AMC → TieUp category.
Only mutual fund tie-ups are used (A Category AMC and B Category AMC).
PMS / AIF entries are excluded from this module.
Outputs: data/processed/tieup_flags.csv
"""
import os
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
IN_FILE  = os.path.join(BASE_DIR, "TieUp_AMCs_List.xlsx")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "tieup_flags.csv")


def normalise_amc_name(name: str) -> str:
    """Strip common legal suffixes for easier fuzzy matching later."""
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
    result = result.strip(" ,-.")
    return result


def main():
    print(f"Reading: {IN_FILE}")
    df = pd.read_excel(IN_FILE, header=0, engine="openpyxl")
    df.columns = ["sr_no", "amc_name", "tieup_type"]

    df = df.dropna(subset=["amc_name", "tieup_type"])
    df["amc_name"]   = df["amc_name"].str.strip().str.replace(r"\n\s*", " ", regex=True)
    df["tieup_type"] = df["tieup_type"].str.strip()

    # Keep only mutual fund tie-ups
    mf_types = {"A Category AMC", "B Category AMC"}
    df_mf = df[df["tieup_type"].isin(mf_types)].copy()

    # Deduplicate: if an AMC appears in both A and B (shouldn't happen, but safe)
    df_mf = df_mf.drop_duplicates(subset="amc_name", keep="first")

    df_mf["amc_normalised"] = df_mf["amc_name"].apply(normalise_amc_name)
    df_mf["tieup_category"] = df_mf["tieup_type"].map({
        "A Category AMC": "A",
        "B Category AMC": "B",
    })

    result = df_mf[["amc_name", "amc_normalised", "tieup_category"]].reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    result.to_csv(OUT_PATH, index=False)

    print(f"\n✅ tieup_flags.csv saved → {OUT_PATH}")
    print(f"   A-Category AMCs: {(result['tieup_category'] == 'A').sum()}")
    print(f"   B-Category AMCs: {(result['tieup_category'] == 'B').sum()}")
    print(f"\n   A-Category AMCs:\n{result[result['tieup_category']=='A']['amc_name'].to_string(index=False)}")
    print(f"\n   B-Category AMCs:\n{result[result['tieup_category']=='B']['amc_name'].to_string(index=False)}")


if __name__ == "__main__":
    main()
