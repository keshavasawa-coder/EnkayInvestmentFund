"""
04_merge_master.py
Merges scheme_performance.csv + scheme_brokerage.csv + tieup_flags.csv
into a single master_scheme_table.csv using fuzzy name matching.
Outputs:
  - data/processed/master_scheme_table.csv
  - data/processed/unmatched_schemes.txt  (performance funds with no brokerage match)
"""
import os
import pandas as pd
from rapidfuzz import process, fuzz

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..", "..")
PERF_FILE  = os.path.join(BASE_DIR, "data", "processed", "scheme_performance.csv")
BROK_FILE  = os.path.join(BASE_DIR, "data", "processed", "scheme_brokerage.csv")
TIEUP_FILE = os.path.join(BASE_DIR, "data", "processed", "tieup_flags.csv")
OUT_MASTER = os.path.join(BASE_DIR, "data", "processed", "master_scheme_table.csv")
OUT_UNMATCH= os.path.join(BASE_DIR, "data", "processed", "unmatched_schemes.txt")

FUZZY_THRESHOLD = 82  # Minimum similarity score (0-100) for a match


def extract_amc_from_scheme(scheme_name: str) -> str:
    """
    Best-effort extraction of AMC prefix from a scheme name.
    E.g. 'HDFC Mid Cap Opportunities Fund' → 'HDFC'
         'Aditya Birla Sun Life Large Cap Fund' → 'Aditya Birla Sun Life'
    """
    # Known multi-word AMC prefixes (longest first to avoid partial match)
    AMC_PREFIXES = [
        "Aditya Birla Sun Life", "Bandhan", "Bajaj Finserv", "Bank of India",
        "Baroda BNP Paribas", "Canara Robeco", "DSP", "Edelweiss",
        "Franklin Templeton", "Franklin India", "HDFC", "HSBC",
        "ICICI Prudential", "Invesco India", "Invesco", "JM Financial",
        "Kotak", "LIC", "Mahindra Manulife", "Mirae Asset",
        "Motilal Oswal", "Navi", "Nippon India", "NJ", "PGIM India",
        "PPFAS", "Quant", "SBI", "Shriram", "Tata", "Trust", "Union",
        "UTI", "WhiteOak Capital", "WhiteOak", "360 ONE",
        "Groww", "Helios", "Axis", "Alchemy",
    ]
    if not isinstance(scheme_name, str):
        return ""
    for prefix in AMC_PREFIXES:
        if scheme_name.startswith(prefix):
            return prefix
    # Fallback: first word
    return scheme_name.split()[0] if scheme_name else ""


def fuzzy_match_brokerage(perf_names: list, brok_names: list,
                           brok_df: pd.DataFrame, threshold: int,
                           exclude_sub_categories: list = None) -> dict:
    """
    Returns a dict: perf_scheme_name → brokerage row (Series).
    Uses rapidfuzz's token_sort_ratio for partial name matching.
    
    Parameters:
    - exclude_sub_categories: list of sub-category names that should NOT be fuzzy matched
      (e.g., ["Contra Fund"]). These will not be matched and will return None.
    """
    # Pre-build lookup
    brok_index = {name: i for i, name in enumerate(brok_names)}
    mapping = {}
    
    # Build set of schemes to exclude from fuzzy matching
    exclude_schemes = set()
    if exclude_sub_categories:
        perf_df_temp = pd.read_csv(PERF_FILE)
        for sub_cat in exclude_sub_categories:
            excluded = perf_df_temp[perf_df_temp["sub_category"] == sub_cat]["scheme_name"].tolist()
            exclude_schemes.update(excluded)
        print(f"  Excluding {len(exclude_schemes)} schemes from fuzzy matching (sub-categories: {exclude_sub_categories})")

    for pname in perf_names:
        # Skip fuzzy matching for excluded sub-categories
        if pname in exclude_schemes:
            continue
        
        result = process.extractOne(
            pname, brok_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result:
            matched_name, score, _ = result
            mapping[pname] = brok_df[brok_df["scheme_name"] == matched_name].iloc[0]
    return mapping


def main():
    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading processed CSVs...")
    perf  = pd.read_csv(PERF_FILE)
    brok  = pd.read_csv(BROK_FILE)
    tieup = pd.read_csv(TIEUP_FILE)

    print(f"  Performance funds: {len(perf)}")
    print(f"  Brokerage entries: {len(brok)}")
    print(f"  TieUp AMCs:        {len(tieup)}")

    # ── Step 1: Fuzzy-match scheme names (performance ↔ brokerage) ─────────
    print("\nFuzzy-matching scheme names (this may take ~30 seconds)...")
    perf_names = perf["scheme_name"].tolist()
    brok_names = brok["scheme_name"].tolist()

    # Sub-categories to exclude from fuzzy matching (no brokerage data available)
    EXCLUDE_FROM_FUZZY = ["Contra Fund"]
    
    match_map   = fuzzy_match_brokerage(perf_names, brok_names, brok, FUZZY_THRESHOLD,
                                        exclude_sub_categories=EXCLUDE_FROM_FUZZY)
    matched_cnt = len(match_map)
    print(f"  Matched: {matched_cnt} / {len(perf_names)} "
          f"({matched_cnt / len(perf_names) * 100:.1f}%)")

    # Build brokerage columns aligned to performance df
    brok_rows = []
    for pname in perf_names:
        if pname in match_map:
            row = match_map[pname]
            brok_rows.append({
                "brok_scheme_name":      row["scheme_name"],
                "trail_brokerage_incl_gst": row["trail_brokerage_incl_gst"],
                "amc_from_brokerage":    row["amc"],
                "amc_normalised_brok":   row.get("amc_normalised", ""),
            })
        else:
            brok_rows.append({
                "brok_scheme_name":      None,
                "trail_brokerage_incl_gst": None,
                "amc_from_brokerage":    None,
                "amc_normalised_brok":   None,
            })

    brok_aligned = pd.DataFrame(brok_rows)
    perf = pd.concat([perf.reset_index(drop=True), brok_aligned.reset_index(drop=True)], axis=1)

    # ── Step 2: Derive AMC name from scheme name ───────────────────────────
    perf["amc_inferred"] = perf["scheme_name"].apply(extract_amc_from_scheme)

    # Use brokerage AMC if available, otherwise use inferred AMC
    perf["amc"] = perf["amc_from_brokerage"].fillna(perf["amc_inferred"])

    # ── Step 3: Join TieUp flag via AMC name (fuzzy on normalised names) ───
    print("\nJoining TieUp flags...")
    tieup_dict = dict(zip(tieup["amc_normalised"].str.lower(),
                           tieup["tieup_category"]))
    tieup_full  = dict(zip(tieup["amc_name"].str.lower(),
                            tieup["tieup_category"]))

    def get_tieup(amc_name):
        if not isinstance(amc_name, str) or not amc_name.strip():
            return None
        amc_lower = amc_name.lower().strip()
        # 1. Direct full-name match (against full AMC names)
        if amc_lower in tieup_full:
            return tieup_full[amc_lower]
        # 2. Substring match: check if any normalised short name (e.g. 'Mirae Asset')
        #    is contained within the input amc_name (handles long legal names)
        for norm_name, cat in tieup_dict.items():
            if norm_name in amc_lower or amc_lower in norm_name:
                return cat
        # 3. Fuzzy match as fallback
        result = process.extractOne(
            amc_lower, list(tieup_dict.keys()),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=78,
        )
        if result:
            return tieup_dict[result[0]]
        return None

    perf["tieup_category"] = perf["amc"].apply(get_tieup)
    tieup_counts = perf["tieup_category"].value_counts(dropna=False)
    print(f"  TieUp distribution: {tieup_counts.to_dict()}")

    # ── Step 4: Save master table ──────────────────────────────────────────
    # Drop intermediate helper columns
    drop_cols = ["amc_from_brokerage", "amc_normalised_brok", "amc_inferred"]
    perf = perf.drop(columns=[c for c in drop_cols if c in perf.columns])

    os.makedirs(os.path.dirname(OUT_MASTER), exist_ok=True)
    perf.to_csv(OUT_MASTER, index=False)
    print(f"\n[OK] master_scheme_table.csv saved -> {OUT_MASTER}")
    print(f"   Total rows: {len(perf)}")
    print(f"   Columns: {list(perf.columns)}")

    # ── Step 5: Log unmatched schemes ─────────────────────────────────────
    unmatched = perf[perf["brok_scheme_name"].isna()]["scheme_name"].tolist()
    with open(OUT_UNMATCH, "w", encoding="utf-8") as f:
        f.write(f"Unmatched schemes (no brokerage data found): {len(unmatched)}\n\n")
        for s in unmatched:
            f.write(f"  - {s}\n")
    print(f"\n[!] Unmatched schemes logged -> {OUT_UNMATCH}  ({len(unmatched)} funds)")


if __name__ == "__main__":
    main()
