"""
04_merge_master.py
Merges scheme_performance.csv + scheme_aum.csv + scheme_brokerage.csv + tieup_flags.csv
into a single master_scheme_table.csv.
Uses scheme_code for AUM join and fuzzy name matching for brokerage.
Outputs:
  - data/processed/master_scheme_table.csv
  - data/processed/unmatched_schemes.txt  (performance funds with no brokerage match)
"""
import os
import pandas as pd
from rapidfuzz import process, fuzz

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..", "..")
PERF_FILE   = os.path.join(BASE_DIR, "data", "processed", "scheme_performance.csv")
AUM_FILE    = os.path.join(BASE_DIR, "data", "processed", "scheme_aum.csv")
BROK_FILE   = os.path.join(BASE_DIR, "data", "processed", "scheme_brokerage.csv")
TIEUP_FILE  = os.path.join(BASE_DIR, "data", "processed", "tieup_flags.csv")
OUT_MASTER  = os.path.join(BASE_DIR, "data", "processed", "master_scheme_table.csv")
OUT_UNMATCH = os.path.join(BASE_DIR, "data", "processed", "unmatched_schemes.txt")

FUZZY_THRESHOLD = 82  # Minimum similarity score (0-100) for a match


def extract_amc_from_scheme(scheme_name: str) -> str:
    """
    Best-effort extraction of AMC prefix from a scheme name.
    E.g. 'HDFC Mid Cap Opportunities Fund' → 'HDFC'
         'Aditya Birla Sun Life Large Cap Fund' → 'Aditya Birla Sun Life'
    """
    AMC_PREFIXES = [
        "Aditya Birla Sun Life", "Bandhan", "Bajaj Finserv", "Bank of India",
        "Baroda BNP Paribas", "Canara Robeco", "DSP", "Edelweiss",
        "Franklin Templeton", "Franklin India", "HDFC", "HSBC",
        "ICICI Prudential", "Invesco India", "Invesco", "JM Financial",
        "Kotak", "LIC", "Mahindra Manulife", "Mirae Asset",
        "Motilal Oswal", "Navi", "Nippon India", "NJ", "PGIM India",
        "PPFAS", "Quant", "SBI", "Shriram", "Tata", "Trust", "Union",
        "UTI", "WhiteOak Capital", "WhiteOak", "360 ONE",
        "Groww", "Helios", "Axis", "Alchemy", "Sundaram",
    ]
    if not isinstance(scheme_name, str):
        return ""
    for prefix in AMC_PREFIXES:
        if scheme_name.startswith(prefix):
            return prefix
    return scheme_name.split()[0] if scheme_name else ""


def fuzzy_match_brokerage(perf_names: list, brok_names: list,
                           brok_df: pd.DataFrame, threshold: int) -> dict:
    """
    Returns a dict: perf_scheme_name → brokerage row (Series).
    Uses rapidfuzz's token_sort_ratio for partial name matching.
    """
    mapping = {}

    for pname in perf_names:
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
    perf = pd.read_csv(PERF_FILE)
    print(f"  Performance funds: {len(perf)}")

    # AUM (optional — may not exist if user hasn't uploaded yet)
    if os.path.exists(AUM_FILE):
        aum = pd.read_csv(AUM_FILE)
        print(f"  AUM entries: {len(aum)}")
    else:
        aum = pd.DataFrame(columns=["scheme_code", "scheme_name", "aum_cr"])
        print("  [WARN] AUM file not found — AUM will be NaN")

    # Brokerage
    if os.path.exists(BROK_FILE):
        brok = pd.read_csv(BROK_FILE)
        print(f"  Brokerage entries: {len(brok)}")
    else:
        brok = pd.DataFrame(columns=["scheme_name", "trail_brokerage_incl_gst", "amc"])
        print("  [WARN] Brokerage file not found")

    # Tie-up
    if os.path.exists(TIEUP_FILE):
        tieup = pd.read_csv(TIEUP_FILE)
        print(f"  TieUp AMCs: {len(tieup)}")
    else:
        tieup = pd.DataFrame(columns=["amc_name", "amc_normalised", "tieup_category"])
        print("  [WARN] TieUp file not found")

    # ── Step 1: Join AUM on scheme_code (or fuzzy name match) ────────────
    print("\nJoining AUM data...")

    # Drop any pre-existing aum_cr from CRISIL performance data
    if "aum_cr" in perf.columns:
        perf = perf.drop(columns=["aum_cr"])
        print("  Dropped old 'aum_cr' from performance data")

    if not aum.empty:
        if "scheme_code" in perf.columns:
            # Primary path: join on scheme_code
            perf = perf.merge(
                aum[["scheme_code", "aum_cr"]],
                on="scheme_code", how="left"
            )
        else:
            # Fallback: fuzzy-match AUM by scheme name
            print("  No scheme_code in performance data — using fuzzy name matching for AUM...")
            aum_names = aum["scheme_name"].tolist()
            aum_lookup = dict(zip(aum["scheme_name"], aum["aum_cr"]))

            aum_values = []
            for pname in perf["scheme_name"].tolist():
                result = process.extractOne(
                    pname, aum_names,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=80,
                )
                if result:
                    aum_values.append(aum_lookup.get(result[0], float("nan")))
                else:
                    aum_values.append(float("nan"))
            perf["aum_cr"] = aum_values

        aum_matched = perf["aum_cr"].notna().sum()
        print(f"  AUM matched: {aum_matched} / {len(perf)} "
              f"({aum_matched / len(perf) * 100:.1f}%)")
    else:
        perf["aum_cr"] = float("nan")

    # ── Step 2: Fuzzy-match brokerage ──────────────────────────────────────
    print("\nFuzzy-matching brokerage (this may take ~30 seconds)...")
    if not brok.empty:
        perf_names = perf["scheme_name"].tolist()
        brok_names = brok["scheme_name"].tolist()

        match_map = fuzzy_match_brokerage(perf_names, brok_names, brok, FUZZY_THRESHOLD)
        matched_cnt = len(match_map)
        print(f"  Matched: {matched_cnt} / {len(perf_names)} "
              f"({matched_cnt / len(perf_names) * 100:.1f}%)")

        # Build brokerage columns aligned to performance df
        brok_rows = []
        for pname in perf_names:
            if pname in match_map:
                row = match_map[pname]
                brok_rows.append({
                    "brok_scheme_name": row["scheme_name"],
                    "trail_brokerage_incl_gst": row["trail_brokerage_incl_gst"],
                    "amc_from_brokerage": row.get("amc", ""),
                })
            else:
                brok_rows.append({
                    "brok_scheme_name": None,
                    "trail_brokerage_incl_gst": None,
                    "amc_from_brokerage": None,
                })

        brok_aligned = pd.DataFrame(brok_rows)
        perf = pd.concat([perf.reset_index(drop=True),
                          brok_aligned.reset_index(drop=True)], axis=1)
    else:
        perf["brok_scheme_name"] = None
        perf["trail_brokerage_incl_gst"] = None
        perf["amc_from_brokerage"] = None

    # ── Step 3: Derive AMC name ────────────────────────────────────────────
    perf["amc_inferred"] = perf["scheme_name"].apply(extract_amc_from_scheme)
    perf["amc"] = perf["amc_from_brokerage"].fillna(perf["amc_inferred"])

    # ── Step 4: Join TieUp flag via AMC name ───────────────────────────────
    print("\nJoining TieUp flags...")
    if not tieup.empty:
        tieup_dict = dict(zip(tieup["amc_normalised"].str.lower(),
                               tieup["tieup_category"]))
        tieup_full = dict(zip(tieup["amc_name"].str.lower(),
                               tieup["tieup_category"]))

        def get_tieup(amc_name):
            if not isinstance(amc_name, str) or not amc_name.strip():
                return None
            amc_lower = amc_name.lower().strip()
            # 1. Direct full-name match
            if amc_lower in tieup_full:
                return tieup_full[amc_lower]
            # 2. Substring match
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
    else:
        perf["tieup_category"] = None

    # ── Step 5: Save master table ──────────────────────────────────────────
    drop_cols = ["amc_from_brokerage", "amc_inferred"]
    perf = perf.drop(columns=[c for c in drop_cols if c in perf.columns])

    os.makedirs(os.path.dirname(OUT_MASTER), exist_ok=True)
    perf.to_csv(OUT_MASTER, index=False)
    print(f"\n[OK] master_scheme_table.csv saved -> {OUT_MASTER}")
    print(f"   Total rows: {len(perf)}")
    print(f"   Columns: {list(perf.columns)}")

    # ── Step 6: Log unmatched schemes ─────────────────────────────────────
    unmatched = perf[perf["brok_scheme_name"].isna()]["scheme_name"].tolist()
    with open(OUT_UNMATCH, "w", encoding="utf-8") as f:
        f.write(f"Unmatched schemes (no brokerage data found): {len(unmatched)}\n\n")
        for s in unmatched:
            f.write(f"  - {s}\n")
    print(f"\n[!] Unmatched schemes logged -> {OUT_UNMATCH}  ({len(unmatched)} funds)")


if __name__ == "__main__":
    main()
