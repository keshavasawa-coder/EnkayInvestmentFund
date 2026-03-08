"""
portfolio_builder.py
Builds curated mutual fund portfolio baskets for Enkay Investments.
Modeled on NJ's Recommended MF Portfolio approach — named baskets with
specific asset-class allocations and sub-category picks.
Each basket auto-selects the top-ranked fund per sub-category from the
active risk profile, with AMC concentration cap enforcement.
"""
import os
import pandas as pd

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..", "..")
RANKED_FILE = os.path.join(BASE_DIR, "data", "processed", "ranked_funds.csv")

# ── Portfolio Baskets ────────────────────────────────────────────────────────
# Each basket: name, equity%, debt%, hybrid%, other%, risk_level,
#   picks = [(sub_category, allocation_pct), ...]
#   allocation_pct values within a basket should sum to 100.

PORTFOLIO_BASKETS = [
    # ── Conservative ─────────────────────────────────────────────────────
    {
        "name": "Conservative - Equity 30",
        "equity_pct": 30, "debt_pct": 70, "hybrid_pct": 0, "other_pct": 0,
        "risk_level": "Moderately High",
        "picks": [
            # Equity 30%
            ("Large Cap Fund",              15),
            ("Flexi Cap Fund",              15),
            # Debt 70%
            ("Liquid Fund",                 10),
            ("Ultra Short Duration Fund",   10),
            ("Short Duration Fund",         10),
            ("Corporate Bond Fund",         10),
            ("Banking & PSU Fund",          10),
            ("Money Market Fund",           10),
            ("Dynamic Bond Fund",           10),
        ],
    },
    # ── Balanced ─────────────────────────────────────────────────────────
    {
        "name": "Balanced - Equity 40",
        "equity_pct": 40, "debt_pct": 60, "hybrid_pct": 0, "other_pct": 0,
        "risk_level": "High",
        "picks": [
            # Equity 40%
            ("Large Cap Fund",               10),
            ("Flexi Cap Fund",               10),
            ("Large & Mid Cap Fund",         10),
            ("ELSS",                         10),
            # Debt 60%
            ("Liquid Fund",                  10),
            ("Ultra Short Duration Fund",    10),
            ("Short Duration Fund",          10),
            ("Corporate Bond Fund",          10),
            ("Banking & PSU Fund",           10),
            ("Money Market Fund",            10),
        ],
    },
    {
        "name": "Balanced - Equity 50",
        "equity_pct": 50, "debt_pct": 50, "hybrid_pct": 0, "other_pct": 0,
        "risk_level": "Very High",
        "picks": [
            # Equity 50%
            ("Large Cap Fund",              10),
            ("Flexi Cap Fund",              10),
            ("Multi Cap Fund",              10),
            ("Mid Cap Fund",                10),
            ("ELSS",                        10),
            # Debt 50%
            ("Liquid Fund",                 10),
            ("Short Duration Fund",         10),
            ("Corporate Bond Fund",         10),
            ("Banking & PSU Fund",          10),
            ("Money Market Fund",           10),
        ],
    },
    {
        "name": "Balanced - Equity 60",
        "equity_pct": 60, "debt_pct": 40, "hybrid_pct": 0, "other_pct": 0,
        "risk_level": "Very High",
        "picks": [
            # Equity 60%
            ("Large Cap Fund",               10),
            ("Flexi Cap Fund",               10),
            ("Multi Cap Fund",               10),
            ("Mid Cap Fund",                 10),
            ("ELSS",                         10),
            ("Large & Mid Cap Fund",         10),
            # Debt 40%
            ("Short Duration Fund",          10),
            ("Corporate Bond Fund",          10),
            ("Banking & PSU Fund",           10),
            ("Liquid Fund",                  10),
        ],
    },
    # ── Growth ───────────────────────────────────────────────────────────
    {
        "name": "Growth - Equity 70",
        "equity_pct": 70, "debt_pct": 30, "hybrid_pct": 0, "other_pct": 0,
        "risk_level": "Very High",
        "picks": [
            # Equity 70%
            ("Large Cap Fund",              10),
            ("Mid Cap Fund",                10),
            ("Flexi Cap Fund",              10),
            ("Multi Cap Fund",              10),
            ("ELSS",                        10),
            ("Small Cap Fund",              10),
            ("Large & Mid Cap Fund",        10),
            # Debt 30%
            ("Short Duration Fund",         10),
            ("Corporate Bond Fund",         10),
            ("Banking & PSU Fund",          10),
        ],
    },
    {
        "name": "Growth - Equity 80",
        "equity_pct": 80, "debt_pct": 20, "hybrid_pct": 0, "other_pct": 0,
        "risk_level": "Very High",
        "picks": [
            # Equity 80%
            ("Large Cap Fund",               10),
            ("Mid Cap Fund",                 10),
            ("Small Cap Fund",               10),
            ("Flexi Cap Fund",               10),
            ("Multi Cap Fund",               10),
            ("ELSS",                         10),
            ("Focused Fund",                 10),
            ("Large & Mid Cap Fund",         10),
            # Debt 20%
            ("Short Duration Fund",          10),
            ("Corporate Bond Fund",          10),
        ],
    },
    {
        "name": "Growth - Equity 90",
        "equity_pct": 90, "debt_pct": 10, "hybrid_pct": 0, "other_pct": 0,
        "risk_level": "Very High",
        "picks": [
            # Equity 90%
            ("Large Cap Fund",                10),
            ("Mid Cap Fund",                  10),
            ("Small Cap Fund",                10),
            ("Flexi Cap Fund",                10),
            ("Multi Cap Fund",                10),
            ("ELSS",                          10),
            ("Focused Fund",                  10),
            ("Value Fund",                    10),
            ("Thematic / Sectoral Fund",      10),
            # Debt 10%
            ("Short Duration Fund",           10),
        ],
    },
    {
        "name": "Growth - Equity 100",
        "equity_pct": 100, "debt_pct": 0, "hybrid_pct": 0, "other_pct": 0,
        "risk_level": "Very High",
        "picks": [
            ("Large Cap Fund",               10),
            ("Mid Cap Fund",                 10),
            ("Small Cap Fund",               10),
            ("Flexi Cap Fund",               10),
            ("Multi Cap Fund",               10),
            ("ELSS",                         10),
            ("Focused Fund",                 10),
            ("Value Fund",                   10),
            ("Thematic / Sectoral Fund",     10),
            ("Contra Fund",                  10),
        ],
    },
    # ── Dynamic ──────────────────────────────────────────────────────────
    {
        "name": "Dynamic - Aggressive",
        "equity_pct": 65, "debt_pct": 35, "hybrid_pct": 0, "other_pct": 0,
        "risk_level": "Very High",
        "picks": [
            # Equity ~65%
            ("Large Cap Fund",               10),
            ("Mid Cap Fund",                 10),
            ("Flexi Cap Fund",               10),
            ("Multi Cap Fund",               10),
            ("Small Cap Fund",                5),
            ("ELSS",                         10),
            ("Large & Mid Cap Fund",         10),
            # Debt ~35%
            ("Short Duration Fund",          10),
            ("Corporate Bond Fund",          10),
            ("Banking & PSU Fund",            5),
            ("Dynamic Bond Fund",            10),
        ],
    },
    # ── SIP Portfolios ───────────────────────────────────────────────────
    {
        "name": "SIP - Diversified",
        "equity_pct": 100, "debt_pct": 0, "hybrid_pct": 0, "other_pct": 0,
        "risk_level": "Very High",
        "picks": [
            ("Large Cap Fund",              20),
            ("Flexi Cap Fund",              20),
            ("Multi Cap Fund",              20),
            ("Mid Cap Fund",                20),
            ("ELSS",                        20),
        ],
    },
    {
        "name": "SIP - Aggressive",
        "equity_pct": 100, "debt_pct": 0, "hybrid_pct": 0, "other_pct": 0,
        "risk_level": "Very High",
        "picks": [
            ("Mid Cap Fund",                20),
            ("Small Cap Fund",              20),
            ("Flexi Cap Fund",              20),
            ("Multi Cap Fund",              20),
            ("ELSS",                        20),
        ],
    },
    {
        "name": "SIP - Focused",
        "equity_pct": 100, "debt_pct": 0, "hybrid_pct": 0, "other_pct": 0,
        "risk_level": "Very High",
        "picks": [
            ("Focused Fund",                34),
            ("Large Cap Fund",              33),
            ("Flexi Cap Fund",              33),
        ],
    },
    # ── Tax Saving ───────────────────────────────────────────────────────
    {
        "name": "Tax Saving",
        "equity_pct": 100, "debt_pct": 0, "hybrid_pct": 0, "other_pct": 0,
        "risk_level": "Very High",
        "picks": [
            ("ELSS",                        100),
        ],
    },
    # ── Liquid ───────────────────────────────────────────────────────────
    {
        "name": "Liquid",
        "equity_pct": 0, "debt_pct": 100, "hybrid_pct": 0, "other_pct": 0,
        "risk_level": "Low to Moderate",
        "picks": [
            ("Liquid Fund",                  40),
            ("Overnight Fund",               30),
            ("Money Market Fund",            30),
        ],
    },
    # ── Hybrid Portfolios ────────────────────────────────────────────────
    {
        "name": "Hybrid - Aggressive",
        "equity_pct": 0, "debt_pct": 0, "hybrid_pct": 100, "other_pct": 0,
        "risk_level": "Very High",
        "picks": [
            ("Aggressive Hybrid Fund",      100),
        ],
    },
    {
        "name": "Hybrid - Balanced Advantage",
        "equity_pct": 0, "debt_pct": 0, "hybrid_pct": 100, "other_pct": 0,
        "risk_level": "Very High",
        "picks": [
            ("Dynamic Asset Allocation / BAF", 100),
        ],
    },
    {
        "name": "Hybrid - Conservative",
        "equity_pct": 0, "debt_pct": 0, "hybrid_pct": 100, "other_pct": 0,
        "risk_level": "Moderately High",
        "picks": [
            ("Conservative Hybrid Fund",    100),
        ],
    },
    # ── Multi-Asset ──────────────────────────────────────────────────────
    {
        "name": "Multi-Asset",
        "equity_pct": 0, "debt_pct": 0, "hybrid_pct": 100, "other_pct": 0,
        "risk_level": "Very High",
        "picks": [
            ("Multi Asset Allocation Fund", 100),
        ],
    },
]

# Quick lookup by name
BASKET_NAMES = [b["name"] for b in PORTFOLIO_BASKETS]
_BASKET_MAP  = {b["name"]: b for b in PORTFOLIO_BASKETS}


def get_basket(name: str) -> dict:
    """Return a basket definition by name. Raises ValueError if not found."""
    b = _BASKET_MAP.get(name)
    if b is None:
        raise ValueError(f"Unknown basket: {name}. Available: {BASKET_NAMES}")
    return b


def build_portfolio(
    ranked_df: pd.DataFrame = None,
    basket_name: str = "Balanced - Equity 50",
    risk_profile: str = "moderate",
    amc_cap_pct: float = 0.30,
) -> dict:
    """
    Build a recommended portfolio for the given basket.

    Parameters
    ----------
    ranked_df : DataFrame
        The full ranked_funds data (all 3 profiles).  If None, loads from disk.
    basket_name : str
        Name of the basket to build (must be in BASKET_NAMES).
    risk_profile : str
        Which risk profile ranking to use ('conservative', 'moderate', 'aggressive').
    amc_cap_pct : float
        Maximum share of portfolio slots a single AMC can occupy (0.0–1.0).

    Returns
    -------
    dict with:
      - 'portfolio' : DataFrame of selected funds
      - 'basket'    : the basket definition dict
      - 'swaps'     : list of swap descriptions (if AMC cap triggered)
      - 'missing'   : list of sub-categories with no data
    """
    if ranked_df is None:
        ranked_df = pd.read_csv(RANKED_FILE)

    basket = get_basket(basket_name)
    profile_df = ranked_df[ranked_df["risk_profile"] == risk_profile].copy()

    selected = []
    missing = []

    for sub_cat, alloc_pct in basket["picks"]:
        candidates = profile_df[profile_df["sub_category"] == sub_cat].sort_values("rank")
        if candidates.empty:
            missing.append(sub_cat)
            continue

        top_fund = candidates.iloc[0].copy()
        top_fund["allocation_pct"] = alloc_pct
        top_fund["template_sub_cat"] = sub_cat
        selected.append(top_fund)

    if not selected:
        return {
            "portfolio": pd.DataFrame(),
            "basket": basket,
            "swaps": [],
            "missing": missing,
        }

    portfolio = pd.DataFrame(selected)
    swaps = []

    # ── AMC concentration cap ────────────────────────────────────────────
    total_slots = len(portfolio)
    if total_slots > 1:                       # skip cap for single-scheme baskets
        max_slots = max(1, int(total_slots * amc_cap_pct))

        for _ in range(20):                   # safety limit
            amc_counts = portfolio["amc"].value_counts()
            over_limit = amc_counts[amc_counts > max_slots]
            if over_limit.empty:
                break

            swapped = False
            for amc_name, count in over_limit.items():
                amc_funds = portfolio[portfolio["amc"] == amc_name].sort_values("composite_score")
                worst = amc_funds.iloc[0]
                worst_idx = worst.name

                sub_cat = worst["sub_category"]
                over_amc_set = set(amc_counts[amc_counts >= max_slots].index)
                alternatives = profile_df[
                    (profile_df["sub_category"] == sub_cat) &
                    (~profile_df["amc"].isin(over_amc_set)) &
                    (~profile_df["scheme_name"].isin(portfolio["scheme_name"]))
                ].sort_values("rank")

                if alternatives.empty:
                    continue

                replacement = alternatives.iloc[0].copy()
                replacement["allocation_pct"] = worst["allocation_pct"]
                replacement["template_sub_cat"] = worst.get("template_sub_cat", sub_cat)

                swap_desc = (
                    f"Replaced '{worst['scheme_name']}' (AMC: {amc_name}, Score: {worst['composite_score']:.1f}) "
                    f"with '{replacement['scheme_name']}' (AMC: {replacement['amc']}, Score: {replacement['composite_score']:.1f}) "
                    f"in {sub_cat} — AMC cap of {amc_cap_pct:.0%} triggered"
                )
                swaps.append(swap_desc)
                portfolio.loc[worst_idx] = replacement
                swapped = True
                break  # re-check counts after each swap

            if not swapped:
                break

    # Clean up columns
    keep_cols = [
        "scheme_name", "amc", "sub_category", "category",
        "allocation_pct", "composite_score", "rank",
        "trail_brokerage_incl_gst", "tieup_category",
        "return_1y_regular", "return_3y_regular", "return_5y_regular",
        "aum_cr", "template_sub_cat",
    ]
    keep_cols = [c for c in keep_cols if c in portfolio.columns]
    portfolio = portfolio[keep_cols].reset_index(drop=True)

    return {
        "portfolio": portfolio,
        "basket": basket,
        "swaps": swaps,
        "missing": missing,
    }


def get_portfolio_stats(portfolio_df: pd.DataFrame) -> dict:
    """Compute summary statistics for a portfolio DataFrame."""
    if portfolio_df.empty:
        return {}

    total_alloc = portfolio_df["allocation_pct"].sum()
    weights = portfolio_df["allocation_pct"] / total_alloc if total_alloc > 0 else 0

    brok   = portfolio_df["trail_brokerage_incl_gst"]
    ret_1y = portfolio_df["return_1y_regular"]

    return {
        "num_schemes":        len(portfolio_df),
        "num_amcs":           portfolio_df["amc"].nunique(),
        "weighted_avg_brok":  (brok.fillna(0) * weights).sum(),
        "weighted_avg_ret1y": (ret_1y.fillna(0) * weights).sum(),
        "avg_score":          portfolio_df["composite_score"].mean(),
        "tieup_a_count":      (portfolio_df["tieup_category"] == "A").sum(),
        "tieup_b_count":      (portfolio_df["tieup_category"] == "B").sum(),
        "no_tieup_count":     portfolio_df["tieup_category"].isna().sum() +
                              (portfolio_df["tieup_category"] == "None").sum(),
        "amc_distribution":   portfolio_df["amc"].value_counts().to_dict(),
        "category_allocation": portfolio_df.groupby("category")["allocation_pct"].sum().to_dict(),
    }


if __name__ == "__main__":
    for basket in PORTFOLIO_BASKETS:
        name = basket["name"]
        result = build_portfolio(basket_name=name, risk_profile="moderate", amc_cap_pct=0.30)
        p = result["portfolio"]
        print(f"\n{'='*70}")
        print(f"  {name} — {len(p)} funds  |  Eq {basket['equity_pct']}%  Debt {basket['debt_pct']}%  Hyb {basket['hybrid_pct']}%")
        print(f"{'='*70}")
        if not p.empty:
            print(p[["scheme_name", "amc", "sub_category", "allocation_pct",
                      "composite_score", "trail_brokerage_incl_gst"]].to_string(index=False))
        if result["swaps"]:
            print(f"\n  Swaps: {len(result['swaps'])}")
            for s in result["swaps"]:
                print(f"    • {s}")
        if result["missing"]:
            print(f"  Missing: {result['missing']}")
