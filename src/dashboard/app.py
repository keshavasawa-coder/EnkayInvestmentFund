"""
app.py – Enkay Investments Fund Recommendation Analytics Dashboard
Streamlit application with 7 tabs:
  1. 🏆 Fund Ranker
  2. 🔍 Peer Comparison
  3. 📋 Portfolio Exposure Review
  4. 🔄 Fund Shift Advisor
  5. 🏦 AMC Concentration
  6. 📊 Brokerage vs Performance
  7. 📦 Recommended Portfolios
"""
import os, sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ───────────────────────────────────────────────────────────────
DASH_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.dirname(DASH_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)
for p in [SRC_DIR, os.path.join(SRC_DIR, "scoring"), os.path.join(SRC_DIR, "analysis")]:
    if p not in sys.path:
        sys.path.insert(0, p)

RANKED_FILE = os.path.join(BASE_DIR, "data", "processed", "ranked_funds.csv")
MASTER_FILE = os.path.join(BASE_DIR, "data", "processed", "master_scheme_table.csv")

from analysis.peer_comparison     import get_peer_comparison
from analysis.fund_shift          import suggest_alternatives
from analysis.amc_concentration   import compute_current_amc_concentration
from analysis.portfolio_review    import (
    load_aum_data,
    flag_underperforming_schemes,
    get_alternatives_for_flagged,
    AUM_THRESHOLDS,
)
from analysis.portfolio_builder   import build_portfolio, get_portfolio_stats, PORTFOLIO_BASKETS, BASKET_NAMES

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enkay Investments – Fund Analytics",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Header gradient band */
  .main-header {
    background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 50%, #93c5fd 100%);
    padding: 28px 32px 20px 32px;
    border-radius: 12px;
    margin-bottom: 24px;
    color: white;
  }
  .main-header h1 { font-size: 1.9rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
  .main-header p  { font-size: 0.95rem; opacity: 0.75; margin: 6px 0 0 0; }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(145deg, #f1f5f9, #e2e8f0);
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 16px 20px;
    color: #1e293b;
    text-align: center;
  }
  .metric-card .value { font-size: 2rem; font-weight: 700; color: #3b82f6; }
  .metric-card .label { font-size: 0.8rem; color: #64748b; margin-top: 4px; }

  /* TieUp badges */
  .badge-A    { background:#10b981; color:white; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
  .badge-B    { background:#3b82f6; color:white; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
  .badge-None { background:#475569; color:white; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

  /* Alert box */
  .alert-box { background:#fef2f2; border:1px solid #fca5a5; border-radius:8px; padding:12px 16px; color:#991b1b; }

  /* Section headers */
  .section-title { font-size:1.1rem; font-weight:600; color:#1e293b; margin:20px 0 12px 0; border-left:4px solid #3b82f6; padding-left:10px; }

  /* Streamlit table tweaks */
  .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading ranked fund data…")
def load_ranked():
    if not os.path.exists(RANKED_FILE):
        return None
    return pd.read_csv(RANKED_FILE)


@st.cache_data(show_spinner="Loading master scheme table…")
def load_master():
    if not os.path.exists(MASTER_FILE):
        return None
    return pd.read_csv(MASTER_FILE)


# ── Helpers ──────────────────────────────────────────────────────────────────
TIEUP_COLOR = {"A": "#10b981", "B": "#3b82f6", "None": "#94a3b8"}

def fmt_pct(v):
    try:
        return f"{float(v):.2f}%"
    except Exception:
        return "—"

def fmt_score(v):
    try:
        return f"{float(v):.1f}"
    except Exception:
        return "—"

def tieup_badge(t):
    t = str(t)
    return f'<span class="badge-{t}">{t} TieUp</span>' if t != "None" else f'<span class="badge-None">No TieUp</span>'

def score_bar(score, max_score=100):
    """Return a small HTML progress-like bar for a score."""
    pct = min(int(score / max_score * 100), 100)
    color = "#10b981" if pct >= 70 else "#f59e0b" if pct >= 40 else "#ef4444"
    return f'<div style="background:#e2e8f0;border-radius:4px;height:8px;width:100%"><div style="background:{color};border-radius:4px;height:8px;width:{pct}%"></div></div>'

# ── Sidebar ──────────────────────────────────────────────────────────────────

# Per-profile default weights (the 4 adjustable sliders must collectively sum  
# to 95; TieUp is always fixed at 5 on top).
PROFILE_DEFAULTS = {
    "conservative": {"w_return": 40, "w_alpha":  0, "w_brokerage": 40, "w_aum": 15},
    "moderate":     {"w_return": 35, "w_alpha": 10, "w_brokerage": 30, "w_aum": 20},
    "aggressive":   {"w_return": 40, "w_alpha": 15, "w_brokerage": 20, "w_aum": 20},
}

PAGES = [
    "🏆 Fund Ranker",
    "🔍 Peer Comparison",
    "📋 Portfolio Exposure Review",
    "🔄 Fund Shift Advisor",
    "🏦 AMC Concentration",
    "📊 Brokerage vs Performance",
    "📦 Recommended Portfolios",
]

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/investment-portfolio.png", width=60)
    st.markdown("## 📌 Navigation")
    
    selected_page = st.radio("Go to", PAGES, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("## ⚙️ Settings")

    risk_profile = st.selectbox(
        "Risk Profile",
        ["moderate", "conservative", "aggressive"],
        index=0,
        format_func=str.capitalize,
        key="risk_profile_select",
    )

    # When the profile changes, reset slider values in session_state
    if st.session_state.get("_last_profile") != risk_profile:
        for key, val in PROFILE_DEFAULTS[risk_profile].items():
            st.session_state[key] = val
        st.session_state["_last_profile"] = risk_profile

    st.markdown("---")
    st.caption("Enkay Investments | Fund Analytics v1.0")


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>💹 Enkay Investments — Fund Recommendation Analytics</h1>
  <p>Data-driven fund selection, peer comparison, fund shift advisory, and AMC concentration analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
ranked_df = load_ranked()
master_df = load_master()

if ranked_df is None or master_df is None:
    st.error("⚠️ Ranked fund data not found. Please run the data pipeline first:")
    st.code("""
# From the Grad Project directory:
py -3 src/data/01_load_performance.py
py -3 src/data/02_load_brokerage.py
py -3 src/data/03_load_tieup.py
py -3 src/data/04_merge_master.py
py -3 src/scoring/scoring_engine.py
    """)
    st.stop()

# Apply custom scoring weights and re-rank on-the-fly from master
@st.cache_data(show_spinner="Applying custom weights…")
def apply_custom_weights(weights_tuple, profile):
    from scoring.scoring_engine import rank_all
    weights = dict(zip(["return", "alpha", "brokerage", "aum", "tieup"], weights_tuple))
    return rank_all(master_df.copy(), profile, weights)

weights_tuple = (0.35, 0.10, 0.30, 0.20, 0.05)
active_df = apply_custom_weights(weights_tuple, risk_profile)

# ── Score Weights Sidebar (Right) ─────────────────────────────────────────────
with st.expander("⚖️ Score Weights", expanded=False):
    st.markdown("#### Adjust Score Weights")
    st.caption("Auto-set by Risk Profile — or override manually below.")
    
    # Get default weights for the selected risk profile
    default_weights = PROFILE_DEFAULTS[risk_profile]
    
    # Reset slider values when risk profile changes
    if st.session_state.get("_last_weight_profile") != risk_profile:
        st.session_state["w_return2"] = default_weights["w_return"]
        st.session_state["w_alpha2"] = default_weights["w_alpha"]
        st.session_state["w_brokerage2"] = default_weights["w_brokerage"]
        st.session_state["w_aum2"] = default_weights["w_aum"]
        st.session_state["_last_weight_profile"] = risk_profile
    
    w_return    = st.slider("Return Score",    0, 60, step=5,
                            key="w_return2",    help="Weight for 1Y/3Y/5Y returns")
    w_alpha     = st.slider("Alpha Score",     0, 30, step=5,
                            key="w_alpha2",     help="Weight for Information Ratio")
    w_brokerage = st.slider("Brokerage Score", 0, 60, step=5,
                            key="w_brokerage2")
    w_aum       = st.slider("AUM Score",       0, 40, step=5,
                            key="w_aum2",       help="Weight for fund size reliability")

    W_TIEUP_FIXED = 5
    st.info(f"🔒 TieUp Bonus: **{W_TIEUP_FIXED}%** (fixed across all profiles)")

    adjustable_total = w_return + w_alpha + w_brokerage + w_aum
    total_w = adjustable_total + W_TIEUP_FIXED
    if total_w == 0:
        total_w = 1
    
    custom_weights = {
        "return":    w_return        / total_w,
        "alpha":     w_alpha         / total_w,
        "brokerage": w_brokerage     / total_w,
        "aum":       w_aum           / total_w,
        "tieup":     W_TIEUP_FIXED   / total_w,
    }
    
    weights_tuple = (custom_weights["return"], custom_weights["alpha"],
                     custom_weights["brokerage"], custom_weights["aum"],
                     custom_weights["tieup"])
    active_df = apply_custom_weights(weights_tuple, risk_profile)

# ── Top KPI row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
profile_df = active_df.copy()

with col1:
    st.markdown(f'<div class="metric-card"><div class="value">{len(profile_df):,}</div><div class="label">Total Funds</div></div>', unsafe_allow_html=True)
with col2:
    brok_matched = profile_df["trail_brokerage_incl_gst"].notna().sum()
    st.markdown(f'<div class="metric-card"><div class="value">{brok_matched:,}</div><div class="label">With Brokerage Data</div></div>', unsafe_allow_html=True)
with col3:
    tieup_a = (profile_df["tieup_category"] == "A").sum()
    st.markdown(f'<div class="metric-card"><div class="value">{tieup_a}</div><div class="label">A-TieUp Funds</div></div>', unsafe_allow_html=True)
with col4:
    avg_brok = profile_df["trail_brokerage_incl_gst"].mean()
    st.markdown(f'<div class="metric-card"><div class="value">{avg_brok:.2f}%</div><div class="label">Avg Brokerage</div></div>', unsafe_allow_html=True)
with col5:
    cats = profile_df["sub_category"].nunique()
    st.markdown(f'<div class="metric-card"><div class="value">{cats}</div><div class="label">Sub-Categories</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Content Based on Selected Page ─────────────────────────────────────

# ═══════════════════════════════════════════════════════
# PAGE 1 — FUND RANKER
# ═══════════════════════════════════════════════════════
if selected_page == "🏆 Fund Ranker":
    st.markdown('<div class="section-title">Fund Rankings by Sub-Category</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ How Scoring & Ranking Works — click to expand", expanded=False):
        st.markdown("""
**Scores are relative, not absolute.**  
Each fund's score (0–100) is calculated by comparing it only to its **direct sub-category peers** — Arbitrage funds vs. Arbitrage funds, Mid Cap vs. Mid Cap, etc. This means a fund with a 6% return can score just as highly as one with a 17% return, if 6% is outstanding within its own peer group.

---

#### Step 1 — Per-Component Peer Score (0–10)
Each fund gets a score of **0 to 10** for five components, scaled using min-max normalisation **within its sub-category**:

| Component | What it measures | Scale |
|---|---|---|
| **Return** | Weighted avg of 1Y (40%), 3Y (35%), 5Y (25%) Regular returns | 0–10 vs. sub-cat peers |
| **Alpha** | 1-Year Information Ratio (how well the manager beat the benchmark per unit of risk) | 0–10 vs. sub-cat peers |
| **Brokerage** | Trail brokerage rate (Incl. GST) — higher = better for the firm | 0–10 vs. sub-cat peers |
| **AUM** | Fund size in Crores — larger AUM signals investor confidence & stability | 0–10 vs. sub-cat peers |
| **Tie-Up Bonus** | A-Category: flat 10 pts · B-Category: flat 5 pts · No Tie-Up: 0 pts | Fixed (weight locked at 5%) |

> The bottom fund in a peer group gets 0/10, the top fund gets 10/10, and all others are interpolated linearly.  
> If a fund is missing a period (e.g., no 5Y data), the available periods are re-weighted proportionally — no fund is penalised for simply being newer.

---

#### Step 2 — Composite Score (0–100)
The five component scores are combined using the weights set by the **Risk Profile** in the sidebar.  
**TieUp is always fixed at 5%.** The remaining 95% is split across the other four components:

| Profile | Return | Alpha | Brokerage | AUM | TieUp |
|---|---|---|---|---|---|
| Conservative | 40% | 0% | 40% | 15% | **5%** |
| **Moderate** | **35%** | **10%** | **30%** | **20%** | **5%** |
| Aggressive | 40% | 15% | 20% | 20% | **5%** |

---

#### Step 3 — Ranking is Category-Wise
**Rank #1 does not mean the single best fund across all 1,900+ schemes.**  
Rank #1 means the top-recommended fund *within that specific sub-category*.  
Every sub-category (Flexi Cap, Mid Cap, Arbitrage, Liquid, ELSS, etc.) has its own independent ranking starting from 1.  
This ensures you always see the best options within the type of fund you are looking for.
        """)

    c1, c2, c3 = st.columns(3)
    with c1:
        categories = sorted(profile_df["category"].dropna().unique())
        sel_cat = st.selectbox("Asset Class", ["All"] + categories)
    with c2:
        if sel_cat != "All":
            sub_cats = sorted(profile_df[profile_df["category"] == sel_cat]["sub_category"].dropna().unique())
        else:
            sub_cats = sorted(profile_df["sub_category"].dropna().unique())
        sel_subcat = st.selectbox("Sub-Category", ["All"] + sub_cats)
    with c3:
        tieup_filter = st.multiselect("TieUp Filter", ["A", "B", "No TieUp"], default=["A", "B", "No TieUp"])

    # Filter
    fdf = profile_df.copy()
    # Normalise NaN tieup to display label
    fdf["tieup_display"] = fdf["tieup_category"].fillna("No TieUp")
    if sel_cat != "All":
        fdf = fdf[fdf["category"] == sel_cat]
    if sel_subcat != "All":
        fdf = fdf[fdf["sub_category"] == sel_subcat]
    fdf = fdf[fdf["tieup_display"].isin(tieup_filter)]
    fdf = fdf.sort_values("rank")

    # Display table
    display_cols = {
        "scheme_name":              "Scheme",
        "sub_category":             "Sub-Category",
        "tieup_category":           "TieUp",
        "return_1y_regular":        "1Y Ret%",
        "return_3y_regular":        "3Y Ret%",
        "return_5y_regular":        "5Y Ret%",
        "trail_brokerage_incl_gst": "Brokerage%",
        "aum_cr":                   "AUM (Cr)",
        "composite_score":          "Score",
        "rank":                     "Rank",
    }
    show_cols = [c for c in display_cols if c in fdf.columns]
    fdf_display = fdf[show_cols].rename(columns=display_cols).copy()

    st.dataframe(
        fdf_display,
        use_container_width=True,
        height=520,
        column_config={
            "1Y Ret%":    st.column_config.NumberColumn(format="%.2f%%"),
            "3Y Ret%":    st.column_config.NumberColumn(format="%.2f%%"),
            "5Y Ret%":    st.column_config.NumberColumn(format="%.2f%%"),
            "Brokerage%": st.column_config.NumberColumn(format="%.2f%%"),
            "AUM (Cr)":   st.column_config.NumberColumn(format="%,.1f"),
            "Score":      st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(f"Showing {len(fdf_display)} funds | Risk Profile: **{risk_profile.capitalize()}**")

    # Score component breakdown chart
    if sel_subcat != "All" and len(fdf) > 0:
        st.markdown('<div class="section-title">Score Component Breakdown (Top 15)</div>', unsafe_allow_html=True)
        top15 = fdf.nsmallest(15, "rank")
        comp_cols = ["score_return", "score_alpha", "score_brokerage", "score_aum", "score_tieup"]
        comp_cols_avail = [c for c in comp_cols if c in top15.columns]
        if comp_cols_avail:
            melt = top15[["scheme_name"] + comp_cols_avail].melt(
                id_vars="scheme_name", var_name="Component", value_name="Score"
            )
            melt["Component"] = melt["Component"].str.replace("score_", "").str.capitalize()
            fig = px.bar(
                melt, x="Score", y="scheme_name", color="Component",
                orientation="h", barmode="stack",
                color_discrete_map={
                    "Return":"#3b82f6", "Alpha":"#8b5cf6",
                    "Brokerage":"#10b981", "Aum":"#f59e0b", "Tieup":"#94a3b8"
                },
                labels={"scheme_name": "", "Score": "Score (0–10)"},
                height=420,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"), legend_title_text="Component",
                yaxis=dict(tickfont=dict(size=11)),
                margin=dict(l=0, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 2 — PEER COMPARISON
# ═══════════════════════════════════════════════════════
elif selected_page == "🔍 Peer Comparison":
    st.markdown('<div class="section-title">Select Funds to Compare</div>', unsafe_allow_html=True)

    all_funds = sorted(profile_df["scheme_name"].dropna().unique())
    selected_peers = st.multiselect(
        "Choose up to 4 funds:",
        all_funds,
        max_selections=4,
        placeholder="Type a fund name…",
    )

    if selected_peers:
        peers_df = get_peer_comparison(selected_peers, risk_profile, df=active_df)

        if peers_df.empty:
            st.warning("No peer data found. Please ensure the data pipeline has been run.")
        else:
            # Highlight selected
            def highlight_selected(row):
                if row.get("is_selected", False):
                    return ["background-color: #1e3a5f; color: white"] * len(row)
                return [""] * len(row)

            show_cols = [c for c in [
                "scheme_name","sub_category","tieup_category",
                "return_1y_regular","return_3y_regular","return_5y_regular",
                "info_ratio_1y_regular","trail_brokerage_incl_gst","aum_cr",
                "composite_score","rank","is_selected"
            ] if c in peers_df.columns]

            peers_display = peers_df[show_cols].copy()
            rename = {
                "scheme_name":"Scheme","sub_category":"Sub-Cat","tieup_category":"TieUp",
                "return_1y_regular":"1Y Ret%","return_3y_regular":"3Y Ret%",
                "return_5y_regular":"5Y Ret%","info_ratio_1y_regular":"Info Ratio",
                "trail_brokerage_incl_gst":"Brokerage%","aum_cr":"AUM (Cr)",
                "composite_score":"Score","rank":"Rank","is_selected":"Selected?",
            }
            peers_display = peers_display.rename(columns={k:v for k,v in rename.items() if k in peers_display.columns})

            st.dataframe(
                peers_display.style
                .apply(highlight_selected, axis=1)
                .format({
                    "1Y Ret%": "{:.2f}%", 
                    "3Y Ret%": "{:.2f}%", 
                    "5Y Ret%": "{:.2f}%",
                    "Brokerage%": "{:.3f}%",
                    "AUM (Cr)": "{:,.1f}",
                    "Score": "{:.1f}",
                    "Info Ratio": "{:.2f}"
                }, na_rep="—"), 
                use_container_width=True, 
                height=480
            )

            # Radar chart for selected funds
            st.markdown('<div class="section-title">Radar Chart — Score Components</div>', unsafe_allow_html=True)
            radar_cols = ["score_return", "score_alpha", "score_brokerage", "score_tieup"]
            radar_avail= [c for c in radar_cols if c in peers_df.columns]
            if radar_avail and len(selected_peers) > 0:
                selected_radar = peers_df[peers_df["is_selected"]][["scheme_name"] + radar_avail]
                fig_radar = go.Figure()
                categories_radar = [c.replace("score_","").capitalize() for c in radar_avail]
                colors = ["#3b82f6","#10b981","#f59e0b","#ef4444"]
                for i, (_, row) in enumerate(selected_radar.iterrows()):
                    vals = [row[c] for c in radar_avail]
                    vals += [vals[0]]
                    cats = categories_radar + [categories_radar[0]]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals, theta=cats,
                        fill="toself", name=row["scheme_name"],
                        line_color=colors[i % len(colors)],
                        fillcolor=colors[i % len(colors)].replace("#", "rgba(").replace(")", ",0.15)") if False else colors[i % len(colors)],
                        opacity=0.8,
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                    showlegend=True, height=420,
                    paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
                    margin=dict(l=40, r=40, t=20, b=20),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                # Selected Funds Summary Table
                st.markdown('<div class="section-title">Selected Funds — Comparison Summary</div>', unsafe_allow_html=True)
                selected_summary = peers_display[peers_display["Selected?"] == True].drop(columns=["Selected?"])
                st.dataframe(
                    selected_summary.style.format({
                        "1Y Ret%": "{:.2f}%", 
                        "3Y Ret%": "{:.2f}%", 
                        "5Y Ret%": "{:.2f}%",
                        "Brokerage%": "{:.3f}%",
                        "AUM (Cr)": "{:,.1f}",
                        "Score": "{:.1f}",
                        "Info Ratio": "{:.2f}"
                    }, na_rep="—"),
                    use_container_width=True
                )
    else:
        st.info("👆 Select 2–4 funds above to start comparing.")

# ═══════════════════════════════════════════════════════
# PAGE 3 — PORTFOLIO EXPOSURE REVIEW
# ═══════════════════════════════════════════════════════
elif selected_page == "📋 Portfolio Exposure Review":
    st.markdown('<div class="section-title">Analyze Your Current Holdings</div>', unsafe_allow_html=True)
    st.markdown(
        "This page shows your current AUM holdings and flags schemes that have high exposure but "
        "underperform on score or brokerage. Review flagged schemes and consider switching to better alternatives.",
        unsafe_allow_html=True
    )
    
    aum_col1, aum_col2, aum_col3 = st.columns([2, 2, 1])
    with aum_col1:
        aum_threshold_label = st.selectbox(
            "Minimum AUM Threshold",
            list(AUM_THRESHOLDS.keys()),
            index=3,
            help="Only flag schemes with AUM above this threshold"
        )
    with aum_col2:
        include_brok = st.checkbox("Flag Low Brokerage", value=True, 
                                   help="Also flag schemes with below-median brokerage in their category")
    with aum_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh_btn = st.button("🔄 Analyze Portfolio", use_container_width=True)
    
    aum_threshold = AUM_THRESHOLDS[aum_threshold_label]
    
    with st.spinner("Loading AUM data and analyzing..."):
        aum_df = load_aum_data()
        analysis_result = flag_underperforming_schemes(
            aum_df, active_df,
            risk_profile=risk_profile,
            aum_threshold=aum_threshold,
            score_percentile=50,
            include_brokerage_flag=include_brok,
        )
    
    summary = analysis_result["summary"]
    
    st.divider()
    st.markdown("### 📊 Portfolio Summary")
    
    ps1, ps2, ps3, ps4, ps5 = st.columns(5)
    with ps1:
        st.markdown(f'<div class="metric-card"><div class="value">₹{summary["total_aum"]/10000000:.2f}Cr</div><div class="label">Total AUM</div></div>', unsafe_allow_html=True)
    with ps2:
        st.markdown(f'<div class="metric-card"><div class="value">{summary["total_schemes"]}</div><div class="label">Total Schemes</div></div>', unsafe_allow_html=True)
    with ps3:
        st.markdown(f'<div class="metric-card"><div class="value">{summary["matched_schemes"]}</div><div class="label">Matched Schemes</div></div>', unsafe_allow_html=True)
    with ps4:
        st.markdown(f'<div class="metric-card"><div class="value">{summary.get("schemes_above_threshold", 0)}</div><div class="label">Above Threshold</div></div>', unsafe_allow_html=True)
    with ps5:
        flagged_count = summary["flagged_count"]
        color = "#ef4444" if flagged_count > 0 else "#10b981"
        st.markdown(f'<div class="metric-card" style="border-color:{color}"><div class="value" style="color:{color}">{flagged_count}</div><div class="label">Flagged</div></div>', unsafe_allow_html=True)
    
    aum_by_asset = analysis_result.get("aum_by_asset", {})
    if aum_by_asset:
        st.markdown("### 📈 AUM by Asset Class")
        asset_df = pd.DataFrame(list(aum_by_asset.items()), columns=["Asset Class", "AUM"])
        asset_df = asset_df.sort_values("AUM", ascending=False)
        
        fig_pie = px.pie(
            asset_df, 
            values="AUM", 
            names="Asset Class",
            title="AUM Distribution by Asset Class",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
        st.plotly_chart(fig_pie, use_container_width=True)
    
    flagged_df = analysis_result.get("flagged", pd.DataFrame())
    
    if flagged_df.empty:
        st.success("✅ No flagged schemes! All your holdings are performing well.")
    else:
        st.markdown(f"### ⚠️ Flagged Schemes ({len(flagged_df)} schemes)")
        st.markdown("These schemes have high AUM but rank in the bottom 50% of their category or have below-median brokerage.")
        
        display_cols = {
            "scheme": "Scheme",
            "amc": "AMC",
            "total": "AUM (₹)",
            "sub_category": "Category",
            "composite_score": "Score",
            "rank": "Rank",
            "trail_brokerage_incl_gst": "Brokerage%",
            "score_flag": "Low Score",
            "brokerage_flag": "Low Brokerage",
        }
        
        show_cols = [c for c in display_cols.keys() if c in flagged_df.columns]
        flagged_display = flagged_df[show_cols].rename(columns=display_cols).copy()
        
        if "AUM (₹)" in flagged_display.columns:
            flagged_display["AUM (₹)"] = flagged_display["AUM (₹)"].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")
        if "Score" in flagged_display.columns:
            flagged_display["Score"] = flagged_display["Score"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        if "Brokerage%" in flagged_display.columns:
            flagged_display["Brokerage%"] = flagged_display["Brokerage%"].apply(lambda x: f"{x:.3f}%" if pd.notna(x) else "—")
        if "Rank" in flagged_display.columns:
            flagged_display["Rank"] = flagged_display["Rank"].apply(lambda x: f"#{int(x)}" if pd.notna(x) else "—")
        
        st.dataframe(flagged_display, use_container_width=True, height=400)
        
        with st.expander("View Better Alternatives for Flagged Schemes", expanded=False):
            alternatives = get_alternatives_for_flagged(flagged_df, active_df, risk_profile=risk_profile, n=3)
            if not alternatives.empty:
                alt_cols = {
                    "flagged_scheme": "Current Scheme",
                    "flagged_aum": "Current AUM",
                    "flagged_score": "Current Score",
                    "alternative_scheme": "Better Alternative",
                    "alternative_score": "Alt Score",
                    "alternative_brokerage": "Alt Brokerage%",
                    "alternative_return_1y": "Alt 1Y Return%",
                    "alternative_rank": "Alt Rank",
                    "sub_category": "Category",
                }
                alt_show = [c for c in alt_cols.keys() if c in alternatives.columns]
                alt_display = alternatives[alt_show].rename(columns=alt_cols).copy()
                
                if "Current AUM" in alt_display.columns:
                    alt_display["Current AUM"] = alt_display["Current AUM"].apply(lambda x: f"₹{x:,.0f}")
                if "Alt Score" in alt_display.columns:
                    alt_display["Alt Score"] = alt_display["Alt Score"].apply(lambda x: f"{x:.1f}")
                if "Alt Brokerage%" in alt_display.columns:
                    alt_display["Alt Brokerage%"] = alt_display["Alt Brokerage%"].apply(lambda x: f"{x:.3f}%" if pd.notna(x) else "—")
                if "Alt 1Y Return%" in alt_display.columns:
                    alt_display["Alt 1Y Return%"] = alt_display["Alt 1Y Return%"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")
                if "Alt Rank" in alt_display.columns:
                    alt_display["Alt Rank"] = alt_display["Alt Rank"].apply(lambda x: f"#{int(x)}" if pd.notna(x) else "—")
                
                st.dataframe(alt_display, use_container_width=True, height=400)
            else:
                st.info("No alternatives found for flagged schemes.")


# ═══════════════════════════════════════════════════════
# PAGE 4 — FUND SHIFT ADVISOR
# ═══════════════════════════════════════════════════════
elif selected_page == "🔄 Fund Shift Advisor":
    st.markdown('<div class="section-title">Find Better-Paying Alternatives</div>', unsafe_allow_html=True)
    st.markdown(
        "Select a fund whose brokerage has dropped or that you want to replace. "
        "The advisor will suggest peers with **equal or better brokerage** and competitive performance.",
        unsafe_allow_html=True
    )

    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        shift_fund = st.selectbox("Choose a fund to replace:", [""] + sorted(active_df["scheme_name"].dropna().unique()))
    with col_s2:
        n_alternatives = st.number_input("# alternatives", 1, 10, 3)

    if shift_fund:
        # Show selected fund info
        sel_info = active_df[active_df["scheme_name"] == shift_fund].iloc[0]
        st.markdown("#### Selected Fund")
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            st.metric("Sub-Category",   sel_info.get("sub_category","—"))
        with s2:
            st.metric("TieUp",          str(sel_info.get("tieup_category","—")))
        with s3:
            bval = sel_info.get("trail_brokerage_incl_gst", None)
            st.metric("Brokerage",      f"{bval:.3f}%" if pd.notna(bval) else "—")
        with s4:
            r1 = sel_info.get("return_1y_regular", None)
            st.metric("1Y Return",      f"{r1:.2f}%" if pd.notna(r1) else "—")
        with s5:
            st.metric("Composite Score", f"{sel_info.get('composite_score',0):.1f}")

        alternatives = suggest_alternatives(shift_fund, risk_profile, n=n_alternatives, df=active_df)

        st.markdown("#### Suggested Alternatives")
        if alternatives.empty:
            st.warning("No suitable alternatives found in the same sub-category with equal or higher brokerage.")
        else:
            # Colour-code delta columns
            for _, alt_row in alternatives.iterrows():
                with st.container():
                    ac1, ac2, ac3, ac4, ac5, ac6 = st.columns([3,1,1,1,1,1])
                    with ac1:
                        st.markdown(f"**{alt_row['scheme_name']}**")
                        st.caption(f"TieUp: {alt_row.get('tieup_category','—')} | Rank #{int(alt_row.get('rank',0))}")
                    with ac2:
                        dbroker = alt_row.get("delta_brokerage", None)
                        color   = "normal" if pd.isna(dbroker) else "normal"
                        st.metric("Brokerage", fmt_pct(alt_row.get("trail_brokerage_incl_gst",None)),
                                  delta=f"{dbroker:+.3f}%" if pd.notna(dbroker) else None)
                    with ac3:
                        dr1 = alt_row.get("delta_return_1y", None)
                        st.metric("1Y Return", fmt_pct(alt_row.get("return_1y_regular",None)),
                                  delta=f"{dr1:+.2f}%" if pd.notna(dr1) else None)
                    with ac4:
                        dr3 = alt_row.get("delta_return_3y", None)
                        st.metric("3Y Return", fmt_pct(alt_row.get("return_3y_regular",None)),
                                  delta=f"{dr3:+.2f}%" if pd.notna(dr3) else None)
                    with ac5:
                        r5 = alt_row.get("return_5y_regular", None)
                        st.metric("5Y Return", fmt_pct(r5))
                    with ac6:
                        dcomp = alt_row.get("delta_composite", None)
                        st.metric("Score", fmt_score(alt_row.get("composite_score",None)),
                                  delta=f"{dcomp:+.1f}" if pd.notna(dcomp) else None)
                    st.divider()
    else:
        st.info("👆 Select a fund above to see suggestions.")

# ═══════════════════════════════════════════════════════
# PAGE 5 — AMC CONCENTRATION
# ═══════════════════════════════════════════════════════
elif selected_page == "🏦 AMC Concentration":
    st.markdown('<div class="section-title">AMC Exposure — Current Holdings</div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Current Holdings AMC Concentration")
    st.markdown("This section shows AMC distribution based on your actual AUM holdings.")
    
    amc_col1, amc_col2 = st.columns([2, 1])
    with amc_col1:
        aum_thresh_label = st.selectbox(
            "Minimum AUM Threshold (Current Holdings)",
            list(AUM_THRESHOLDS.keys()),
            index=3,
            key="amc_aum_thresh",
            help="Only include schemes with AUM above this threshold"
        )
    with amc_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh_current = st.button("🔄 Analyze Current Holdings", use_container_width=True)
    
    aum_thresh_val = AUM_THRESHOLDS[aum_thresh_label]
    
    with st.spinner("Analyzing current holdings..."):
        current_amc_result = compute_current_amc_concentration(
            aum_threshold=aum_thresh_val,
            ranked_df=active_df,
        )
    
    cur_summary = current_amc_result["summary"]
    
    camc1, camc2, camc3, camc4 = st.columns(4)
    with camc1:
        st.markdown(f'<div class="metric-card"><div class="value">₹{current_amc_result["total_aum"]/10000000:.2f}Cr</div><div class="label">Total AUM</div></div>', unsafe_allow_html=True)
    with camc2:
        st.markdown(f'<div class="metric-card"><div class="value">{current_amc_result["total_amcs"]}</div><div class="label">Total AMCs</div></div>', unsafe_allow_html=True)
    with camc3:
        st.markdown(f'<div class="metric-card"><div class="value">{current_amc_result["schemes_matched"]}</div><div class="label">Schemes</div></div>', unsafe_allow_html=True)
    with camc4:
        cur_alerts = cur_summary[cur_summary["alert"]]["amc"].tolist() if not cur_summary.empty else []
        alert_count = len(cur_alerts)
        color = "#ef4444" if alert_count > 0 else "#10b981"
        st.markdown(f'<div class="metric-card" style="border-color:{color}"><div class="value" style="color:{color}">{alert_count}</div><div class="label">High Concentration</div></div>', unsafe_allow_html=True)
    
    if not cur_summary.empty:
        cur_alert_list = cur_summary[cur_summary["alert"]]["amc"].tolist()
        if cur_alert_list:
            alert_str = ", ".join(cur_alert_list)
            st.markdown(
                f'<div class="alert-box">⚠️ <strong>Concentration Alert!</strong> '
                f'The following AMC(s) represent >30% of your current AUM: <strong>{alert_str}</strong>.</div>',
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
        
        fig_cur_tree = px.treemap(
            cur_summary,
            path=["amc"],
            values="aum",
            color="pct",
            color_continuous_scale=["#1e40af","#3b82f6","#93c5fd","#fef08a","#fca5a5","#dc2626"],
            color_continuous_midpoint=0.15,
            custom_data=["pct", "aum"],
            title=f"AMC Distribution — Current Holdings (AUM ≥ {aum_thresh_label})",
        )
        fig_cur_tree.update_traces(
            texttemplate="<b>%{label}</b><br>%{customdata[0]:.1%}",
            hovertemplate="<b>%{label}</b><br>AUM: ₹%{customdata[1]:,.0f}<br>Share: %{customdata[0]:.1%}<extra></extra>",
        )
        fig_cur_tree.update_layout(
            height=380, margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
        )
        st.plotly_chart(fig_cur_tree, use_container_width=True)
        
        cur_display = cur_summary.copy()
        cur_display["aum_fmt"] = cur_display["aum"].apply(lambda x: f"₹{x:,.0f}")
        cur_display["pct_fmt"] = cur_display["pct"].apply(lambda x: f"{x:.1%}")
        cur_display["status"] = cur_display["alert"].apply(lambda x: "⚠️ HIGH" if x else "✅ OK")
        st.dataframe(
            cur_display[["amc", "aum_fmt", "pct_fmt", "status"]].rename(columns={"amc": "AMC", "aum_fmt": "AUM (₹)", "pct_fmt": "Share %", "status": "Status"}),
            use_container_width=True,
            height=300,
        )
# ═══════════════════════════════════════════════════════
# PAGE 6 — BROKERAGE vs PERFORMANCE
# ═══════════════════════════════════════════════════════
elif selected_page == "📊 Brokerage vs Performance":
    st.markdown('<div class="section-title">Brokerage vs Returns — Scatter Explorer</div>', unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        x_axis = st.selectbox("X-Axis (Returns)", [
            "return_1y_regular","return_3y_regular","return_5y_regular","composite_score"
        ], index=1, format_func=lambda x: x.replace("return_","").replace("_regular"," Return (Regular)").replace("composite_score","Composite Score"))
    with b2:
        cat_filter_scatter = st.multiselect(
            "Asset Class",
            sorted(active_df["category"].dropna().unique()),
            default=list(sorted(active_df["category"].dropna().unique())),
            key="scatter_cat"
        )
    with b3:
        min_aum = st.number_input("Min AUM (Cr.)", 0, 100000, 0, step=500)

    scatter_df = active_df[
        (active_df["category"].isin(cat_filter_scatter)) &
        (active_df["trail_brokerage_incl_gst"].notna()) &
        (active_df[x_axis].notna())
    ].copy()

    if min_aum > 0 and "aum_cr" in scatter_df.columns:
        scatter_df = scatter_df[scatter_df["aum_cr"] >= min_aum]

    if scatter_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # ── Fix: fill NaN aum_cr before using as bubble size ──────────────
        use_size = None
        if "aum_cr" in scatter_df.columns and scatter_df["aum_cr"].notna().sum() > 0:
            median_aum = scatter_df["aum_cr"].median()
            scatter_df["aum_cr_plot"] = scatter_df["aum_cr"].fillna(median_aum).clip(lower=1)
            use_size = "aum_cr_plot"
        # Normalise tieup NaN for colour
        scatter_df["tieup_label"] = scatter_df["tieup_category"].fillna("No TieUp")

        fig_scatter = px.scatter(
            scatter_df,
            x=x_axis,
            y="trail_brokerage_incl_gst",
            color="tieup_label",
            size=use_size,
            size_max=35,
            hover_name="scheme_name",
            hover_data={
                "sub_category": True,
                "tieup_label": True,
                "composite_score": ":.1f",
                "trail_brokerage_incl_gst": ":.3f",
                x_axis: ":.2f",
                "aum_cr": ":.1f",
            },
            color_discrete_map={"A": "#10b981", "B": "#3b82f6", "No TieUp": "#94a3b8"},
            labels={
                x_axis: x_axis.replace("return_","").replace("_regular"," Return (Regular)%").replace("composite_score","Composite Score"),
                "trail_brokerage_incl_gst": "Trail Brokerage (Incl. GST) %",
                "tieup_label": "TieUp",
            },
            title=f"Brokerage vs {x_axis} — {risk_profile.capitalize()} Profile",
            height=560,
        )

        # Add quadrant lines
        x_med = scatter_df[x_axis].median()
        y_med = scatter_df["trail_brokerage_incl_gst"].median()
        fig_scatter.add_hline(y=y_med, line_dash="dot", line_color="#475569", annotation_text="Median Brokerage", annotation_position="bottom right")
        fig_scatter.add_vline(x=x_med, line_dash="dot", line_color="#475569", annotation_text="Median Return", annotation_position="top left")

        # Quadrant labels
        fig_scatter.add_annotation(text="✅ HIGH RETURN<br>HIGH BROKERAGE",
            x=scatter_df[x_axis].quantile(0.85), y=scatter_df["trail_brokerage_incl_gst"].quantile(0.85),
            showarrow=False, font=dict(size=10, color="#10b981"), opacity=0.6)
        fig_scatter.add_annotation(text="⚠️ HIGH RETURN<br>LOW BROKERAGE",
            x=scatter_df[x_axis].quantile(0.85), y=scatter_df["trail_brokerage_incl_gst"].quantile(0.15),
            showarrow=False, font=dict(size=10, color="#f59e0b"), opacity=0.6)

        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
            font=dict(family="Inter"),
            xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1"),
            yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # High-value quadrant table
        st.markdown('<div class="section-title">⭐ High Return + High Brokerage Funds</div>', unsafe_allow_html=True)
        hq = scatter_df[
            (scatter_df[x_axis] >= x_med) &
            (scatter_df["trail_brokerage_incl_gst"] >= y_med)
        ].nlargest(20, "composite_score")
        # Create unique column list to avoid duplicate column crashes (if x_axis == composite_score)
        hq_cols = ["scheme_name", "sub_category", "tieup_category", x_axis, "trail_brokerage_incl_gst", "composite_score", "rank"]
        hq_show = []
        for c in hq_cols:
            if c not in hq_show and c in hq.columns:
                hq_show.append(c)

        # Dynamic rename based on what x_axis is
        rename_dict = {
            "scheme_name":              "Scheme",
            "sub_category":             "Sub-Cat",
            "tieup_category":           "TieUp",
            "trail_brokerage_incl_gst": "Brokerage%",
            "composite_score":          "Score",
            "rank":                     "Rank",
        }
        # Only add a dynamic rename for x_axis if it's not already one of the standard columns
        if x_axis not in rename_dict:
             rename_dict[x_axis] = "Return%"

        st.dataframe(
            hq[hq_show].rename(columns=rename_dict),
            use_container_width=True,
            height=340,
            column_config={
                "Return%":    st.column_config.NumberColumn(format="%.2f%%"),
                "Brokerage%": st.column_config.NumberColumn(format="%.2f%%"),
                "Score":      st.column_config.NumberColumn(format="%.1f"),
            }
        )

# ═══════════════════════════════════════════════════════════
# PAGE 7 — RECOMMENDED PORTFOLIOS
# ═══════════════════════════════════════════════════════════
elif selected_page == "📦 Recommended Portfolios":
    st.markdown('<div class="section-title">Enkay Recommended MF Portfolios</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ What are Recommended Portfolios? — click to expand", expanded=False):
        st.markdown("""
**Enkay Recommended Portfolios** are curated baskets of mutual fund schemes, modeled on
NJ's Recommended MF Portfolio approach but built **by Enkay, for Enkay**.

Each basket defines:
- **Asset-class allocation** (Equity / Debt / Hybrid %)
- **Sub-category picks** from which the top-ranked fund is auto-selected
- **Risk level** indicator

The scoring engine picks the **Rank 1 fund** per sub-category using the active risk profile
and scoring weights from the sidebar. An **AMC concentration cap** ensures diversification.
        """)

    # ── Controls ─────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([3, 2, 2])
    with ctrl1:
        selected_basket = st.selectbox(
            "Select Portfolio Basket",
            BASKET_NAMES,
            index=BASKET_NAMES.index("Balanced - Equity 50"),
            key="basket_select",
        )
    with ctrl2:
        amc_cap = st.slider(
            "AMC Concentration Cap",
            min_value=10, max_value=50, value=30, step=5,
            format="%d%%",
            help="Max % of portfolio slots one AMC can occupy.",
            key="portfolio_amc_cap",
        )
    with ctrl3:
        compare_mode = st.checkbox(
            "Compare Multiple Baskets",
            value=False,
            key="portfolio_compare",
        )

    amc_cap_frac = amc_cap / 100.0

    # ── All Baskets Overview Table ────────────────────────────────────────
    with st.expander("📋 View All Available Baskets", expanded=False):
        overview_rows = []
        for b in PORTFOLIO_BASKETS:
            overview_rows.append({
                "Portfolio": b["name"],
                "Equity %": b["equity_pct"],
                "Debt %": b["debt_pct"],
                "Hybrid %": b["hybrid_pct"],
                "Schemes": len(b["picks"]),
                "Risk Level": b["risk_level"],
            })
        overview_df = pd.DataFrame(overview_rows)
        st.dataframe(overview_df, use_container_width=True, height=400)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Single Basket View ───────────────────────────────────────────────
    if not compare_mode:
        result = build_portfolio(
            ranked_df=active_df, basket_name=selected_basket,
            risk_profile=risk_profile, amc_cap_pct=amc_cap_frac,
        )
        portfolio = result["portfolio"]
        basket = result["basket"]
        swaps = result["swaps"]
        missing = result["missing"]

        if portfolio.empty:
            st.warning("Could not build this portfolio — no matching funds found.")
        else:
            stats = get_portfolio_stats(portfolio)

            # ── Basket header card ───────────────────────────────────────
            risk_colors = {
                "Low to Moderate": "#10b981",
                "Moderately High": "#f59e0b",
                "High": "#f97316",
                "Very High": "#ef4444",
            }
            rcolor = risk_colors.get(basket["risk_level"], "#64748b")
            eq, dt, hy = basket["equity_pct"], basket["debt_pct"], basket["hybrid_pct"]

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#f8fafc,#e2e8f0);border:1px solid #cbd5e1;
                        border-radius:12px;padding:20px 28px;margin-bottom:20px;">
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px;">
                    <div>
                        <h2 style="margin:0;color:#1e293b;">{basket['name']}</h2>
                        <span style="background:{rcolor};color:white;padding:4px 14px;border-radius:20px;
                                     font-size:0.8rem;font-weight:600;">{basket['risk_level']}</span>
                    </div>
                    <div style="display:flex;gap:24px;text-align:center;">
                        <div><div style="font-size:1.8rem;font-weight:700;color:#3b82f6;">{eq}%</div>
                             <div style="font-size:0.75rem;color:#64748b;">Equity</div></div>
                        <div><div style="font-size:1.8rem;font-weight:700;color:#10b981;">{dt}%</div>
                             <div style="font-size:0.75rem;color:#64748b;">Debt</div></div>
                        <div><div style="font-size:1.8rem;font-weight:700;color:#f59e0b;">{hy}%</div>
                             <div style="font-size:0.75rem;color:#64748b;">Hybrid</div></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── KPI row ──────────────────────────────────────────────────
            pk1, pk2, pk3, pk4, pk5 = st.columns(5)
            with pk1:
                st.markdown(f'<div class="metric-card"><div class="value">{stats["num_schemes"]}</div><div class="label">Schemes</div></div>', unsafe_allow_html=True)
            with pk2:
                st.markdown(f'<div class="metric-card"><div class="value">{stats["weighted_avg_brok"]:.2f}%</div><div class="label">Wtd Avg Brokerage</div></div>', unsafe_allow_html=True)
            with pk3:
                st.markdown(f'<div class="metric-card"><div class="value">{stats["weighted_avg_ret1y"]:.2f}%</div><div class="label">Wtd Avg 1Y Return</div></div>', unsafe_allow_html=True)
            with pk4:
                st.markdown(f'<div class="metric-card"><div class="value">{stats["num_amcs"]}</div><div class="label">AMCs</div></div>', unsafe_allow_html=True)
            with pk5:
                st.markdown(f'<div class="metric-card"><div class="value">{stats["avg_score"]:.1f}</div><div class="label">Avg Score</div></div>', unsafe_allow_html=True)

            # ── Swap alerts ──────────────────────────────────────────────
            if swaps:
                st.markdown("<br>", unsafe_allow_html=True)
                for swap_msg in swaps:
                    st.warning(f"🔄 {swap_msg}")
            if missing:
                st.info(f"ℹ️ Sub-categories not available in data: {', '.join(missing)}")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Charts row ───────────────────────────────────────────────
            num_funds = len(portfolio)
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown('<div class="section-title">Sub-Category Allocation</div>', unsafe_allow_html=True)
                alloc_df = portfolio[["sub_category", "allocation_pct"]].copy()
                fig_alloc = px.pie(
                    alloc_df, values="allocation_pct", names="sub_category",
                    hole=0.45, color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_alloc.update_traces(textposition="inside", textinfo="percent+label", textfont_size=11)
                fig_alloc.update_layout(
                    height=370, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"), margin=dict(l=0, r=0, t=10, b=10),
                )
                st.plotly_chart(fig_alloc, use_container_width=True)

            with chart_col2:
                if num_funds > 1:
                    st.markdown('<div class="section-title">AMC Distribution</div>', unsafe_allow_html=True)
                    amc_dist = portfolio["amc"].value_counts().reset_index()
                    amc_dist.columns = ["AMC", "Funds"]
                    fig_amc = px.bar(
                        amc_dist, x="Funds", y="AMC", orientation="h",
                        color="Funds", color_continuous_scale=["#93c5fd", "#3b82f6", "#1e40af"],
                        text="Funds",
                    )
                    fig_amc.update_traces(textposition="outside")
                    fig_amc.update_layout(
                        height=370, showlegend=False, coloraxis_showscale=False,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter"),
                        xaxis=dict(title="", showticklabels=False),
                        yaxis=dict(title="", tickfont=dict(size=11)),
                        margin=dict(l=0, r=30, t=10, b=10),
                    )
                    st.plotly_chart(fig_amc, use_container_width=True)
                else:
                    st.markdown('<div class="section-title">Fund Details</div>', unsafe_allow_html=True)
                    fund = portfolio.iloc[0]
                    st.metric("Scheme", fund["scheme_name"])
                    st.metric("AMC", fund["amc"])
                    bval = fund.get("trail_brokerage_incl_gst", None)
                    st.metric("Brokerage", f"{bval:.2f}%" if pd.notna(bval) else "—")

            # -- Asset-class allocation bar (only if multi-class basket) --
            if stats.get("category_allocation") and len(stats["category_allocation"]) > 1:
                st.markdown('<div class="section-title">Allocation by Asset Class</div>', unsafe_allow_html=True)
                cat_alloc = pd.DataFrame(
                    list(stats["category_allocation"].items()), columns=["Asset Class", "Allocation %"]
                ).sort_values("Allocation %", ascending=False)
                fig_cat = px.bar(
                    cat_alloc, x="Allocation %", y="Asset Class",
                    orientation="h", text="Allocation %",
                    color="Asset Class", color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_cat.update_traces(texttemplate="%{text}%", textposition="outside")
                fig_cat.update_layout(
                    height=250, showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                    xaxis=dict(title="", showticklabels=False), yaxis=dict(title=""),
                    margin=dict(l=0, r=40, t=10, b=10),
                )
                st.plotly_chart(fig_cat, use_container_width=True)

            # ── Selected Funds Table ─────────────────────────────────────
            st.markdown('<div class="section-title">Selected Funds</div>', unsafe_allow_html=True)
            fund_display_cols = {
                "scheme_name":              "Scheme",
                "amc":                      "AMC",
                "sub_category":             "Sub-Category",
                "allocation_pct":           "Allocation %",
                "tieup_category":           "TieUp",
                "return_1y_regular":        "1Y Ret%",
                "return_3y_regular":        "3Y Ret%",
                "return_5y_regular":        "5Y Ret%",
                "trail_brokerage_incl_gst": "Brokerage%",
                "aum_cr":                   "AUM (Cr)",
                "composite_score":          "Score",
                "rank":                     "Rank",
            }
            show_cols = [c for c in fund_display_cols if c in portfolio.columns]
            port_display = portfolio[show_cols].rename(columns=fund_display_cols).copy()

            st.dataframe(
                port_display, use_container_width=True,
                height=min(440, 60 + 35 * len(port_display)),
                column_config={
                    "Allocation %": st.column_config.NumberColumn(format="%d%%"),
                    "1Y Ret%":      st.column_config.NumberColumn(format="%.2f%%"),
                    "3Y Ret%":      st.column_config.NumberColumn(format="%.2f%%"),
                    "5Y Ret%":      st.column_config.NumberColumn(format="%.2f%%"),
                    "Brokerage%":   st.column_config.NumberColumn(format="%.2f%%"),
                    "AUM (Cr)":     st.column_config.NumberColumn(format="%,.1f"),
                    "Score":        st.column_config.NumberColumn(format="%.1f"),
                },
            )

            # ── TieUp Summary ────────────────────────────────────────────
            tu1, tu2, tu3 = st.columns(3)
            with tu1:
                st.markdown(f'<div class="metric-card" style="border-left:4px solid #10b981"><div class="value" style="color:#10b981">{stats["tieup_a_count"]}</div><div class="label">A-TieUp Funds</div></div>', unsafe_allow_html=True)
            with tu2:
                st.markdown(f'<div class="metric-card" style="border-left:4px solid #3b82f6"><div class="value" style="color:#3b82f6">{stats["tieup_b_count"]}</div><div class="label">B-TieUp Funds</div></div>', unsafe_allow_html=True)
            with tu3:
                st.markdown(f'<div class="metric-card" style="border-left:4px solid #94a3b8"><div class="value" style="color:#94a3b8">{stats["no_tieup_count"]}</div><div class="label">No TieUp</div></div>', unsafe_allow_html=True)

    else:
        # ── Compare Multiple Baskets ─────────────────────────────────────
        compare_baskets = st.multiselect(
            "Select Baskets to Compare (2–4):",
            BASKET_NAMES,
            default=["Conservative - Equity 30", "Balanced - Equity 50", "Growth - Equity 100"],
            max_selections=4,
            key="basket_compare_select",
        )

        if len(compare_baskets) < 2:
            st.info("👆 Select at least 2 baskets to compare.")
        else:
            all_results = {}
            all_stats = {}
            for bname in compare_baskets:
                r = build_portfolio(ranked_df=active_df, basket_name=bname,
                                    risk_profile=risk_profile, amc_cap_pct=amc_cap_frac)
                all_results[bname] = r
                all_stats[bname] = get_portfolio_stats(r["portfolio"]) if not r["portfolio"].empty else {}

            # ── Comparison cards ─────────────────────────────────────────
            card_cols = st.columns(len(compare_baskets))
            card_colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
            for i, (bname, col) in enumerate(zip(compare_baskets, card_cols)):
                s = all_stats.get(bname, {})
                b = all_results[bname]["basket"]
                clr = card_colors[i % len(card_colors)]
                with col:
                    st.markdown(f"""
                    <div style="border:2px solid {clr};border-radius:12px;padding:14px;text-align:center;">
                        <h4 style="color:{clr};margin:0 0 8px 0;font-size:0.95rem;">{bname}</h4>
                        <div style="font-size:0.75rem;color:#64748b;margin-bottom:10px;">
                            Eq {b['equity_pct']}% · Debt {b['debt_pct']}% · Hyb {b['hybrid_pct']}%
                        </div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:0.85rem;">
                            <div><strong>{s.get('num_schemes', 0)}</strong><br><small style="color:#94a3b8">Schemes</small></div>
                            <div><strong>{s.get('num_amcs', 0)}</strong><br><small style="color:#94a3b8">AMCs</small></div>
                            <div><strong>{s.get('weighted_avg_brok', 0):.2f}%</strong><br><small style="color:#94a3b8">Brokerage</small></div>
                            <div><strong>{s.get('weighted_avg_ret1y', 0):.2f}%</strong><br><small style="color:#94a3b8">1Y Return</small></div>
                            <div><strong>{s.get('avg_score', 0):.1f}</strong><br><small style="color:#94a3b8">Avg Score</small></div>
                            <div><strong>{s.get('tieup_a_count', 0)}</strong><br><small style="color:#94a3b8">A-TieUp</small></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Combined Table ───────────────────────────────────────────
            st.markdown('<div class="section-title">Fund Selections Across Baskets</div>', unsafe_allow_html=True)
            combined_rows = []
            for bname in compare_baskets:
                p = all_results[bname]["portfolio"]
                if not p.empty:
                    view = p[["scheme_name", "amc", "sub_category", "allocation_pct",
                              "composite_score", "trail_brokerage_incl_gst", "tieup_category"]].copy()
                    view.insert(0, "Basket", bname)
                    combined_rows.append(view)

            if combined_rows:
                combined_df = pd.concat(combined_rows, ignore_index=True)
                combined_df = combined_df.rename(columns={
                    "scheme_name": "Scheme", "amc": "AMC", "sub_category": "Sub-Category",
                    "allocation_pct": "Alloc %", "composite_score": "Score",
                    "trail_brokerage_incl_gst": "Brokerage%", "tieup_category": "TieUp",
                })
                st.dataframe(
                    combined_df, use_container_width=True, height=520,
                    column_config={
                        "Alloc %":    st.column_config.NumberColumn(format="%d%%"),
                        "Score":      st.column_config.NumberColumn(format="%.1f"),
                        "Brokerage%": st.column_config.NumberColumn(format="%.2f%%"),
                    },
                )

            # ── Comparison bar chart ─────────────────────────────────────
            st.markdown('<div class="section-title">KPI Comparison</div>', unsafe_allow_html=True)
            kpi_rows = []
            for bname in compare_baskets:
                s = all_stats.get(bname, {})
                kpi_rows.append({
                    "Basket": bname,
                    "Wtd Brokerage %": s.get("weighted_avg_brok", 0),
                    "Wtd 1Y Return %": s.get("weighted_avg_ret1y", 0),
                    "Avg Score": s.get("avg_score", 0),
                })
            kpi_df = pd.DataFrame(kpi_rows)
            kpi_melt = kpi_df.melt(id_vars="Basket", var_name="Metric", value_name="Value")
            fig_kpi = px.bar(
                kpi_melt, x="Value", y="Basket", color="Metric",
                barmode="group", orientation="h",
                color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b"],
                height=max(250, 60 * len(compare_baskets)),
            )
            fig_kpi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title=""), yaxis=dict(title=""),
                margin=dict(l=0, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_kpi, use_container_width=True)

