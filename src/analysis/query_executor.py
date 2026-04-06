"""
query_executor.py
Deterministic executor for parsed Q&A intents.
"""
from typing import Dict, List

import pandas as pd


METRIC_COLUMN_MAP = {
    "brokerage": "trail_brokerage_incl_gst",
    "aum": "aum_cr",
    "score": "composite_score",
    "return_1y_regular": "return_1y_regular",
    "return_3y_regular": "return_3y_regular",
    "return_5y_regular": "return_5y_regular",
}


def _apply_category_filters(df: pd.DataFrame, parsed_query: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()

    category_terms = parsed_query.get("categories", []) or []
    sub_terms = parsed_query.get("sub_category_terms", []) or []

    if category_terms and "category" in out.columns:
        # Match category text loosely in source category labels.
        mask = pd.Series(False, index=out.index)
        for t in category_terms:
            mask = mask | out["category"].fillna("").str.lower().str.contains(str(t), na=False)
        out = out[mask]

    if sub_terms and "sub_category" in out.columns:
        mask = pd.Series(False, index=out.index)
        for t in sub_terms:
            mask = mask | out["sub_category"].fillna("").str.lower().str.contains(str(t), na=False)
        out = out[mask]

    return out


def _canonical_client_df(ci_business_df: pd.DataFrame, ci_sip_df: pd.DataFrame) -> pd.DataFrame:
    if ci_business_df is None or ci_business_df.empty:
        return pd.DataFrame()

    base = ci_business_df.copy()

    group_col = "Group" if "Group" in base.columns else None
    if group_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["Client"] = base[group_col].astype(str)

    if "Total_MF_AUM" in base.columns:
        out["Total_MF_AUM"] = pd.to_numeric(base["Total_MF_AUM"], errors="coerce")
    else:
        out["Total_MF_AUM"] = 0.0

    if "Live_SIP_Amount" in base.columns:
        out["Live_SIP_Amount"] = pd.to_numeric(base["Live_SIP_Amount"], errors="coerce")
    else:
        out["Live_SIP_Amount"] = 0.0

    if "Live_SIP_Count" in base.columns:
        out["Live_SIP_Count"] = pd.to_numeric(base["Live_SIP_Count"], errors="coerce")
    else:
        out["Live_SIP_Count"] = 0.0

    # If SIP detail dataset is available, enrich amount by grouped totals when missing/zero.
    if ci_sip_df is not None and not ci_sip_df.empty and "Group" in ci_sip_df.columns:
        sip = ci_sip_df.copy()
        amount_col = None
        for c in ["Monthly_SIP_Amount", "SIP_Amount"]:
            if c in sip.columns:
                amount_col = c
                break
        if amount_col is not None:
            sip[amount_col] = pd.to_numeric(sip[amount_col], errors="coerce").fillna(0)
            grouped = sip.groupby("Group", as_index=False)[amount_col].sum()
            grouped.columns = ["Client", "Total_SIP_Amount_From_SIP_Report"]
            out = out.merge(grouped, on="Client", how="left")
            out["Total_SIP_Amount_From_SIP_Report"] = out["Total_SIP_Amount_From_SIP_Report"].fillna(0)
            out["Live_SIP_Amount"] = out[["Live_SIP_Amount", "Total_SIP_Amount_From_SIP_Report"]].max(axis=1)

    out = out.fillna(0)
    return out


def _execute_fund_alternative(parsed_query: Dict[str, object], context: Dict[str, object]) -> Dict[str, object]:
    active_df = context.get("active_df")
    top_n = int(parsed_query.get("top_n", 3) or 3)
    scheme_name = parsed_query.get("scheme_name")

    if active_df is None or active_df.empty:
        return {
            "ok": False,
            "title": "No fund data available",
            "result_df": pd.DataFrame(),
            "message": "Fund data is not loaded.",
            "warnings": ["Active fund dataframe is empty."],
            "applied_filters": [],
            "highlight_top": False,
        }

    if not scheme_name:
        return {
            "ok": False,
            "title": "Reference fund required",
            "result_df": pd.DataFrame(),
            "message": "Please include a reference fund name (for example: alternative to Kotak Midcap Fund).",
            "warnings": ["No scheme identified for alternative query."],
            "applied_filters": [],
            "highlight_top": False,
        }

    selected_rows = active_df[active_df["scheme_name"] == scheme_name]
    if selected_rows.empty:
        return {
            "ok": False,
            "title": "Reference fund not found",
            "result_df": pd.DataFrame(),
            "message": f"Could not find '{scheme_name}' in current fund universe.",
            "warnings": ["Scheme match failed in active data."],
            "applied_filters": [],
            "highlight_top": False,
        }

    selected = selected_rows.iloc[0]
    sub_cat = selected.get("sub_category")
    category = selected.get("category")

    peers = active_df[active_df["scheme_name"] != scheme_name].copy()
    if pd.notna(sub_cat):
        peers = peers[peers["sub_category"] == sub_cat]
        peer_scope = "same sub-category"
    elif pd.notna(category):
        peers = peers[peers["category"] == category]
        peer_scope = "same category"
    else:
        peer_scope = "full universe"

    if peers.empty:
        return {
            "ok": True,
            "title": f"Alternatives for {scheme_name}",
            "result_df": pd.DataFrame(),
            "message": "No peers found for this fund in the current universe.",
            "warnings": [],
            "applied_filters": [peer_scope, "exclude selected fund"],
            "highlight_top": False,
        }

    if "rank" in peers.columns:
        peers = peers.sort_values(["rank", "composite_score"], ascending=[True, False])
    elif "composite_score" in peers.columns:
        peers = peers.sort_values("composite_score", ascending=False)

    peers = peers.head(top_n).copy()

    sel_brok = selected.get("trail_brokerage_incl_gst")
    sel_r1 = selected.get("return_1y_regular")
    sel_r3 = selected.get("return_3y_regular")
    sel_score = selected.get("composite_score")

    if "trail_brokerage_incl_gst" in peers.columns:
        peers["delta_brokerage"] = (peers["trail_brokerage_incl_gst"] - sel_brok).round(3)
    if "return_1y_regular" in peers.columns:
        peers["delta_return_1y"] = (peers["return_1y_regular"] - sel_r1).round(2)
    if "return_3y_regular" in peers.columns:
        peers["delta_return_3y"] = (peers["return_3y_regular"] - sel_r3).round(2)
    if "composite_score" in peers.columns:
        peers["delta_composite"] = (peers["composite_score"] - sel_score).round(2)

    display_cols = [
        "scheme_name",
        "sub_category",
        "trail_brokerage_incl_gst",
        "return_1y_regular",
        "return_3y_regular",
        "return_5y_regular",
        "composite_score",
        "rank",
        "delta_brokerage",
        "delta_return_1y",
        "delta_return_3y",
        "delta_composite",
    ]
    display_cols = [c for c in display_cols if c in peers.columns]

    rename_map = {
        "scheme_name": "Scheme",
        "sub_category": "Sub-Category",
        "trail_brokerage_incl_gst": "Brokerage%",
        "return_1y_regular": "1Y Ret%",
        "return_3y_regular": "3Y Ret%",
        "return_5y_regular": "5Y Ret%",
        "composite_score": "Score",
        "rank": "Rank",
        "delta_brokerage": "Brokerage Delta",
        "delta_return_1y": "1Y Ret Delta",
        "delta_return_3y": "3Y Ret Delta",
        "delta_composite": "Score Delta",
    }

    out_df = peers[display_cols].rename(columns=rename_map).reset_index(drop=True)

    return {
        "ok": True,
        "title": f"Best alternatives to {scheme_name}",
        "result_df": out_df,
        "message": "Showing best peer funds in the same sub-category based on rank and score.",
        "warnings": parsed_query.get("warnings", []),
        "applied_filters": [peer_scope, "exclude selected fund", "sorted by rank (best first)", f"top {top_n}"],
        "highlight_top": True,
    }


def _execute_fund_rank(parsed_query: Dict[str, object], context: Dict[str, object]) -> Dict[str, object]:
    active_df = context.get("active_df")
    if active_df is None or active_df.empty:
        return {
            "ok": False,
            "title": "No fund data available",
            "result_df": pd.DataFrame(),
            "message": "Fund data is not loaded.",
            "warnings": ["Active fund dataframe is empty."],
            "applied_filters": [],
            "highlight_top": False,
        }

    metric = parsed_query.get("metric", "score")
    metric_col = METRIC_COLUMN_MAP.get(metric, "composite_score")
    sort_order = parsed_query.get("sort_order", "desc")
    ascending = sort_order == "asc"
    top_n = int(parsed_query.get("top_n", 10) or 10)

    fdf = _apply_category_filters(active_df, parsed_query)

    # For score queries, rank is authoritative. For others, use metric sort order.
    if metric == "score" and "rank" in fdf.columns:
        if ascending:
            fdf = fdf.sort_values(["rank", "composite_score"], ascending=[False, True])
        else:
            fdf = fdf.sort_values(["rank", "composite_score"], ascending=[True, False])
    else:
        if metric_col in fdf.columns:
            fdf = fdf.sort_values(metric_col, ascending=ascending)

    if top_n > 0:
        fdf = fdf.head(top_n)

    show_cols = [
        "scheme_name",
        "category",
        "sub_category",
        "trail_brokerage_incl_gst",
        "aum_cr",
        "return_1y_regular",
        "return_3y_regular",
        "return_5y_regular",
        "composite_score",
        "rank",
    ]
    show_cols = [c for c in show_cols if c in fdf.columns]
    out_df = fdf[show_cols].rename(columns={
        "scheme_name": "Scheme",
        "category": "Category",
        "sub_category": "Sub-Category",
        "trail_brokerage_incl_gst": "Brokerage%",
        "aum_cr": "AUM (Cr)",
        "return_1y_regular": "1Y Ret%",
        "return_3y_regular": "3Y Ret%",
        "return_5y_regular": "5Y Ret%",
        "composite_score": "Score",
        "rank": "Rank",
    }).reset_index(drop=True)

    filters: List[str] = []
    if parsed_query.get("categories"):
        filters.append("category filter")
    if parsed_query.get("sub_category_terms"):
        filters.append("sub-category filter")
    filters.append(f"sorted by {metric_col} ({'ascending' if ascending else 'descending'})")
    filters.append(f"top {top_n}")

    return {
        "ok": True,
        "title": "Fund query results",
        "result_df": out_df,
        "message": "Showing ranked funds based on detected filters and metric.",
        "warnings": parsed_query.get("warnings", []),
        "applied_filters": filters,
        "highlight_top": True,
    }


def _execute_client_rank(parsed_query: Dict[str, object], context: Dict[str, object]) -> Dict[str, object]:
    ci_business_df = context.get("ci_business_df")
    ci_sip_df = context.get("ci_sip_df")
    top_n = int(parsed_query.get("top_n", 10) or 10)

    cdf = _canonical_client_df(ci_business_df, ci_sip_df)
    if cdf.empty:
        return {
            "ok": False,
            "title": "Client data not available",
            "result_df": pd.DataFrame(),
            "message": "Please upload and process Client Insights files first.",
            "warnings": ["Missing or unprocessed client-level dataset in this session."],
            "applied_filters": [],
            "highlight_top": False,
        }

    metric = parsed_query.get("metric", "client_aum")
    if metric == "sip":
        sort_col = "Live_SIP_Amount"
    else:
        sort_col = "Total_MF_AUM"

    cdf = cdf.sort_values(sort_col, ascending=False).head(top_n)
    out_df = cdf[["Client", "Total_MF_AUM", "Live_SIP_Amount", "Live_SIP_Count"]].copy()
    out_df = out_df.rename(columns={
        "Total_MF_AUM": "AUM (Rs)",
        "Live_SIP_Amount": "Live SIP (Rs)",
        "Live_SIP_Count": "Live SIP Count",
    }).reset_index(drop=True)

    title = "Top clients by AUM" if sort_col == "Total_MF_AUM" else "Top clients by SIP"
    return {
        "ok": True,
        "title": title,
        "result_df": out_df,
        "message": "Showing client ranking from currently uploaded Client Insights data.",
        "warnings": parsed_query.get("warnings", []),
        "applied_filters": [f"sorted by {sort_col}", f"top {top_n}"],
        "highlight_top": True,
    }


def execute_query(parsed_query: Dict[str, object], context: Dict[str, object]) -> Dict[str, object]:
    """
    Execute parsed intent against in-memory app dataframes.
    """
    intent = parsed_query.get("intent")

    if intent == "fund_alternative":
        return _execute_fund_alternative(parsed_query, context)
    if intent == "client_rank":
        return _execute_client_rank(parsed_query, context)
    return _execute_fund_rank(parsed_query, context)
