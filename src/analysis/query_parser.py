"""
query_parser.py
Rule-based natural language parser for dashboard Q&A (non-AI).
"""
import re
from typing import Dict, List, Optional

from rapidfuzz import process, fuzz


CATEGORY_SYNONYMS = {
    "equity": ["equity", "equity fund", "equity funds"],
    "debt": ["debt", "debt fund", "debt funds", "liquid", "income"],
    "hybrid": ["hybrid", "balanced", "asset allocation"],
    "solution oriented": ["solution", "solution oriented", "retirement", "children"],
    "other": ["other", "index", "fof", "fund of funds", "etf"],
}

SUBCATEGORY_KEYWORDS = {
    "mid cap": ["midcap", "mid cap"],
    "small cap": ["smallcap", "small cap"],
    "large cap": ["largecap", "large cap"],
    "flexi cap": ["flexicap", "flexi cap"],
    "multi cap": ["multicap", "multi cap"],
    "aggressive hybrid": ["aggressive hybrid"],
    "balanced advantage": ["balanced advantage"],
    "liquid": ["liquid"],
    "arbitrage": ["arbitrage"],
    "elss": ["elss", "tax saver", "tax saving"],
}

METRIC_KEYWORDS = {
    "brokerage": ["brokerage", "commission", "trail"],
    "aum": ["aum", "assets", "assets under management"],
    "score": ["score", "best", "top", "rank"],
    "return_1y_regular": ["1y", "1 year", "one year"],
    "return_3y_regular": ["3y", "3 year", "three year"],
    "return_5y_regular": ["5y", "5 year", "five year"],
    "sip": ["sip", "monthly sip"],
    "client_aum": ["client aum", "clients by aum", "highest aum"],
}

ALT_KEYWORDS = ["alternative", "replace", "instead of", "shift from", "switch from"]
CLIENT_KEYWORDS = ["client", "clients", "investor", "investors", "sip"]


def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_top_n(text: str, default_value: int = 10) -> int:
    m = re.search(r"\btop\s+(\d{1,3})\b", text)
    if m:
        return max(1, min(200, int(m.group(1))))
    m = re.search(r"\b(\d{1,3})\s+(funds?|clients?|schemes?)\b", text)
    if m:
        return max(1, min(200, int(m.group(1))))
    if "top" in text or "list" in text:
        return default_value
    return default_value


def _has_explicit_limit(text: str) -> bool:
    if re.search(r"\btop\s+\d{1,3}\b", text):
        return True
    if re.search(r"\b\d{1,3}\s+(funds?|clients?|schemes?)\b", text):
        return True
    return False


def _detect_metric(text: str) -> str:
    # Prioritize explicit SIP/AUM client intents before generic words like "top".
    if "live sip" in text or "monthly sip" in text or "sip" in text:
        return "sip"
    if "client aum" in text or "clients by aum" in text or "highest aum" in text:
        return "client_aum"

    for metric, kws in METRIC_KEYWORDS.items():
        if any(kw in text for kw in kws):
            return metric
    if "highest" in text and "broker" in text:
        return "brokerage"
    if "best" in text:
        return "score"
    return "score"


def _detect_sort_order(text: str) -> str:
    low_markers = ["lowest", "least", "minimum", "min", "bottom", "worst"]
    high_markers = ["highest", "top", "maximum", "max", "best"]
    if any(m in text for m in low_markers):
        return "asc"
    if any(m in text for m in high_markers):
        return "desc"
    return "desc"


def _extract_category_terms(text: str) -> Dict[str, List[str]]:
    matched_categories: List[str] = []
    matched_subcats: List[str] = []

    for category, kws in CATEGORY_SYNONYMS.items():
        if any(kw in text for kw in kws):
            matched_categories.append(category)

    for subcat, kws in SUBCATEGORY_KEYWORDS.items():
        if any(kw in text for kw in kws):
            matched_subcats.append(subcat)

    return {
        "categories": sorted(set(matched_categories)),
        "sub_category_terms": sorted(set(matched_subcats)),
    }


def _extract_scheme_hint(text: str) -> Optional[str]:
    patterns = [
        r"alternative\s+to\s+(.+)$",
        r"instead\s+of\s+(.+)$",
        r"replace\s+(.+)$",
        r"switch\s+from\s+(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            hint = m.group(1).strip(" ?,.\"")
            if hint:
                return hint
    return None


def _match_scheme_name(query_text: str, scheme_names: List[str], scheme_hint: Optional[str]) -> Dict[str, Optional[object]]:
    if not scheme_names:
        return {"scheme_name": None, "scheme_match_score": None}

    lowered_map = {name.lower(): name for name in scheme_names if isinstance(name, str)}

    if scheme_hint:
        hint_clean = scheme_hint.lower().strip()
        if hint_clean in lowered_map:
            return {"scheme_name": lowered_map[hint_clean], "scheme_match_score": 100}

        matched = process.extractOne(
            hint_clean,
            list(lowered_map.keys()),
            scorer=fuzz.token_sort_ratio,
        )
        if matched:
            best_name_lc, score, _ = matched
            if score >= 70:
                return {"scheme_name": lowered_map[best_name_lc], "scheme_match_score": float(score)}

    # fallback: direct contains on full query
    for low, original in lowered_map.items():
        if low in query_text:
            return {"scheme_name": original, "scheme_match_score": 95}

    return {"scheme_name": None, "scheme_match_score": None}


def parse_query(user_input: str, scheme_names: Optional[List[str]] = None) -> Dict[str, object]:
    """
    Parse a natural-language query into deterministic intent metadata.
    """
    text = _normalize(user_input)
    top_n = _extract_top_n(text)
    metric = _detect_metric(text)
    cat_terms = _extract_category_terms(text)
    scheme_hint = _extract_scheme_hint(text)

    is_client = any(k in text for k in CLIENT_KEYWORDS)
    is_alternative = any(k in text for k in ALT_KEYWORDS)

    if is_client:
        intent = "client_rank"
        entity = "client"
    elif is_alternative:
        intent = "fund_alternative"
        entity = "fund"
    else:
        intent = "fund_rank"
        entity = "fund"

    if intent == "fund_alternative" and not _has_explicit_limit(text):
        top_n = 3

    sort_order = _detect_sort_order(text)
    if sort_order == "asc":
        action = "bottom"
    elif "highest" in text or "top" in text or "best" in text:
        action = "top"
    else:
        action = "list"

    scheme_match = _match_scheme_name(text, scheme_names or [], scheme_hint)

    warnings: List[str] = []
    if intent == "fund_alternative" and not scheme_match["scheme_name"]:
        warnings.append("I could not confidently identify the reference fund for alternatives.")

    if entity == "client" and metric not in {"sip", "client_aum"}:
        metric = "client_aum"

    parsed = {
        "raw_query": user_input,
        "normalized_query": text,
        "intent": intent,
        "entity": entity,
        "action": action,
        "metric": metric,
        "sort_order": sort_order,
        "top_n": top_n,
        "categories": cat_terms["categories"],
        "sub_category_terms": cat_terms["sub_category_terms"],
        "scheme_hint": scheme_hint,
        "scheme_name": scheme_match["scheme_name"],
        "scheme_match_score": scheme_match["scheme_match_score"],
        "warnings": warnings,
    }
    return parsed
