"""
02_load_nav_returns.py
Fetches historical NAV data from the MFAPI.in API for each scheme in scheme_master.csv.
Computes 1Y, 3Y, 5Y CAGR returns from NAV history.
Outputs: data/processed/scheme_performance.csv
"""
import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..", "..")
MASTER_CSV = os.path.join(BASE_DIR, "data", "processed", "scheme_master.csv")
CACHE_DIR  = os.path.join(BASE_DIR, "data", "api_cache")
OUT_PATH   = os.path.join(BASE_DIR, "data", "processed", "scheme_performance.csv")

API_BASE   = "https://api.mfapi.in/mf"
REQUEST_DELAY = 0.1  # seconds between API calls (be respectful to the free API)


def fetch_nav_history(scheme_code: int, cache_dir: str) -> list:
    """Fetch full NAV history for a scheme. Uses local cache if available."""
    cache_file = os.path.join(cache_dir, f"{scheme_code}.json")

    # Check cache (valid for 24 hours)
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if (time.time() - mtime) < 86400:  # 24 hours
            with open(cache_file, "r") as f:
                data = json.load(f)
            return data.get("data", [])

    # Fetch from API
    url = f"{API_BASE}/{scheme_code}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Cache the response
        with open(cache_file, "w") as f:
            json.dump(data, f)

        time.sleep(REQUEST_DELAY)
        return data.get("data", [])
    except Exception as e:
        print(f"    [ERROR] Failed to fetch scheme {scheme_code}: {e}")
        return []


def compute_cagr(nav_today: float, nav_past: float, years: float) -> float:
    """Compute Compound Annual Growth Rate (CAGR) as a percentage."""
    if nav_past <= 0 or nav_today <= 0 or years <= 0:
        return np.nan
    return ((nav_today / nav_past) ** (1.0 / years) - 1.0) * 100.0


def find_nav_on_date(nav_data: list, target_date: datetime, tolerance_days: int = 7) -> float:
    """Find NAV closest to a target date within a tolerance window.
    nav_data is sorted newest-first with format [{"date": "DD-MM-YYYY", "nav": "123.456"}, ...]
    """
    target_str = target_date.strftime("%d-%m-%Y")

    for entry in nav_data:
        try:
            entry_date = datetime.strptime(entry["date"], "%d-%m-%Y")
        except (ValueError, KeyError):
            continue

        diff = abs((entry_date - target_date).days)
        if diff <= tolerance_days:
            try:
                return float(entry["nav"])
            except (ValueError, TypeError):
                continue

    return np.nan


def process_scheme(scheme_code: int, nav_data: list) -> dict:
    """Compute returns from NAV history for a single scheme."""
    if not nav_data:
        return {
            "nav_regular": np.nan,
            "return_1y_regular": np.nan,
            "return_3y_regular": np.nan,
            "return_5y_regular": np.nan,
        }

    # Latest NAV
    try:
        nav_today = float(nav_data[0]["nav"])
        latest_date = datetime.strptime(nav_data[0]["date"], "%d-%m-%Y")
    except (ValueError, TypeError, KeyError, IndexError):
        return {
            "nav_regular": np.nan,
            "return_1y_regular": np.nan,
            "return_3y_regular": np.nan,
            "return_5y_regular": np.nan,
        }

    # Compute returns
    results = {"nav_regular": nav_today}

    periods = {
        "return_1y_regular": (1, 365),
        "return_3y_regular": (3, 365 * 3),
        "return_5y_regular": (5, 365 * 5),
    }

    for col, (years, days_back) in periods.items():
        target_date = latest_date - timedelta(days=days_back)
        nav_past = find_nav_on_date(nav_data, target_date, tolerance_days=7)
        results[col] = compute_cagr(nav_today, nav_past, years)

    return results


def main():
    print(f"Loading scheme master: {MASTER_CSV}")
    master = pd.read_csv(MASTER_CSV)
    print(f"  Schemes to process: {len(master)}")

    os.makedirs(CACHE_DIR, exist_ok=True)

    results = []
    total = len(master)

    for i, row in master.iterrows():
        scheme_code = int(row["scheme_code"])
        scheme_name = row["scheme_name"]

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing {i + 1}/{total}: {scheme_name[:50]}...")

        nav_data = fetch_nav_history(scheme_code, CACHE_DIR)
        metrics = process_scheme(scheme_code, nav_data)

        results.append({
            "scheme_code": scheme_code,
            "scheme_name": scheme_name,
            "category": row["category"],
            "sub_category": row["sub_category"],
            **metrics,
        })

    df = pd.DataFrame(results)

    # Summary stats
    valid_1y = df["return_1y_regular"].notna().sum()
    valid_3y = df["return_3y_regular"].notna().sum()
    valid_5y = df["return_5y_regular"].notna().sum()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\n[OK] scheme_performance.csv saved -> {OUT_PATH}")
    print(f"   Total schemes: {len(df)}")
    print(f"   With 1Y returns: {valid_1y}")
    print(f"   With 3Y returns: {valid_3y}")
    print(f"   With 5Y returns: {valid_5y}")
    print(f"   Categories: {df['category'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
