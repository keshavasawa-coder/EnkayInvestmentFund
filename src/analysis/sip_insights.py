"""
sip_insights.py
Analyzes Business Insight Report to identify client gaps, revenue potential,
and conversion opportunities for Enkay Investments.
"""
import os
import pandas as pd
import numpy as np
import io

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")

DEFAULT_BROKERAGE_RATE = 0.012

AUM_THRESHOLDS = {
    "5 Lakh": 500000,
    "10 Lakh": 1000000,
    "25 Lakh": 2500000,
    "50 Lakh": 5000000,
    "1 Cr": 10000000,
    "2 Cr": 20000000,
    "5 Cr": 50000000,
}


def _build_composite_header(raw_df, header_rows=(1, 2, 3)):
    """
    Build a single header row from a multi-row merged Excel header.
    For each column, pick the longest descriptive label across the header rows.
    Short year-only values like '2025-2026' are deprioritised.
    """
    import re
    year_pattern = re.compile(r'^\d{4}(-\d{4})?$')

    headers = []
    for col_idx in range(len(raw_df.columns)):
        best_label = None
        for row_idx in header_rows:
            if row_idx < len(raw_df):
                val = raw_df.iloc[row_idx, col_idx]
                if pd.notna(val) and str(val).strip() and not str(val).startswith('Unnamed'):
                    candidate = str(val).strip()
                    # Skip pure year labels if we already have a better one
                    if year_pattern.match(candidate) and best_label is not None:
                        continue
                    # Prefer longer (more descriptive) labels
                    if best_label is None or len(candidate) > len(best_label):
                        best_label = candidate
        headers.append(best_label if best_label else f"Column_{col_idx}")

    # Deduplicate: append _2, _3 etc. for any remaining duplicates
    seen = {}
    deduped = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            deduped.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            deduped.append(h)
    return deduped



def load_business_insights(uploaded_file=None):
    """
    Load and clean Business Insight Report from uploaded file or default.
    Handles the 3-row merged header structure (rows 0-3 are headers, data from row 4).
    """
    if uploaded_file is None:
        return None

    file_bytes = uploaded_file.read()
    buf = io.BytesIO(file_bytes)
    xl = pd.ExcelFile(buf)

    raw = None
    for sheet_name in xl.sheet_names:
        try:
            temp = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=None)
            # Look for a sheet that has 'Group' somewhere in the first few rows
            for r in range(min(5, len(temp))):
                if any('Group' in str(v) for v in temp.iloc[r].values if pd.notna(v)):
                    raw = temp
                    break
            if raw is not None:
                break
        except Exception:
            continue

    if raw is None:
        raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, header=None)

    # Build composite header from the multi-row header (rows 1, 2, 3)
    composite_header = _build_composite_header(raw, header_rows=(1, 2, 3))
    # Data starts at row 4
    df = raw.iloc[4:].reset_index(drop=True)
    df.columns = composite_header

    # Drop rows where Group is empty (summary/total rows)
    if 'Group' in df.columns:
        df = df.dropna(subset=['Group'])
        df = df[df['Group'].astype(str).str.strip() != '']

    return df


def load_live_sip(uploaded_file=None):
    """
    Load Live SIP Report from uploaded file.
    """
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        
        df = None
        for sheet_name in xl.sheet_names:
            try:
                temp_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=1)
                if any('Group' in str(c) for c in temp_df.columns):
                    df = temp_df
                    break
            except:
                continue
        
        if df is None:
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, header=1)
    else:
        return None
    
    return df


def standardize_sip_columns(df):
    """Standardize Live SIP Report column names."""
    col_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        
        if 'group' in col_lower:
            col_mapping[col] = 'Group'
        elif 'scheme' in col_lower:
            col_mapping[col] = 'Scheme'
        elif 'amount' in col_lower and '₹' in str(col):
            col_mapping[col] = 'SIP_Amount'
        elif 'monthly sip' in col_lower:
            col_mapping[col] = 'Monthly_SIP_Amount'
        elif 'frequency' in col_lower:
            col_mapping[col] = 'Frequency'
        elif 'start date' in col_lower:
            col_mapping[col] = 'Start_Date'
        elif 'end date' in col_lower:
            col_mapping[col] = 'End_Date'
        elif 'top-up' in col_lower and 'date' in col_lower:
            col_mapping[col] = 'TopUp_Start_Date'
        elif 'registered top-up' in col_lower:
            col_mapping[col] = 'Registered_TopUp_Amt'
        elif 'sip type' in col_lower:
            col_mapping[col] = 'SIP_Type'
        elif 'total' in col_lower and 'installment' in col_lower:
            col_mapping[col] = 'Total_Installments'
    
    df = df.rename(columns=col_mapping)
    return df


def aggregate_sip_by_client(sip_df):
    """Aggregate Live SIP data by client (Group)."""
    if sip_df is None or sip_df.empty:
        return pd.DataFrame()
    
    if 'Group' not in sip_df.columns or 'Monthly_SIP_Amount' not in sip_df.columns:
        return pd.DataFrame()
    
    sip_df['Monthly_SIP_Amount'] = pd.to_numeric(sip_df['Monthly_SIP_Amount'], errors='coerce').fillna(0)
    
    agg_df = sip_df.groupby('Group').agg({
        'Monthly_SIP_Amount': 'sum',
        'Scheme': 'count',
    }).reset_index()
    
    agg_df.columns = ['Group', 'Total_SIP_Amount', 'SIP_Schemes_Count']
    
    return agg_df


def merge_client_data(business_df, sip_df):
    """Merge business insight data with SIP data."""
    if business_df is None or business_df.empty:
        return business_df
    
    # Remove duplicate columns before merge
    business_df = business_df.loc[:, ~business_df.columns.duplicated()]
    
    if sip_df is not None and not sip_df.empty:
        sip_agg = aggregate_sip_by_client(sip_df)
        if not sip_agg.empty:
            # Remove duplicates from sip_agg too
            sip_agg = sip_agg.loc[:, ~sip_agg.columns.duplicated()]
            
            # Check if 'Group' column exists in both
            if 'Group' in business_df.columns and 'Group' in sip_agg.columns:
                business_df = business_df.merge(sip_agg, on='Group', how='left')
                business_df['Total_SIP_Amount'] = business_df['Total_SIP_Amount'].fillna(0)
                business_df['SIP_Schemes_Count'] = business_df['SIP_Schemes_Count'].fillna(0)
    
    return business_df


def standardize_columns(df):
    """Standardize column names for easier access.
    Skips FY-scoped columns (containing '(FY)') to avoid collisions with as-on snapshot columns.
    """
    col_mapping = {}
    mapped_values = set()
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        is_fy_col = '(fy)' in col_lower  # Financial-year scoped column
        
        # Only map to Group if exact match and not already mapped
        if col_lower == 'group' and 'Group' not in mapped_values:
            col_mapping[col] = 'Group'
            mapped_values.add('Group')
        elif 'mobile' in col_lower and 'no' in col_lower and 'Mobile' not in mapped_values:
            col_mapping[col] = 'Mobile'
            mapped_values.add('Mobile')
        elif ('email' in col_lower or ('mail' in col_lower and 'id' in col_lower)) and 'Email' not in mapped_values:
            col_mapping[col] = 'Email'
            mapped_values.add('Email')
        elif ('total mf aum' in col_lower or ('total' in col_lower and 'aum' in col_lower and 'mf' in col_lower)) and 'Total_MF_AUM' not in mapped_values and not is_fy_col:
            col_mapping[col] = 'Total_MF_AUM'
            mapped_values.add('Total_MF_AUM')
        elif 'live sip amount' in col_lower and 'Live_SIP_Amount' not in mapped_values and not is_fy_col:
            col_mapping[col] = 'Live_SIP_Amount'
            mapped_values.add('Live_SIP_Amount')
        elif 'top-up' in col_lower and 'sip' in col_lower and 'amount' in col_lower and 'TopUp_SIP_Amount' not in mapped_values and not is_fy_col:
            col_mapping[col] = 'TopUp_SIP_Amount'
            mapped_values.add('TopUp_SIP_Amount')
        elif 'mf sip' in col_lower and ('closed' in col_lower or 'terminated' in col_lower) and 'SIP_Closed' not in mapped_values:
            col_mapping[col] = 'SIP_Closed'
            mapped_values.add('SIP_Closed')
        elif 'change in 2 years' in col_lower and 'SIP_Change_2Yrs' not in mapped_values:
            col_mapping[col] = 'SIP_Change_2Yrs'
            mapped_values.add('SIP_Change_2Yrs')
        elif 'needs' in col_lower and 'identified' in col_lower and 'Needs_Identified' not in mapped_values:
            col_mapping[col] = 'Needs_Identified'
            mapped_values.add('Needs_Identified')
        elif 'investment mapping' in col_lower and 'Investment_Mapping' not in mapped_values:
            col_mapping[col] = 'Investment_Mapping'
            mapped_values.add('Investment_Mapping')
        elif 'mf gross sales' in col_lower and 'MF_Gross_Sales' not in mapped_values:
            col_mapping[col] = 'MF_Gross_Sales'
            mapped_values.add('MF_Gross_Sales')
        elif 'mf net sales' in col_lower and 'MF_Net_Sales' not in mapped_values:
            col_mapping[col] = 'MF_Net_Sales'
            mapped_values.add('MF_Net_Sales')
        elif 'mf sip aum' in col_lower and 'SIP_AUM' not in mapped_values:
            col_mapping[col] = 'SIP_AUM'
            mapped_values.add('SIP_AUM')
        elif 'no.' in col_lower and 'sip' in col_lower and 'Live_SIP_Count' not in mapped_values:
            col_mapping[col] = 'Live_SIP_Count'
            mapped_values.add('Live_SIP_Count')
    
    df = df.rename(columns=col_mapping)
    
    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df


def calculate_revenue(aum, brokerage_rate=DEFAULT_BROKERAGE_RATE):
    """Calculate estimated annual revenue from AUM."""
    return aum * brokerage_rate


def identify_gaps(df, aum_threshold=1000000):
    """
    Identify all gap types from the business insights data.
    """
    gaps = {
        'high_aum_no_sip': pd.DataFrame(),
        'reduced_sip': pd.DataFrame(),
        'no_topup': pd.DataFrame(),
        'sip_terminated': pd.DataFrame(),
        'below_benchmark': pd.DataFrame(),
    }
    
    if df is None or df.empty:
        return gaps
    
    # Check if required columns exist
    if 'Group' not in df.columns:
        return gaps
    
    # Check for AUM column - use standard name if already present, else fuzzy search
    if 'Total_MF_AUM' not in df.columns:
        aum_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if '(fy)' in col_lower:
                continue
            if 'total' in col_lower and 'aum' in col_lower and 'mf' in col_lower:
                aum_col = col
                break
        if aum_col:
            df = df.rename(columns={aum_col: 'Total_MF_AUM'})
    
    # Check for SIP column - use standard name if already present, else fuzzy search
    if 'Live_SIP_Amount' not in df.columns:
        sip_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if '(fy)' in col_lower:
                continue
            if 'live sip' in col_lower and 'amount' in col_lower:
                sip_col = col
                break
        if sip_col:
            df = df.rename(columns={sip_col: 'Live_SIP_Amount'})
    
    # Now check for required columns after renaming
    if 'Total_MF_AUM' not in df.columns or 'Live_SIP_Amount' not in df.columns:
        return gaps
    
    df = df.copy()
    
    df['Total_MF_AUM'] = pd.to_numeric(df['Total_MF_AUM'], errors='coerce').fillna(0)
    df['Live_SIP_Amount'] = pd.to_numeric(df['Live_SIP_Amount'], errors='coerce').fillna(0)
    
    gaps['high_aum_no_sip'] = df[
        (df['Total_MF_AUM'] >= aum_threshold) & 
        (df['Live_SIP_Amount'] == 0)
    ].copy()
    
    if 'SIP_Change_2Yrs' in df.columns:
        df['SIP_Change_2Yrs'] = pd.to_numeric(df['SIP_Change_2Yrs'], errors='coerce').fillna(0)
        gaps['reduced_sip'] = df[
            (df['Live_SIP_Amount'] > 0) & 
            (df['SIP_Change_2Yrs'] < 0)
        ].copy()
    
    if 'TopUp_SIP_Amount' in df.columns:
        df['TopUp_SIP_Amount'] = pd.to_numeric(df['TopUp_SIP_Amount'], errors='coerce').fillna(0)
        gaps['no_topup'] = df[
            (df['Live_SIP_Amount'] > 0) & 
            (df['TopUp_SIP_Amount'] == 0)
        ].copy()
    
    if 'SIP_Closed' in df.columns:
        df['SIP_Closed'] = pd.to_numeric(df['SIP_Closed'], errors='coerce').fillna(0)
        gaps['sip_terminated'] = df[df['SIP_Closed'] > 0].copy()
    
    mask_benchmark = (df['Live_SIP_Amount'] > 0) & (df['Total_MF_AUM'] > 0)
    df['AUM_SIP_Ratio'] = np.where(mask_benchmark, 
                                    df['Total_MF_AUM'] / df['Live_SIP_Amount'], 
                                    np.nan)
    gaps['below_benchmark'] = df[
        (df['Live_SIP_Amount'] > 0) & 
        (df['AUM_SIP_Ratio'] > 1.5 * 12)
    ].copy()
    
    return gaps


def calculate_pareto(df):
    """
    Calculate Pareto analysis - top 20% clients contribution.
    """
    if df is None or df.empty or 'Total_MF_AUM' not in df.columns:
        return {}
    
    df = df.copy()
    df['Total_MF_AUM'] = pd.to_numeric(df['Total_MF_AUM'], errors='coerce').fillna(0)
    df = df[df['Total_MF_AUM'] > 0].sort_values('Total_MF_AUM', ascending=False)
    
    total_aum = df['Total_MF_AUM'].sum()
    total_clients = len(df)
    
    if total_clients == 0 or total_aum == 0:
        return {}
    
    df['cumsum'] = df['Total_MF_AUM'].cumsum()
    df['cum_pct'] = df['cumsum'] / total_aum
    
    top_20_pct = int(np.ceil(total_clients * 0.2))
    top_20_clients = df.head(top_20_pct)
    top_20_aum = top_20_clients['Total_MF_AUM'].sum()
    
    return {
        'total_aum': total_aum,
        'total_clients': total_clients,
        'top_20_pct_clients': top_20_pct,
        'top_20_aum': top_20_aum,
        'top_20_aum_pct': (top_20_aum / total_aum * 100) if total_aum > 0 else 0,
        'full_data': df,
    }


def get_client_tiers(df):
    """
    Assign client tiers based on AUM: Platinum (top 5%), Gold (5-20%), Silver (rest)
    """
    if df is None or df.empty or 'Total_MF_AUM' not in df.columns:
        return df
    
    df = df.copy()
    df['Total_MF_AUM'] = pd.to_numeric(df['Total_MF_AUM'], errors='coerce').fillna(0)
    
    df_sorted = df[df['Total_MF_AUM'] > 0].sort_values('Total_MF_AUM', ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    
    platinum_threshold = int(np.ceil(n * 0.05))
    gold_threshold = int(np.ceil(n * 0.20))
    
    tiers = []
    for i, row in df_sorted.iterrows():
        if i < platinum_threshold:
            tiers.append('Platinum')
        elif i < gold_threshold:
            tiers.append('Gold')
        else:
            tiers.append('Silver')
    
    df_sorted['Client_Tier'] = tiers
    return df_sorted


def calculate_sip_lumpsum_ratio(df):
    """
    Calculate SIP vs Lumpsum ratio for each client.
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()

    if 'Total_MF_AUM' not in df.columns:
        return df

    df['Total_MF_AUM'] = pd.to_numeric(df['Total_MF_AUM'], errors='coerce').fillna(0)
    df['SIP_AUM'] = pd.to_numeric(df.get('SIP_AUM', 0), errors='coerce').fillna(0)
    
    df['Lumpsum_AUM'] = df['Total_MF_AUM'] - df['SIP_AUM']
    
    mask = df['Total_MF_AUM'] > 0
    df['SIP_Pct'] = np.where(mask, (df['SIP_AUM'] / df['Total_MF_AUM'] * 100), 0)
    df['Lumpsum_Pct'] = np.where(mask, (df['Lumpsum_AUM'] / df['Total_MF_AUM'] * 100), 0)
    
    return df


def identify_conversion_opportunities(df):
    """
    Identify clients with needs identified but no investment mapping.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    if 'Needs_Identified' not in df.columns or 'Investment_Mapping' not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['Needs_Identified'] = pd.to_numeric(df['Needs_Identified'], errors='coerce').fillna(0)
    df['Investment_Mapping'] = pd.to_numeric(df['Investment_Mapping'], errors='coerce').fillna(0)
    
    conversion_df = df[
        (df['Needs_Identified'] > 0) & 
        (df['Investment_Mapping'] == 0)
    ].copy()
    
    return conversion_df


def calculate_revenue_potential(df, brokerage_rate=DEFAULT_BROKERAGE_RATE):
    """
    Calculate estimated annual revenue from each client.
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()

    if 'Total_MF_AUM' not in df.columns:
        df['Est_Annual_Revenue'] = 0
        return df

    df['Total_MF_AUM'] = pd.to_numeric(df['Total_MF_AUM'], errors='coerce').fillna(0)
    df['Est_Annual_Revenue'] = df['Total_MF_AUM'] * brokerage_rate
    
    return df


def get_summary_metrics(df, gaps):
    """
    Calculate summary metrics for the dashboard.
    """
    metrics = {
        'total_aum': 0,
        'live_sip_amount': 0,
        'est_annual_revenue': 0,
        'total_clients': 0,
        'high_aum_no_sip_count': 0,
        'reduced_sip_count': 0,
        'no_topup_count': 0,
        'sip_terminated_count': 0,
        'below_benchmark_count': 0,
    }
    
    if df is None or df.empty:
        return metrics
    
    # Try to find and rename AUM column (skip if already standardized)
    if 'Total_MF_AUM' not in df.columns:
        aum_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if '(fy)' in col_lower:
                continue
            if 'total' in col_lower and 'aum' in col_lower and 'mf' in col_lower:
                aum_col = col
                break
        if aum_col:
            df = df.rename(columns={aum_col: 'Total_MF_AUM'})
    
    # Try to find and rename SIP column (skip if already standardized)
    if 'Live_SIP_Amount' not in df.columns:
        sip_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if '(fy)' in col_lower:
                continue
            if 'live sip' in col_lower and 'amount' in col_lower:
                sip_col = col
                break
        if sip_col:
            df = df.rename(columns={sip_col: 'Live_SIP_Amount'})
    
    if 'Total_MF_AUM' not in df.columns:
        return metrics
    
    df['Total_MF_AUM'] = pd.to_numeric(df['Total_MF_AUM'], errors='coerce').fillna(0)
    df['Live_SIP_Amount'] = pd.to_numeric(df.get('Live_SIP_Amount', 0), errors='coerce').fillna(0)
    
    metrics['total_aum'] = df['Total_MF_AUM'].sum()
    metrics['live_sip_amount'] = df['Live_SIP_Amount'].sum()
    metrics['est_annual_revenue'] = metrics['total_aum'] * DEFAULT_BROKERAGE_RATE
    metrics['total_clients'] = len(df)
    metrics['high_aum_no_sip_count'] = len(gaps.get('high_aum_no_sip', pd.DataFrame()))
    metrics['reduced_sip_count'] = len(gaps.get('reduced_sip', pd.DataFrame()))
    metrics['no_topup_count'] = len(gaps.get('no_topup', pd.DataFrame()))
    metrics['sip_terminated_count'] = len(gaps.get('sip_terminated', pd.DataFrame()))
    metrics['below_benchmark_count'] = len(gaps.get('below_benchmark', pd.DataFrame()))
    
    return metrics


def format_mobile_number(mobile):
    """Format mobile number for privacy - show first 2 and last 4 digits."""
    if pd.isna(mobile) or not mobile:
        return "—"
    mobile_str = str(mobile).strip()
    digits = ''.join(c for c in mobile_str if c.isdigit())
    if len(digits) >= 6:
        return digits[:2] + "****" + digits[-4:]
    elif len(digits) >= 3:
        return digits[:2] + "****"
    return "****"


def get_client_list_for_gap(gap_type, gaps, df):
    """
    Get formatted client list for a specific gap type with contact info.
    """
    if gap_type == 'high_aum_no_sip':
        clients = gaps.get('high_aum_no_sip', pd.DataFrame())
    elif gap_type == 'reduced_sip':
        clients = gaps.get('reduced_sip', pd.DataFrame())
    elif gap_type == 'no_topup':
        clients = gaps.get('no_topup', pd.DataFrame())
    elif gap_type == 'sip_terminated':
        clients = gaps.get('sip_terminated', pd.DataFrame())
    elif gap_type == 'below_benchmark':
        clients = gaps.get('below_benchmark', pd.DataFrame())
    else:
        return pd.DataFrame()
    
    if clients.empty:
        return pd.DataFrame()
    
    display_cols = ['Group']
    for col in ['Mobile', 'Email', 'Total_MF_AUM', 'Live_SIP_Amount', 'TopUp_SIP_Amount', 'SIP_Change_2Yrs']:
        if col in clients.columns:
            display_cols.append(col)
    
    result = clients[display_cols].copy()
    
    if 'Mobile' in result.columns:
        result['Mobile_Display'] = result['Mobile'].apply(format_mobile_number)
        result = result.drop(columns=['Mobile'])
    
    if 'Total_MF_AUM' in result.columns:
        result['Total_MF_AUM_Lakh'] = (result['Total_MF_AUM'] / 100000).round(1)
    
    if 'Live_SIP_Amount' in result.columns:
        result['Live_SIP_Amount_K'] = (result['Live_SIP_Amount'] / 1000).round(0)
    
    return result


if __name__ == "__main__":
    print("Business Insights Analysis Module")
    print("Upload BusinessInsightReport to use this module")
