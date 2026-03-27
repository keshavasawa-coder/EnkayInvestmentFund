# Enkay Investments - Fund Recommendation Analytics

A data-driven fund recommendation and SIP growth analytics system for mutual fund distribution firms.

## Features

### Dashboard Tabs (9 Total)

1. **Fund Ranker** - Score and rank funds based on performance, brokerage, AUM, and tie-up status with customizable weights
2. **Peer Comparison** - Compare funds within the same category using radar charts
3. **Portfolio Exposure Review** - Analyze current holdings and flag underperforming schemes with better alternatives
4. **Fund Shift Advisor** - Find better-paying alternatives when brokerage changes
5. **AMC Concentration** - View AMC distribution in your current portfolio
6. **Brokerage vs Performance** - Visualize the trade-off between returns and commission
7. **Recommended Portfolios** - Pre-built model portfolios (Conservative, Moderate, Aggressive, SIP, Tax Saving, etc.)
8. **Client Insights** - Client gap analysis, Pareto analysis, revenue potential, and conversion opportunities
9. **Upload AUM Data** - Upload latest AUM data from AMFI

### Key Features

- **Customizable Scoring Weights** - Adjust weights for Return, Alpha, Brokerage, AUM, and Tie-Up
- **Risk Profiles** - Conservative, Moderate, and Aggressive scoring configurations
- **AUM File Upload** - Upload custom Scheme_wise_AUM reports
- **SIP & Business Insight Upload** - Upload Live_SIP_Report and BusinessInsightReport for client analysis
- **Login Authentication** - Secured access with username/password
- **Light Theme** - Clean, modern UI design

## Authentication

- **Username**: `admin`
- **Password**: `enkayinv123`

## Tech Stack

- Python 3.11+
- Streamlit (Dashboard)
- Pandas, NumPy (Data Processing)
- Plotly (Visualizations)
- Scikit-learn (Scoring)
- RapidFuzz (Fuzzy Matching)

## Installation

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
streamlit run src/dashboard/app.py
```

## Data Files

### Required Files (in project root)
- `Brokerage_Rates_Enkay.xlsx` - Brokerage rates for mutual fund schemes
- `TieUp_AMCs_List.xlsx` - List of AMCs with tie-up categories (A/B)

### Optional Files (for Portfolio Exposure Review)
- `Scheme_wise_AUM.xls` - Current AUM holdings

### Optional Files (for Client Insights)
- `Live_SIP_Report.xls` - Live SIP transaction details
- `BusinessInsightReport-FinancialYearWise.xls` - Client business insights

## Deployment

### Streamlit Community Cloud (Recommended)
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repository
4. Deploy for free!

**Configuration:**
- Main file path: `src/dashboard/app.py`
- Python version: 3.11

### Hugging Face Spaces
1. Go to https://huggingface.co/spaces
2. Create a new Space (Streamlit)
3. Push your code to the Space's repository

## Client Insights Features

The Client Insights tab provides:

| Feature | Description |
|---------|-------------|
| **Gap Analysis** | Identify clients with High AUM/No SIP, Reduced SIP, No Top-Up, SIP Terminated, Below Benchmark |
| **Pareto Analysis** | Top 20% clients contribution analysis |
| **Client Tiers** | Platinum/Gold/Silver segmentation |
| **Revenue Calculator** | Estimated annual revenue from each client |
| **Conversion Opportunities** | Clients with needs but no investment mapping |
| **Privacy** | Partial mobile numbers displayed (e.g., 98****1234) |

## Project Structure

```
Grad Project/
├── src/
│   ├── analysis/           # Analysis modules
│   │   ├── portfolio_review.py      # Portfolio exposure analysis
│   │   ├── sip_insights.py         # Client insights & gap analysis
│   │   ├── portfolio_builder.py    # Portfolio basket builder
│   │   ├── amc_concentration.py  # AMC concentration analysis
│   │   ├── peer_comparison.py    # Peer fund comparison
│   │   └── fund_shift.py         # Fund shift advisor
│   ├── dashboard/          # Streamlit dashboard (app.py)
│   ├── data/             # Data processing scripts
│   └── scoring/           # Fund scoring engine
├── data/
│   └── processed/         # Processed data files
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .streamlit/         # Streamlit configuration
```

## License

MIT
