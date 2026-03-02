# Enkay Investments - Fund Recommendation Analytics

A data-driven fund recommendation and SIP growth analytics system for mutual fund distribution firms.

## Features

- **Fund Ranker**: Score and rank funds based on performance, brokerage, AUM, and tie-up status
- **Peer Comparison**: Compare funds within the same category
- **Portfolio Exposure Review**: Analyze current holdings and flag underperforming schemes
- **AMC Concentration**: View AMC distribution in your current portfolio
- **Brokerage vs Performance**: Visualize the trade-off between returns and commission

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

## Deployment Options

### Option 1: Streamlit Community Cloud (Recommended - Free)
1. Push your code to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repository
4. Deploy for free!

**Configuration:**
- Main file path: `src/dashboard/app.py`
- Python version: 3.11

### Option 2: Hugging Face Spaces (Free)
1. Go to https://huggingface.co/spaces
2. Create a new Space (Streamlit)
3. Push your code to the Space's repository

### Option 3: Railway/Render/Heroku (Paid)
These platforms support Streamlit natively:
- **Railway**: https://railway.app
- **Render**: https://render.com
- **Heroku**: https://heroku.com

### Option 4: Netlify (Limited)
Netlify doesn't natively support Streamlit. Use Streamlit Cloud instead.

## Project Structure

```
Grad Project/
├── src/
│   ├── analysis/        # Analysis modules
│   ├── dashboard/      # Streamlit dashboard
│   ├── data/           # Data processing scripts
│   └── scoring/        # Fund scoring engine
├── data/
│   └── processed/      # Processed data files
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .streamlit/        # Streamlit configuration
```

## License

MIT
