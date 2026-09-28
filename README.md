# Mutual Fund Overlap Detector

Streamlit app that computes pairwise holdings overlap between two funds at a month-end.

- Cash (`CASH_EQV`) is removed before any maths
- Securities are matched on ISIN
- Overlap = sum over shared securities of `min(weight_A, weight_B)`

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Data
Upload a CSV in the sidebar (columns: `fund_name, date, isin, pct_nav`; optional `amc, company, market_value`),
or tick "Use synthetic demo data".

## Files
- `app.py` – Streamlit UI
- `overlap_core.py` – overlap and data-check logic
- `demo_data.py` – synthetic sample data
