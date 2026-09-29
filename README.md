# Mutual Fund Overlap Detector

Streamlit app that shows how much two mutual funds overlap, and what an investor really
owns after merging several funds into one look-through portfolio.

📘 **Educational tool, not investment advice.** All data is synthetic — fictional funds
and companies generated from realistic mandate rules, not real fund holdings.

## What it does

- **App (Phase 3)** — pick funds, enter allocation amounts, choose a month: get a
  pairwise overlap heatmap, top-20 merged stock exposures, a concentration-flag table
  (stocks above 5% of the total corpus), and a 36-month overlap-over-time chart.
- **Pairwise overlap** — overlap between any two funds, with the largest contributing
  stocks.
- **All pairs & sanity check** — overlap for every fund pair in a month, sanity checks
  (no self-pairs, no duplicates, values in range), and category-level expectations
  (e.g. Large Cap vs Large Cap should run higher than Small Cap vs Index).
- **Data checks** — dataset integrity: weights sum to 100% per fund-month, no nulls,
  no duplicate rows, no negative weights.

**Rule:** cash is excluded before any overlap math, securities are matched on **ISIN**
(never company name), and overlap between two funds = the sum, over every security both
hold, of the **smaller** of their two weights.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Data

The app looks for a bundled CSV next to `app.py` (or inside a `data/` folder), trying
these names in order: `fund_overlap_dataset.csv`, `fund_overlap_dataset.csv.gz`,
`fund_overlap_holdings_dataset.csv`, `fund_overlap_holdings_dataset.csv.gz`. Both plain
and gzip-compressed CSVs are supported.

Required columns: `fund_name, date, isin, pct_nav`. Optional: `amc, category, company,
market_value`. If no bundled file is found, upload one in the sidebar, or tick
"Use synthetic demo data" to try the app without real data.

## Files
- `app.py` - Streamlit UI (all four tabs)
- `overlap_core.py` - overlap, merged-portfolio, and data-check logic
- `demo_data.py` - synthetic sample data used by the demo-data toggle
- `fund_overlap_holdings_dataset.csv` - the shared dataset (60 funds, 36 months,
  350 ISINs, 118,800 rows)
