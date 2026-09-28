"""
Core logic for the Mutual Fund Overlap Detector (no Streamlit imports,
so it can be tested and reused on its own).

Rules implemented (same as pairwise_overlap.py / Exercise 2.3b):
  * Cash (isin == CASH_EQV) is removed before any overlap maths.
  * Securities are matched on ISIN, never on company name.
  * For every security held by BOTH funds take the SMALLER pct_nav weight,
    then sum those minimums = pairwise overlap.
"""

import pandas as pd

CASH_ISIN = "CASH_EQV"
REQUIRED_COLUMNS = ["fund_name", "date", "isin", "pct_nav"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns and normalise types."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required column(s): %s. Found: %s"
            % (", ".join(missing), ", ".join(df.columns))
        )
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["pct_nav"] = pd.to_numeric(df["pct_nav"], errors="coerce")
    return df


def equity_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["isin"] != CASH_ISIN]


def snapshot(df: pd.DataFrame, date) -> pd.DataFrame:
    """Equity rows for one month-end (cash removed)."""
    snap = df[df["date"] == pd.Timestamp(date)]
    if snap.empty:
        raise ValueError("No rows found for date %s" % date)
    return equity_only(snap)


def fund_holdings(equity: pd.DataFrame, fund_name: str) -> pd.Series:
    """One fund's holdings as a Series indexed by ISIN (grouped, so safe
    even if a security is reported on two rows)."""
    rows = equity[equity["fund_name"] == fund_name]
    if rows.empty:
        raise ValueError("Fund not found at this date: %s" % fund_name)
    return rows.groupby("isin")["pct_nav"].sum()


def pairwise_overlap(equity: pd.DataFrame, fund_a: str, fund_b: str):
    """Return (overlap_pct, merged_df, summary_dict)."""
    a = fund_holdings(equity, fund_a)
    b = fund_holdings(equity, fund_b)

    merged = pd.concat([a.rename("wt_a"), b.rename("wt_b")], axis=1, join="inner")
    merged["shared"] = merged[["wt_a", "wt_b"]].min(axis=1)

    if "company" in equity.columns:
        names = equity.groupby("isin")["company"].first()
        merged.insert(0, "company", merged.index.map(names))

    merged = merged.sort_values("shared", ascending=False)
    overlap = float(merged["shared"].sum())

    summary = {
        "holdings_a": len(a),
        "holdings_b": len(b),
        "common": len(merged),
        "weight_sum_a": float(a.sum()),
        "weight_sum_b": float(b.sum()),
        "overlap": overlap,
    }
    return overlap, merged, summary


def integrity_report(df: pd.DataFrame, tol: float = 0.01) -> dict:
    """The Exercise 2.1 style checks, returned as a dict."""
    cash = df[df["isin"] == CASH_ISIN]
    equity = equity_only(df)
    totals = df.groupby(["fund_name", "date"])["pct_nav"].sum()
    outside = totals[(totals < 100 - tol) | (totals > 100 + tol)]

    return {
        "rows": len(df),
        "funds": df["fund_name"].nunique(),
        "dates": df["date"].nunique(),
        "first_date": df["date"].min().date(),
        "last_date": df["date"].max().date(),
        "equity_isins": equity["isin"].nunique(),
        "cash_rows": len(cash),
        "fund_months": len(totals),
        "total_min": float(totals.min()),
        "total_max": float(totals.max()),
        "outside_tolerance": int(len(outside)),
        "nulls": int(df[REQUIRED_COLUMNS].isnull().sum().sum()),
        "duplicates": int(df.duplicated(subset=["fund_name", "date", "isin"]).sum()),
        "negative_pct_nav": int((df["pct_nav"] < 0).sum()),
    }
