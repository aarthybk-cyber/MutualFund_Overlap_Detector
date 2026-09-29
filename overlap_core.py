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

    # Some builds of the dataset carry one CASH_EQV row per fund-month; others
    # (fully-invested, no separate cash line) carry none at all. Either is a
    # consistent representation - what's WRONG is something in between.
    cash_consistent = len(cash) == 0 or len(cash) == len(totals)

    return {
        "rows": len(df),
        "funds": df["fund_name"].nunique(),
        "dates": df["date"].nunique(),
        "first_date": df["date"].min().date(),
        "last_date": df["date"].max().date(),
        "equity_isins": equity["isin"].nunique(),
        "cash_rows": len(cash),
        "cash_consistent": bool(cash_consistent),
        "fund_months": len(totals),
        "total_min": float(totals.min()),
        "total_max": float(totals.max()),
        "outside_tolerance": int(len(outside)),
        "nulls": int(df[REQUIRED_COLUMNS].isnull().sum().sum()),
        "duplicates": int(df.duplicated(subset=["fund_name", "date", "isin"]).sum()),
        "negative_pct_nav": int((df["pct_nav"] < 0).sum()),
    }


# ======================================================================
# Phase 2 additions: all fund pairs, merged portfolio, sanity checks
# ======================================================================
from itertools import combinations

import numpy as np


def weight_matrix(equity: pd.DataFrame) -> pd.DataFrame:
    """Funds (rows) x ISINs (columns) of pct_nav; 0 where a fund does not hold the stock."""
    return equity.pivot_table(index="fund_name", columns="isin", values="pct_nav",
                              aggfunc="sum", fill_value=0.0)


def fund_categories(df: pd.DataFrame) -> pd.Series:
    """fund_name -> category (or 'Unknown' if the CSV has no category column)."""
    if "category" in df.columns:
        return df.groupby("fund_name")["category"].first()
    return pd.Series("Unknown", index=df["fund_name"].unique())


def all_pairs_overlap(equity: pd.DataFrame, categories: pd.Series = None) -> pd.DataFrame:
    """Overlap for every UNIQUE pair of funds in one month's equity snapshot.

    combinations(n, 2) never yields (A, A) and never yields both (A, B) and
    (B, A), so self-pairs and mirror duplicates cannot appear.
    """
    M = weight_matrix(equity)
    funds = M.index.to_numpy()
    X = M.to_numpy()
    held = X > 0
    rows = []
    for i in range(len(funds) - 1):
        shared = np.minimum(X[i], X[i + 1:]).sum(axis=1)
        common = (held[i] & held[i + 1:]).sum(axis=1)
        for k, (ov, c) in enumerate(zip(shared, common)):
            rows.append((funds[i], funds[i + 1 + k], float(ov), int(c)))
    out = pd.DataFrame(rows, columns=["fund_a", "fund_b", "overlap", "common_stocks"])
    if categories is not None:
        out["category_a"] = out["fund_a"].map(categories)
        out["category_b"] = out["fund_b"].map(categories)
        out["category_pair"] = [" vs ".join(sorted(p)) for p in zip(out["category_a"], out["category_b"])]
    return out


def all_pairs_all_months(df: pd.DataFrame) -> pd.DataFrame:
    cats = fund_categories(df)
    frames = []
    for d in sorted(df["date"].unique()):
        p = all_pairs_overlap(equity_only(df[df["date"] == d]), cats)
        p.insert(0, "date", pd.Timestamp(d))
        frames.append(p)
    return pd.concat(frames, ignore_index=True)


def pair_sanity_checks(pairs: pd.DataFrame, n_funds: int, equity: pd.DataFrame = None) -> pd.DataFrame:
    """Hard checks on one month's pair table. Returns a Check / Passed / Detail table."""
    expected = n_funds * (n_funds - 1) // 2
    checks = [
        ("Pair count = n(n-1)/2", len(pairs) == expected,
         "%d pairs for %d funds (expected %d)" % (len(pairs), n_funds, expected)),
        ("No self-pairs (A vs A)", int((pairs["fund_a"] == pairs["fund_b"]).sum()) == 0,
         "%d self-pairs" % int((pairs["fund_a"] == pairs["fund_b"]).sum())),
        ("No mirror duplicates (A,B and B,A)",
         len({frozenset(p) for p in zip(pairs["fund_a"], pairs["fund_b"])}) == len(pairs),
         "%d unique unordered pairs" % len({frozenset(p) for p in zip(pairs["fund_a"], pairs["fund_b"])})),
        ("Every overlap between 0 and 100", bool(pairs["overlap"].between(0, 100 + 1e-9).all()),
         "range %.2f to %.2f" % (pairs["overlap"].min(), pairs["overlap"].max())),
        ("No missing overlap values", int(pairs["overlap"].isna().sum()) == 0,
         "%d missing" % int(pairs["overlap"].isna().sum())),
    ]
    if equity is not None:
        eq_sum = equity.groupby("fund_name")["pct_nav"].sum()
        cap = np.minimum(pairs["fund_a"].map(eq_sum), pairs["fund_b"].map(eq_sum))
        bad = int((pairs["overlap"] > cap + 1e-9).sum())
        checks.append(("Overlap never exceeds the smaller fund's equity weight", bad == 0,
                       "%d violations" % bad))
    return pd.DataFrame(checks, columns=["Check", "Passed", "Detail"])


def category_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Overlap distribution per category pair - the basis for 'does this look right?'."""
    g = pairs.groupby("category_pair")["overlap"]
    out = g.agg(pairs="count", min="min", median="median", mean="mean", max="max").round(2)
    return out.sort_values("median", ascending=False)


def flag_outliers(pairs: pd.DataFrame, z: float = 3.0) -> pd.DataFrame:
    """Pairs unusually far from the rest of their own category pair (robust z-score).

    Flagged rows are for REVIEW, not deletion."""
    p = pairs.copy()
    grp = p.groupby("category_pair")["overlap"]
    med = grp.transform("median")
    mad = grp.transform(lambda s: (s - s.median()).abs().median())
    scale = (1.4826 * mad).replace(0, np.nan)
    p["robust_z"] = ((p["overlap"] - med) / scale).round(2)
    return p[p["robust_z"].abs() > z].sort_values("robust_z", key=lambda s: s.abs(), ascending=False)


def merged_portfolio(equity_and_cash: pd.DataFrame, allocations: dict, flag_pct: float = 5.0) -> pd.DataFrame:
    """Merge several funds into one look-through portfolio.

    allocations: {fund_name: rupees invested}. Exposure of a stock =
    sum over funds of (rupees in fund x stock's pct_nav / 100). Cash is kept as
    its own row so the total reconciles to the corpus.
    """
    total = float(sum(allocations.values()))
    parts = []
    for fund, amount in allocations.items():
        rows = equity_and_cash[equity_and_cash["fund_name"] == fund]
        if rows.empty:
            raise ValueError("Fund not found at this date: %s" % fund)
        h = rows.groupby("isin")["pct_nav"].sum().rename("wt").reset_index()
        h["fund"] = fund
        h["exposure"] = amount * h["wt"] / 100.0
        parts.append(h)
    allp = pd.concat(parts, ignore_index=True)
    out = allp.groupby("isin").agg(exposure=("exposure", "sum"), funds_holding=("fund", "nunique"))
    if "company" in equity_and_cash.columns:
        names = equity_and_cash.groupby("isin")["company"].first()
        out.insert(0, "company", out.index.map(names))
    out["pct_of_corpus"] = out["exposure"] / total * 100.0
    out["flag"] = (out["pct_of_corpus"] > flag_pct) & (out.index != CASH_ISIN)
    out = out.sort_values("exposure", ascending=False)
    return out.reset_index()


# ======================================================================
# Phase 3 additions: heatmap matrix + overlap-over-time for a fund set
# ======================================================================

def overlap_matrix(equity: pd.DataFrame, funds: list) -> pd.DataFrame:
    """Symmetric fund x fund overlap matrix for the given funds (diagonal = NaN,
    since a fund's overlap with itself is not a meaningful finding)."""
    funds = [f for f in dict.fromkeys(funds)]  # de-dupe, keep order
    M = weight_matrix(equity)
    missing = [f for f in funds if f not in M.index]
    if missing:
        raise ValueError("Fund(s) not found at this date: %s" % ", ".join(missing))
    X = M.loc[funds].to_numpy()
    n = len(funds)
    mat = np.empty((n, n))
    for i in range(n):
        row = np.minimum(X[i], X).sum(axis=1)
        mat[i] = row
    np.fill_diagonal(mat, np.nan)
    return pd.DataFrame(mat, index=funds, columns=funds)


def heat_color(val, vmin=0.0, vmax=100.0) -> str:
    """CSS background-color for one heatmap cell: pale yellow (low) -> deep red (high)."""
    if pd.isna(val):
        return "background-color:#f2f2f2; color:#999"
    t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin + 1e-9)))
    r = round(255 + (178 - 255) * t)
    g = round(255 + (24 - 255) * t)
    b = round(224 + (43 - 224) * t)
    text = "#fff" if t > 0.55 else "#1c2430"
    return "background-color: rgb(%d,%d,%d); color:%s" % (r, g, b, text)


def pair_time_series(all_months_pairs: pd.DataFrame, funds: list) -> pd.DataFrame:
    """Wide table: date x pair, overlap %, restricted to pairs within `funds`."""
    fs = set(funds)
    sub = all_months_pairs[
        all_months_pairs["fund_a"].isin(fs) & all_months_pairs["fund_b"].isin(fs)
    ].copy()
    if sub.empty:
        return sub
    short = lambda n: n.replace(" Fund", "")
    sub["pair"] = sub["fund_a"].map(short) + " / " + sub["fund_b"].map(short)
    wide = sub.pivot(index="date", columns="pair", values="overlap").sort_index()
    return wide
