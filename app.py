"""Mutual Fund Overlap Detector - Streamlit app.

Run locally:  streamlit run app.py
"""

import io
from pathlib import Path

import pandas as pd
import streamlit as st

import overlap_core as oc
from demo_data import make_demo_df

st.set_page_config(page_title="Mutual Fund Overlap Detector", page_icon="📊", layout="wide")

BUNDLED_CSV = Path(__file__).parent / "data" / "fund_overlap_dataset.csv"


# ---------- data loading ----------
@st.cache_data(show_spinner="Reading data...")
def load_from_bytes(raw: bytes) -> pd.DataFrame:
    return oc.prepare(pd.read_csv(io.BytesIO(raw)))


@st.cache_data
def load_demo() -> pd.DataFrame:
    return oc.prepare(make_demo_df())


st.title("📊 Mutual Fund Overlap Detector")
st.caption(
    "Pairwise holdings overlap between two funds at a month-end. "
    "Cash is excluded, securities are matched on ISIN, and each shared stock "
    "contributes the smaller of the two weights."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload holdings CSV", type="csv")
    use_demo = st.checkbox("Use synthetic demo data", value=False)
    st.caption("Required columns: fund_name, date, isin, pct_nav. "
               "Optional: amc, company, market_value.")

try:
    if uploaded is not None:
        df = load_from_bytes(uploaded.getvalue())
        source = "uploaded file"
    elif use_demo:
        df = load_demo()
        source = "synthetic demo data (fictional securities)"
    elif BUNDLED_CSV.exists():
        df = load_from_bytes(BUNDLED_CSV.read_bytes())
        source = "bundled dataset"
    else:
        st.info("Upload a holdings CSV in the sidebar, or tick **Use synthetic demo data** to try the app.")
        st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

st.sidebar.success("Loaded: %s" % source)

tab_overlap, tab_checks = st.tabs(["Pairwise overlap", "Data checks"])

# ---------- overlap tab ----------
with tab_overlap:
    dates = sorted(df["date"].dt.date.unique())
    c1, c2, c3 = st.columns([1, 2, 2])
    date = c1.selectbox("Month-end", dates, index=len(dates) - 1)

    equity = oc.snapshot(df, date)
    funds = sorted(equity["fund_name"].unique())
    if len(funds) < 2:
        st.warning("Need at least two funds at this date.")
        st.stop()

    fund_a = c2.selectbox("Fund A", funds, index=0)
    fund_b = c3.selectbox("Fund B", funds, index=1)

    if fund_a == fund_b:
        st.warning("Pick two different funds.")
        st.stop()

    overlap, merged, s = oc.pairwise_overlap(equity, fund_a, fund_b)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pairwise overlap", "%.2f%%" % overlap)
    m2.metric("Stocks in common", s["common"])
    m3.metric("Holdings in A / B", "%d / %d" % (s["holdings_a"], s["holdings_b"]))
    m4.metric("Equity weight A / B", "%.1f%% / %.1f%%" % (s["weight_sum_a"], s["weight_sum_b"]))

    if merged.empty:
        st.info("These two funds share no securities at this date.")
    else:
        top_n = st.slider("Top contributors to show", 3, min(30, len(merged)), min(10, len(merged))) \
            if len(merged) > 3 else len(merged)
        top = merged.head(top_n)
        label = top["company"] if "company" in top.columns else top.index.to_series()
        chart = pd.DataFrame(
            {"Fund A weight": top["wt_a"].values, "Fund B weight": top["wt_b"].values},
            index=label.values,
        )
        st.subheader("Largest contributors")
        st.bar_chart(chart)
        st.caption("Top %d contribute %.2f of the %.2f total." % (top_n, top["shared"].sum(), overlap))

        st.subheader("All shared holdings")
        table = merged.reset_index().rename(columns={
            "index": "isin", "wt_a": "weight_A", "wt_b": "weight_B", "shared": "shared (min)"})
        st.dataframe(table, width="stretch", hide_index=True)
        st.download_button(
            "Download shared holdings (CSV)",
            table.to_csv(index=False).encode("utf-8"),
            file_name="overlap_%s.csv" % date,
            mime="text/csv",
        )

# ---------- checks tab ----------
with tab_checks:
    r = oc.integrity_report(df)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", "{:,}".format(r["rows"]))
    k2.metric("Funds", r["funds"])
    k3.metric("Month-ends", r["dates"])
    k4.metric("Equity ISINs", r["equity_isins"])
    st.write("Date range: **%s** to **%s**" % (r["first_date"], r["last_date"]))

    checks = pd.DataFrame([
        ("pct_nav totals within 100 ± 0.01 per fund-month",
         r["outside_tolerance"] == 0, "%d outside (range %.4f to %.4f)" % (
             r["outside_tolerance"], r["total_min"], r["total_max"])),
        ("One cash row per fund-month", r["cash_rows"] == r["fund_months"],
         "%d cash rows vs %d fund-months" % (r["cash_rows"], r["fund_months"])),
        ("No nulls in required columns", r["nulls"] == 0, "%d nulls" % r["nulls"]),
        ("No duplicate fund+date+isin rows", r["duplicates"] == 0, "%d duplicates" % r["duplicates"]),
        ("No negative pct_nav", r["negative_pct_nav"] == 0, "%d negative" % r["negative_pct_nav"]),
    ], columns=["Check", "Passed", "Detail"])
    checks["Passed"] = checks["Passed"].map({True: "✅", False: "❌"})
    st.dataframe(checks, width="stretch", hide_index=True)
