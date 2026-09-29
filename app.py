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

ROOT = Path(__file__).parent
# Works whether the CSV sits next to app.py or inside a data/ folder.
BUNDLED_CANDIDATES = [ROOT / "fund_overlap_dataset.csv", ROOT / "data" / "fund_overlap_dataset.csv"]


# ---------- data loading ----------
@st.cache_data(show_spinner="Reading data...")
def load_from_bytes(raw: bytes) -> pd.DataFrame:
    return oc.prepare(pd.read_csv(io.BytesIO(raw)))


@st.cache_data
def load_demo() -> pd.DataFrame:
    return oc.prepare(make_demo_df())


@st.cache_data(show_spinner="Computing overlap for every fund pair, every month...")
def pairs_all_months(df: pd.DataFrame) -> pd.DataFrame:
    return oc.all_pairs_all_months(df)


st.title("📊 Mutual Fund Overlap Detector")
st.caption(
    "Cash is excluded, securities are matched on ISIN, and each shared stock "
    "contributes the smaller of the two weights."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload holdings CSV", type="csv")
    use_demo = st.checkbox("Use synthetic demo data", value=False)
    st.caption("Required columns: fund_name, date, isin, pct_nav. "
               "Optional: amc, category, company, market_value.")

bundled = next((p for p in BUNDLED_CANDIDATES if p.exists()), None)
try:
    if uploaded is not None:
        df = load_from_bytes(uploaded.getvalue())
        source = "uploaded file"
    elif use_demo:
        df = load_demo()
        source = "synthetic demo data (fictional securities)"
    elif bundled is not None:
        df = load_from_bytes(bundled.read_bytes())
        source = "bundled dataset"
    else:
        st.info("Upload a holdings CSV in the sidebar, or tick **Use synthetic demo data** to try the app.")
        st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

st.sidebar.success("Loaded: %s" % source)

dates = sorted(df["date"].dt.date.unique())
date = st.sidebar.selectbox("Month-end", dates, index=len(dates) - 1)
snap_all = df[df["date"] == pd.Timestamp(date)]          # equity + cash
equity = oc.equity_only(snap_all)                        # equity only
funds = sorted(equity["fund_name"].unique())
categories = oc.fund_categories(df)

tab_overlap, tab_port, tab_pairs, tab_checks = st.tabs(
    ["Pairwise overlap", "Merged portfolio", "All pairs & sanity check", "Data checks"])

# ---------- pairwise overlap ----------
with tab_overlap:
    if len(funds) < 2:
        st.warning("Need at least two funds at this date.")
    else:
        c2, c3 = st.columns(2)
        fund_a = c2.selectbox("Fund A", funds, index=0)
        fund_b = c3.selectbox("Fund B", funds, index=1)

        if fund_a == fund_b:
            st.warning("Pick two different funds (a fund compared with itself is not a finding).")
        else:
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
                st.download_button("Download shared holdings (CSV)",
                                   table.to_csv(index=False).encode("utf-8"),
                                   file_name="overlap_%s.csv" % date, mime="text/csv")

# ---------- merged portfolio ----------
with tab_port:
    st.write("Enter how much you have invested in each fund. Each stock's rupee exposure is "
             "(amount in fund × stock's weight in that fund), added up across funds.")
    picked = st.multiselect("Funds held", funds, default=funds[:2])
    if len(picked) < 1:
        st.info("Select at least one fund.")
    else:
        cols = st.columns(min(len(picked), 4))
        alloc = {}
        for i, f in enumerate(picked):
            alloc[f] = cols[i % len(cols)].number_input(
                f, min_value=0, value=50000, step=5000, key="alloc_" + f)
        corpus = sum(alloc.values())
        flag_pct = st.slider("Concentration flag: flag a stock above this % of the total corpus", 1.0, 20.0, 5.0, 0.5)

        if corpus <= 0:
            st.warning("Total invested amount must be above zero.")
        else:
            port = oc.merged_portfolio(snap_all, alloc, flag_pct)
            flagged = port[port["flag"]]
            stocks = port[port["isin"] != oc.CASH_ISIN]

            p1, p2, p3 = st.columns(3)
            p1.metric("Total corpus", "₹{:,.0f}".format(corpus))
            p2.metric("Distinct stocks", len(stocks))
            p3.metric("Stocks above %.1f%%" % flag_pct, len(flagged))

            show = port.copy()
            show["flag"] = [("🚩 held by %d of %d funds" % (n, len(picked))) if f else ""
                            for f, n in zip(port["flag"], port["funds_holding"])]
            show["exposure"] = show["exposure"].round(2)
            show["pct_of_corpus"] = show["pct_of_corpus"].round(2)
            st.dataframe(show, width="stretch", hide_index=True)

            recon = port["exposure"].sum()
            st.caption("Reconciliation: exposures (stocks + cash) add up to ₹{:,.2f} "
                       "vs corpus ₹{:,.2f} → {}".format(recon, corpus, "✅ match" if abs(recon - corpus) < 1 else "❌ mismatch"))
            st.download_button("Download merged portfolio (CSV)", port.to_csv(index=False).encode("utf-8"),
                               file_name="merged_portfolio_%s.csv" % date, mime="text/csv")

# ---------- all pairs & sanity ----------
with tab_pairs:
    pairs = oc.all_pairs_overlap(equity, categories)
    n = len(funds)
    st.write("**%d funds → %d unique pairs** for %s (self-pairs and mirror pairs excluded)." % (n, len(pairs), date))

    st.subheader("Sanity checks (this month)")
    sc = oc.pair_sanity_checks(pairs, n, equity)
    sc["Passed"] = sc["Passed"].map({True: "✅", False: "❌"})
    st.dataframe(sc, width="stretch", hide_index=True)

    st.subheader("Overlap by category pair")
    st.caption("Use this to judge whether results look right for what each category holds.")
    st.dataframe(oc.category_summary(pairs), width="stretch")

    st.subheader("Pairs to review")
    z = st.slider("Robust z-score threshold", 2.0, 5.0, 3.0, 0.5)
    out = oc.flag_outliers(pairs, z)
    st.caption("%d pair(s) sit unusually far from the rest of their own category pair. "
               "These are for review, not deletion." % len(out))
    if len(out):
        st.dataframe(out.round(2), width="stretch", hide_index=True)

    st.subheader("All pairs")
    top_first = pairs.sort_values("overlap", ascending=False)
    st.dataframe(top_first.round(2), width="stretch", hide_index=True)
    st.download_button("Download pairs for this month (CSV)", top_first.to_csv(index=False).encode("utf-8"),
                       file_name="all_pairs_%s.csv" % date, mime="text/csv")

    st.subheader("Full run: every month")
    if st.button("Compute all months"):
        allm = pairs_all_months(df)
        st.success("%s pair-month results across %d months." % ("{:,}".format(len(allm)), allm["date"].nunique()))
        st.write("Overlap range: %.2f to %.2f · missing values: %d" % (
            allm["overlap"].min(), allm["overlap"].max(), int(allm["overlap"].isna().sum())))
        st.download_button("Download all months (CSV)", allm.to_csv(index=False).encode("utf-8"),
                           file_name="all_pairs_all_months.csv", mime="text/csv")

# ---------- data checks ----------
with tab_checks:
    r = oc.integrity_report(df)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", "{:,}".format(r["rows"]))
    k2.metric("Funds", r["funds"])
    k3.metric("Month-ends", r["dates"])
    k4.metric("Equity ISINs", r["equity_isins"])
    st.write("Date range: **%s** to **%s**" % (r["first_date"], r["last_date"]))
    st.write("Unique fund pairs per month for this many funds: **{:,}**".format(r["funds"] * (r["funds"] - 1) // 2))

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
