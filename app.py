"""Mutual Fund Overlap Detector - Streamlit app.

Run locally:  streamlit run app.py
"""

import io
from pathlib import Path

import pandas as pd
import streamlit as st

import overlap_core as oc
from demo_data import make_demo_df

st.set_page_config(page_title="Fund Overlap Detector", page_icon="📊", layout="wide")

ROOT = Path(__file__).parent
# Works whether the CSV (plain or gzipped) sits next to app.py or inside a data/ folder.
_NAMES = ["fund_overlap_dataset.csv", "fund_overlap_dataset.csv.gz",
          "fund_overlap_holdings_dataset.csv", "fund_overlap_holdings_dataset.csv.gz"]
BUNDLED_CANDIDATES = [ROOT / n for n in _NAMES] + [ROOT / "data" / n for n in _NAMES]


# ---------- data loading ----------
@st.cache_data(show_spinner="Reading data...")
def load_from_bytes(raw: bytes, gzipped: bool = False) -> pd.DataFrame:
    comp = "gzip" if gzipped else None
    return oc.prepare(pd.read_csv(io.BytesIO(raw), compression=comp))


@st.cache_data
def load_demo() -> pd.DataFrame:
    return oc.prepare(make_demo_df())


@st.cache_data(show_spinner="Computing overlap for every fund pair, every month...")
def pairs_all_months(df: pd.DataFrame) -> pd.DataFrame:
    return oc.all_pairs_all_months(df)


st.title("📊 Fund Overlap Detector")
st.markdown(
    "##### Uncover the stocks hiding across your mutual funds — before a single bad quarter hits your whole portfolio."
)
st.caption(
    "Cash is excluded, securities are matched on ISIN, and each shared stock "
    "contributes the smaller of the two weights."
)
st.info(
    "📘 **Educational tool, not investment advice.** No output here is a "
    "buy/sell/hold recommendation. **All data is synthetic** — fictional funds and "
    "companies generated from realistic mandate rules, not real fund holdings.",
    icon="ℹ️",
)

bundled = next((p for p in BUNDLED_CANDIDATES if p.exists()), None)

# ---------- sidebar: data source ----------
with st.sidebar:
    st.markdown("### 🗂️ Data source")
    src_options = (["📦 Bundled dataset"] if bundled is not None else []) + \
        ["⬆️ Upload my own CSV", "🧪 Synthetic demo data"]
    data_source = st.radio("Data source", src_options, label_visibility="collapsed")

    uploaded = None
    if data_source == "⬆️ Upload my own CSV":
        uploaded = st.file_uploader("Holdings CSV", type=["csv"], label_visibility="collapsed")

    with st.expander("ℹ️ CSV format"):
        st.caption(
            "**Required:** `fund_name`, `date`, `isin`, `pct_nav`  \n"
            "**Optional:** `amc`, `category`, `company`, `market_value`"
        )

try:
    if data_source == "⬆️ Upload my own CSV":
        if uploaded is None:
            st.info("👈 Upload a holdings CSV in the sidebar to get started.")
            st.stop()
        df = load_from_bytes(uploaded.getvalue(), gzipped=uploaded.name.endswith(".gz"))
        source = "uploaded file"
    elif data_source == "🧪 Synthetic demo data":
        df = load_demo()
        source = "synthetic demo data (fictional securities)"
    else:
        df = load_from_bytes(bundled.read_bytes(), gzipped=bundled.suffix == ".gz")
        source = "bundled dataset (%s)" % bundled.name
except ValueError as e:
    st.error(str(e))
    st.stop()

n_isins = df.loc[df["isin"] != oc.CASH_ISIN, "isin"].nunique()
with st.sidebar:
    st.success("Loaded: %s" % source)
    st.caption(
        "**{:,}** rows · **{}** funds · **{}** months · **{}** ISINs".format(
            len(df), df["fund_name"].nunique(), df["date"].nunique(), n_isins)
    )
    st.divider()
    st.markdown("### 📅 Filters")
    dates = sorted(df["date"].dt.date.unique())
    date = st.selectbox("Month-end", dates, index=len(dates) - 1)

snap_all = df[df["date"] == pd.Timestamp(date)]          # equity + cash
equity = oc.equity_only(snap_all)                        # equity only
funds = sorted(equity["fund_name"].unique())
categories = oc.fund_categories(df)

tab_product, tab_overlap, tab_pairs, tab_checks = st.tabs(
    ["📱 My Portfolio", "🔍 Pairwise overlap", "🧮 All pairs & sanity check", "✅ Data checks"])

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
                                   file_name="overlap_%s.csv" % date, mime="text/csv",
                                   type="primary")

# ---------- Phase 3 product: fund/allocation picker, heatmap, top-20, flags, time series ----------
with tab_product:
    st.write("Search and select the funds in your portfolio, enter how much you've put into "
             "each, and pick a month. Everything below updates for that selection.")
    picked = st.multiselect("Funds held (type to search)", funds, default=funds[:3], key="p3_funds")

    if len(picked) < 1:
        st.info("Select at least one fund to see your merged position.")
    else:
        cols = st.columns(min(len(picked), 4))
        alloc = {}
        for i, f in enumerate(picked):
            alloc[f] = cols[i % len(cols)].number_input(
                f, min_value=0, value=50000, step=5000, key="p3_alloc_" + f)
        corpus = sum(alloc.values())

        if corpus <= 0:
            st.warning("Total invested amount must be above zero.")
        else:
            port = oc.merged_portfolio(snap_all, alloc, flag_pct=5.0)
            stocks = port[port["isin"] != oc.CASH_ISIN]
            flagged = stocks[stocks["flag"]]

            m1, m2, m3 = st.columns(3)
            m1.metric("Total corpus", "₹{:,.0f}".format(corpus))
            m2.metric("Distinct stocks", len(stocks))
            m3.metric("Stocks above 5% of corpus", len(flagged))

            # ----- Heatmap: pairwise overlap among selected funds -----
            st.subheader("Pairwise overlap heatmap")
            if len(picked) < 2:
                st.caption("Select at least two funds to compare their overlap.")
            else:
                hm = oc.overlap_matrix(equity, picked)
                short_names = [n.replace(" Fund", "") for n in hm.index]
                hm_display = hm.copy()
                hm_display.index = short_names
                hm_display.columns = short_names
                styled = (hm_display.style
                          .map(oc.heat_color)
                          .format("{:.1f}", na_rep="—"))
                st.dataframe(styled, width="stretch")
                st.caption("Diagonal is blank (a fund vs. itself isn't a finding). Darker = more overlap.")

            # ----- Top-20 merged holdings -----
            st.subheader("Top 20 merged stock exposures")
            top20 = stocks.sort_values("exposure", ascending=False).head(20)
            label = top20["company"] if "company" in top20.columns else top20["isin"]
            st.bar_chart(pd.Series(top20["pct_of_corpus"].values, index=label.values, name="% of corpus"))
            show20 = top20.copy()
            show20["flag"] = [("🚩 held by %d of %d funds" % (n, len(picked))) if f else ""
                              for f, n in zip(top20["flag"], top20["funds_holding"])]
            show20["exposure"] = show20["exposure"].round(2)
            show20["pct_of_corpus"] = show20["pct_of_corpus"].round(2)
            st.dataframe(show20, width="stretch", hide_index=True)

            # ----- Concentration flag table -----
            st.subheader("Concentration flags (> 5% of total corpus)")
            if flagged.empty:
                st.caption("No stock crosses 5% of the total corpus for this allocation.")
            else:
                cols_f = ["isin", "company", "exposure", "pct_of_corpus", "funds_holding"] \
                    if "company" in flagged.columns else ["isin", "exposure", "pct_of_corpus", "funds_holding"]
                ft = flagged[cols_f].copy()
                ft["exposure"] = ft["exposure"].round(2)
                ft["pct_of_corpus"] = ft["pct_of_corpus"].round(2)
                st.dataframe(ft, width="stretch", hide_index=True)

            recon = port["exposure"].sum()
            st.caption("Reconciliation: exposures (stocks + cash) add up to ₹{:,.2f} vs corpus ₹{:,.2f} → {}".format(
                recon, corpus, "✅ match" if abs(recon - corpus) < 1 else "❌ mismatch"))
            st.download_button("Download merged portfolio (CSV)", port.to_csv(index=False).encode("utf-8"),
                               file_name="merged_portfolio_%s.csv" % date, mime="text/csv",
                               type="primary")

            # ----- Overlap over time (36 months) for the selected funds -----
            st.subheader("Overlap over time")
            if len(picked) < 2:
                st.caption("Select at least two funds to see how their overlap has moved over time.")
            else:
                allm = pairs_all_months(df)
                ts = oc.pair_time_series(allm, picked)
                if ts.empty:
                    st.caption("No overlapping months found for this fund selection.")
                else:
                    st.line_chart(ts)
                    st.caption("Overlap %% between each selected pair, across all %d months on file."
                               % df["date"].nunique())



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
                       file_name="all_pairs_%s.csv" % date, mime="text/csv",
                       type="primary")

    st.subheader("Full run: every month")
    if st.button("Compute all months"):
        allm = pairs_all_months(df)
        st.success("%s pair-month results across %d months." % ("{:,}".format(len(allm)), allm["date"].nunique()))
        st.write("Overlap range: %.2f to %.2f · missing values: %d" % (
            allm["overlap"].min(), allm["overlap"].max(), int(allm["overlap"].isna().sum())))
        st.download_button("Download all months (CSV)", allm.to_csv(index=False).encode("utf-8"),
                           file_name="all_pairs_all_months.csv", mime="text/csv",
                           type="primary")

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
        ("Cash rows consistent (either none, or exactly one per fund-month)", r["cash_consistent"],
         "%d cash rows vs %d fund-months" % (r["cash_rows"], r["fund_months"])),
        ("No nulls in required columns", r["nulls"] == 0, "%d nulls" % r["nulls"]),
        ("No duplicate fund+date+isin rows", r["duplicates"] == 0, "%d duplicates" % r["duplicates"]),
        ("No negative pct_nav", r["negative_pct_nav"] == 0, "%d negative" % r["negative_pct_nav"]),
    ], columns=["Check", "Passed", "Detail"])
    checks["Passed"] = checks["Passed"].map({True: "✅", False: "❌"})
    st.dataframe(checks, width="stretch", hide_index=True)
