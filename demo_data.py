"""Small synthetic dataset so the app can be demoed without the real data."""

import numpy as np
import pandas as pd

FUNDS = [
    ("Aravali Large Cap Fund", "Aravali AMC", 40, "large"),
    ("Nilgiri Large Cap Fund", "Nilgiri AMC", 45, "large"),
    ("Chenab Small Cap Fund", "Chenab AMC", 50, "small"),
    ("Chenab Index Fund", "Chenab AMC", 60, "large"),
]
DATES = ["2025-10-31", "2025-11-30", "2025-12-31"]


def make_demo_df(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # 40 "large" and 60 "small" fictional securities
    universe = {
        "large": [("INEDEMO%04d" % i, "Demo Large Co %d" % i) for i in range(40)],
        "small": [("INEDEMO%04d" % i, "Demo Small Co %d" % i) for i in range(40, 100)],
    }
    rows = []
    for date in DATES:
        for fund, amc, n, kind in FUNDS:
            pool = universe["large"] + (universe["small"] if fund.endswith("Index Fund") else [])
            if kind == "small":
                pool = universe["small"] + universe["large"][:15]
            picks = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
            cash = round(float(rng.uniform(1, 4)), 4)
            w = rng.dirichlet(np.ones(len(picks)) * 2) * (100 - cash)
            for k, wt in zip(picks, w):
                isin, name = pool[k]
                rows.append((fund, amc, date, isin, name, round(float(wt), 4), round(float(wt) * 1e6, 2)))
            rows.append((fund, amc, date, "CASH_EQV", "Cash and Equivalents", cash, round(cash * 1e6, 2)))
    df = pd.DataFrame(rows, columns=["fund_name", "amc", "date", "isin", "company", "pct_nav", "market_value"])
    # make each fund-month sum to exactly 100
    tot = df.groupby(["fund_name", "date"])["pct_nav"].transform("sum")
    df["pct_nav"] = (df["pct_nav"] * 100 / tot).round(4)
    return df
