import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px

from app_eda.components.bootstrap import add_repo_root_to_path
repo_root = add_repo_root_to_path()

from app_eda.components.data import load_data, enrich_transactions
from app_eda.components.filters import render_sidebar_filters, apply_filters
from app_eda.components.charts import hist, bar

st.title("Recency & Repeat Timing")
st.markdown(
    """
**Purpose:** quantify how long customers typically take to buy again.  
This helps reduce churn by identifying **when** to trigger recommendations (before customers go “quiet”).
"""
)

data = load_data(repo_root / "data" / "raw")
txe = enrich_transactions(data.transactions, data.clients, data.products, data.stores)

filters = render_sidebar_filters(txe)
txf = apply_filters(txe, filters)

# --- Build interpurchase gaps (days between purchases) ---
required = ["ClientID", "SaleTransactionDate"]
missing = [c for c in required if c not in txf.columns]
if missing:
    st.info(f"Missing required columns for recency analysis: {missing}")
    st.stop()

t = txf[["ClientID", "SaleTransactionDate", "ClientSegment"]].copy() if "ClientSegment" in txf.columns else txf[["ClientID", "SaleTransactionDate"]].copy()
t["SaleTransactionDate"] = pd.to_datetime(t["SaleTransactionDate"], errors="coerce")
t = t.dropna(subset=["ClientID", "SaleTransactionDate"])

# Sort and compute gaps per client
t = t.sort_values(["ClientID", "SaleTransactionDate"])
t["prev_date"] = t.groupby("ClientID")["SaleTransactionDate"].shift(1)
t["gap_days"] = (t["SaleTransactionDate"] - t["prev_date"]).dt.days

gaps = t.dropna(subset=["gap_days"]).copy()
gaps = gaps[gaps["gap_days"] >= 0]  # safety

# Keep only repeat customers (>=2 purchases)
repeat_clients = gaps["ClientID"].nunique()

# -----------------------------
# 1) Histogram: interpurchase time
# -----------------------------
st.subheader("1) Days between purchases (repeat customers)")
if repeat_clients == 0:
    st.info("No repeat customers found under the current filters.")
    st.stop()

st.plotly_chart(
    hist(gaps["gap_days"], title="Distribution of days between purchases", nbins=50),
    use_container_width=True,
)
st.caption("Each point is the gap (in days) between two consecutive purchases from the same client.")

# -----------------------------
# 2) Segment comparison: median and p75
# -----------------------------
st.subheader("2) Repeat timing by segment (median and 75th percentile)")

if "ClientSegment" not in gaps.columns:
    st.info("ClientSegment not available in the filtered data, so segment comparison cannot be shown.")
else:
    seg_stats = (
        gaps.groupby("ClientSegment")["gap_days"]
        .agg(
            median_days="median",
            p75_days=lambda s: s.quantile(0.75),
            n_gaps="size",
            n_clients=lambda s: gaps.loc[s.index, "ClientID"].nunique(),
        )
        .reset_index()
        .sort_values("median_days")
    )

    # Plot median days (simple, readable)
    st.plotly_chart(
        bar(seg_stats, x="ClientSegment", y="median_days", title="Median days to next purchase (by segment)"),
        use_container_width=True,
    )
    st.dataframe(seg_stats, use_container_width=True)

    st.caption("Shorter gaps imply faster repeat cycles; longer gaps indicate higher churn risk / slower return.")

# -----------------------------
# 3) Return within X days (overall + by segment)
# -----------------------------
st.subheader("3) Return within 30 days (retention KPI)")
x_days = 30

# Event-level return rate: share of interpurchase events <= X days
overall_return_event = (gaps["gap_days"] <= x_days).mean() * 100

st.metric("Overall return-within-30-days (event-level)", f"{overall_return_event:.1f}%")

if "ClientSegment" in gaps.columns:
    seg_return = (
        gaps.assign(return_within_x=gaps["gap_days"] <= x_days)
        .groupby("ClientSegment")["return_within_x"]
        .mean()
        .mul(100)
        .rename("return_within_30_pct")
        .reset_index()
        .sort_values("return_within_30_pct", ascending=False)
    )

    fig = px.bar(
        seg_return,
        x="ClientSegment",
        y="return_within_30_pct",
        title="Return within 30 days (by segment)",
    )
    fig.update_yaxes(title="% of repeat transitions within 30 days")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(seg_return, use_container_width=True)

st.caption(
    "Business value: this quantifies when customers typically return. "
    "It supports sending recommendations around the ‘late’ window to increase next purchase probability and reduce churn."
)

# -----------------------------
# Page conclusion
# -----------------------------
st.divider()

st.markdown(
    """
### Conclusion
Repeat timing is highly concentrated: many customers buy again very quickly, and most repeat purchases happen within **30 days** (**76.9%** overall).  
There are also clear segment differences. **TOP** customers return fastest (median gap **0 days**) and have the highest 30-day return rate (**94.4%**), while **LOYAL** and **INACTIVE_1Y** return more slowly (medians **2** and **4** days) and have much lower 30-day return rates (both ~**72.7%**).

For churn reduction, this supports a recency-triggered approach: keep frequent, low-friction recommendations for **TOP** customers, and focus intervention on **LOYAL/INACTIVE_1Y** customers as they approach or pass the “late” window (around the **30-day** mark) to increase the chance of a next purchase.
"""
)

