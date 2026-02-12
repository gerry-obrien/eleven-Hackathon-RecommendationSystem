import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_eda.components.bootstrap import add_repo_root_to_path
add_repo_root_to_path()

from app_demo.components.demo_data import load_enriched_transactions, segment_gap_thresholds

st.set_page_config(page_title="Business Value", layout="wide")

st.title("Business Value (Churn Reduction)")
st.markdown(
    """
**Purpose:** translate the recommender into business terms.  
We focus on churn reduction by improving the probability of **return within 30 days**.
"""
)

txe = load_enriched_transactions()

date_col = None
for c in ["SaleTransactionDate", "Date", "TransactionDate"]:
    if c in txe.columns:
        date_col = c
        break

if date_col is None or "ClientID" not in txe.columns:
    st.error("Missing ClientID or transaction date column in enriched transactions.")
    st.stop()

seg_col = "ClientSegment" if "ClientSegment" in txe.columns else None

t = txe[["ClientID", date_col] + ([seg_col] if seg_col else [])].copy()
t[date_col] = pd.to_datetime(t[date_col], errors="coerce")
t = t.dropna(subset=["ClientID", date_col]).sort_values(["ClientID", date_col])

# interpurchase gaps
t["prev_date"] = t.groupby("ClientID")[date_col].shift(1)
t["gap_days"] = (t[date_col] - t["prev_date"]).dt.days
gaps = t.dropna(subset=["gap_days"])
gaps = gaps[gaps["gap_days"] >= 0]

if gaps.empty:
    st.info("No repeat purchase gaps available under the current data.")
    st.stop()

X = st.sidebar.slider("Return window X (days)", 7, 60, 30, 1)
uplift_pp = st.sidebar.slider("Assumed uplift in return-within-X (percentage points)", 0.0, 10.0, 3.0, 0.5)

# Overall return-within-X (event-level)
overall_return = (gaps["gap_days"] <= X).mean() * 100

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Return-within-X baseline", f"{overall_return:.1f}%")
with c2:
    st.metric("Window", f"{X} days")
with c3:
    st.metric("Assumed uplift", f"+{uplift_pp:.1f} pp")

st.subheader("Baseline return-within-X by segment")
if seg_col:
    seg_rates = (
        gaps.assign(return_within_x=gaps["gap_days"] <= X)
        .groupby(seg_col)["return_within_x"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"return_within_x": "return_within_x_pct"})
        .sort_values("return_within_x_pct", ascending=False)
    )
    fig = px.bar(seg_rates, x=seg_col, y="return_within_x_pct", title=f"Return within {X} days (baseline)")
    fig.update_yaxes(title="% of repeat transitions")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(seg_rates, use_container_width=True)
else:
    st.info("ClientSegment not available, so segment breakdown cannot be shown.")

st.subheader("Scenario impact (simple estimate)")
st.markdown(
    """
This is a **scenario calculator** (not an A/B result). It shows how small uplifts can translate into meaningful retention impact.
We apply uplift to the number of eligible clients/transitions.
"""
)

# Eligible clients: repeat customers (>=2 purchases)
purchase_counts = txe["ClientID"].value_counts()
eligible_clients = (purchase_counts >= 2).sum()

# Use interpurchase events count as a proxy "opportunities"
n_events = len(gaps)

baseline_prob = overall_return / 100.0
uplift_prob = min(1.0, baseline_prob + uplift_pp / 100.0)

incremental_events = int(round(n_events * (uplift_prob - baseline_prob)))
st.write(f"- Eligible repeat customers: **{eligible_clients:,}**")
st.write(f"- Repeat-purchase transitions observed: **{n_events:,}**")
st.write(f"- Baseline return-within-{X}-days: **{overall_return:.1f}%**")
st.write(f"- With +{uplift_pp:.1f}pp uplift: **{uplift_prob*100:.1f}%**")
st.success(f"Estimated incremental repeat-within-{X}-days events: **{incremental_events:,}**")

st.divider()
st.markdown(
    """
### Conclusion
The data shows many customers return within a predictable time window.  
A churn-focused recommender aims to increase the **return-within-X-days** rate by keeping recommendations relevant and timely.
Even a small uplift can translate into a meaningful increase in repeat purchases at scale.
"""
)
