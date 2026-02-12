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

from app_demo.components.demo_data import load_enriched_transactions, load_private_data, dataset_today, segment_gap_thresholds

st.set_page_config(page_title="Business Value (Evidence-First)", layout="wide")

st.title("Business Value (Evidence-First)")
st.markdown(
    """
**Goal:** reduce churn by increasing the probability of a *next purchase* among repeat customers.  
This page avoids big “€ impact” claims and instead builds a clear story using **measurable** numbers from the dataset,
plus a small optional scenario at the end.
"""
)

# -----------------------------
# Load data
# -----------------------------
txe = load_enriched_transactions()
data = load_private_data()

date_col = None
for c in ["SaleTransactionDate", "Date", "TransactionDate"]:
    if c in txe.columns:
        date_col = c
        break

if date_col is None or "ClientID" not in txe.columns:
    st.error("Missing ClientID or transaction date column in enriched transactions.")
    st.stop()

seg_col = "ClientSegment" if "ClientSegment" in txe.columns else None

# Normalize datetime to tz-naive
t = txe[["ClientID", date_col] + ([seg_col] if seg_col else [])].copy()
t[date_col] = pd.to_datetime(t[date_col], errors="coerce", utc=True).dt.tz_convert(None)
t = t.dropna(subset=["ClientID", date_col])

# -----------------------------
# Build client-level recency table (days since last purchase)
# -----------------------------
today = dataset_today()

last_purchase = (
    t.groupby("ClientID")[date_col].max().rename("last_purchase_date").reset_index()
)
last_purchase["days_since_last"] = (today - last_purchase["last_purchase_date"].dt.normalize()).dt.days

if seg_col:
    # attach segment per client (first non-null)
    seg_map = (
        t.dropna(subset=[seg_col])
         .groupby("ClientID")[seg_col]
         .first()
         .rename("ClientSegment")
         .reset_index()
    )
    client_recency = last_purchase.merge(seg_map, on="ClientID", how="left")
else:
    client_recency = last_purchase.copy()
    client_recency["ClientSegment"] = None

# Eligible repeat customers (>=2 purchases)
purchase_counts = t["ClientID"].value_counts()
eligible_ids = purchase_counts[purchase_counts >= 2].index
eligible_clients = int(len(eligible_ids))

client_recency["is_eligible"] = client_recency["ClientID"].isin(eligible_ids)
eligible_recency = client_recency[client_recency["is_eligible"]].copy()

# -----------------------------
# Define "at risk" using segment p75 (evidence-based)
# -----------------------------
thr = segment_gap_thresholds()  # columns: ClientSegment, p75_gap_days
if thr.empty or thr["p75_gap_days"].isna().all():
    st.warning(
        "Could not compute segment p75 repeat windows. Falling back to a simple global threshold (30 days)."
    )
    eligible_recency["p75_gap_days"] = 30.0
else:
    eligible_recency = eligible_recency.merge(thr, on="ClientSegment", how="left")

# For clients with missing segment or missing p75, use global p75 of gaps as fallback
# Build gaps to get global p75
t_sorted = t.sort_values(["ClientID", date_col]).copy()
t_sorted["prev_date"] = t_sorted.groupby("ClientID")[date_col].shift(1)
t_sorted["gap_days"] = (t_sorted[date_col] - t_sorted["prev_date"]).dt.days
gaps = t_sorted.dropna(subset=["gap_days"])
gaps = gaps[gaps["gap_days"] >= 0]

global_p75 = float(gaps["gap_days"].quantile(0.75)) if not gaps.empty else 30.0
eligible_recency["p75_gap_days"] = eligible_recency["p75_gap_days"].fillna(global_p75)

eligible_recency["at_risk"] = eligible_recency["days_since_last"] > eligible_recency["p75_gap_days"]

# -----------------------------
# Evidence KPIs (no big assumptions)
# -----------------------------
n_at_risk = int(eligible_recency["at_risk"].sum())
pct_at_risk = (n_at_risk / eligible_clients * 100) if eligible_clients else 0.0

# Baseline "natural return" among at-risk customers:
# look at historical gaps: for each gap, if the previous gap was already "late" relative to segment p75,
# did the customer still return within a win-back window W?
W = st.sidebar.slider("Win-back window W (days after becoming late)", 7, 60, 14, 1)

# Build gap-level table with segment and threshold at the time of the gap
if seg_col:
    gap_df = t_sorted[["ClientID", date_col, seg_col, "prev_date", "gap_days"]].dropna(subset=["gap_days"]).copy()
    gap_df = gap_df.rename(columns={seg_col: "ClientSegment"})
    gap_df = gap_df.merge(thr, on="ClientSegment", how="left")
    gap_df["p75_gap_days"] = gap_df["p75_gap_days"].fillna(global_p75)
else:
    gap_df = t_sorted[["ClientID", date_col, "prev_date", "gap_days"]].dropna(subset=["gap_days"]).copy()
    gap_df["ClientSegment"] = None
    gap_df["p75_gap_days"] = global_p75

# "Late event" means the gap exceeded p75 (they took longer than typical to return)
gap_df["was_late"] = gap_df["gap_days"] > gap_df["p75_gap_days"]

# Among late events, define "win-back success" = return happens within p75 + W days
gap_df["winback_success"] = gap_df["gap_days"] <= (gap_df["p75_gap_days"] + W)

late_events = gap_df[gap_df["was_late"]].copy()
late_events_count = int(len(late_events))
baseline_winback_rate = float(late_events["winback_success"].mean() * 100) if late_events_count else 0.0

# Segment breakdown for at-risk share (optional but useful)
seg_break = None
if eligible_recency["ClientSegment"].notna().any():
    seg_break = (
        eligible_recency.groupby("ClientSegment")["at_risk"]
        .agg(at_risk_rate=lambda s: s.mean() * 100, n_clients="size")
        .reset_index()
        .sort_values("at_risk_rate", ascending=False)
    )

# -----------------------------
# Storyline through numbers
# -----------------------------
st.subheader("1) How big is the churn-risk problem (in our data)?")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Eligible repeat customers", f"{eligible_clients:,}")
    st.caption("Clients with **≥ 2 purchases** (we exclude cold-start).")
with c2:
    st.metric("At-risk customers (now)", f"{n_at_risk:,}")
    st.caption("Eligible clients whose **days since last purchase** is above their segment’s **p75 repeat window**.")
with c3:
    st.metric("At-risk share", f"{pct_at_risk:.1f}%")
    st.caption("How much of the eligible base is currently ‘late’ relative to their segment’s typical timing.")

st.caption(f"Demo timeline: treating **{today.date()}** as 'today' (max transaction date in the dataset).")

if seg_break is not None and not seg_break.empty:
    st.subheader("At-risk share by segment")
    fig = px.bar(seg_break, x="ClientSegment", y="at_risk_rate", title="At-risk share (eligible clients)")
    fig.update_yaxes(title="% at risk")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(seg_break, use_container_width=True)

st.divider()

st.subheader("2) What typically happens without intervention?")
st.markdown(
    """
We look at historical cases where customers were already **late** (their gap exceeded the segment p75).
Then we measure how often they still returned within a short **win-back window**.
"""
)

c4, c5 = st.columns(2)
with c4:
    st.metric("Late repeat events in history", f"{late_events_count:,}")
    st.caption("Count of historical purchase gaps where the customer returned **later than p75** for their segment.")
with c5:
    st.metric(f"Baseline win-back rate within p75 + {W} days", f"{baseline_winback_rate:.1f}%")
    st.caption(
        "Among those late events, % where the customer still returned within a limited time window "
        "(a proxy for the ‘natural’ return rate when someone is already late)."
    )

st.divider()

st.subheader("3) Where the recommender adds value")
st.markdown(
    """
A churn-reduction recommender helps by:
- targeting **at-risk** customers (late relative to their segment),
- showing **high-relevance** items (based on purchase history and observed next-purchase patterns),
- so that more of them return during the win-back window.
"""
)

# -----------------------------
# Small optional scenario (clearly labeled)
# -----------------------------
st.subheader("4) Optional scenario (small, explicit assumption)")
st.markdown(
    """
This is intentionally small and transparent:  
**If** recommendations increase the win-back rate among at-risk customers by +Δ percentage points,
how many additional customers would return within the win-back window?
"""
)

delta_pp = st.slider("Assumed win-back uplift among at-risk customers (Δ pp)", 0.0, 10.0, 2.0, 0.5)

# Apply to at-risk pool size as a simple scale estimate
delta_prob = delta_pp / 100.0
incremental_saved = int(round(n_at_risk * delta_prob))

st.success(
    f"Estimated additional at-risk customers returning within the win-back window: **{incremental_saved:,}**"
)

st.caption(
    "Note: This is a pitch-friendly scale estimate. It does not claim causal uplift—only quantifies what a given uplift would mean in volume."
)

st.divider()

st.markdown(
    """
### Conclusion
- We can quantify the **at-risk pool** using segment-specific repeat timing (p75).  
- We can also measure a baseline **natural win-back rate** from historical “late” cases.  
- The recommender’s job is to improve that win-back rate by making outreach **timely and relevant**, reducing churn.
"""
)
