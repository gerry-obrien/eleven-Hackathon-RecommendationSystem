import sys
from pathlib import Path
import streamlit as st
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_eda.components.bootstrap import add_repo_root_to_path
add_repo_root_to_path()

from app_demo.components.demo_data import (
    load_enriched_transactions,
    dataset_today,
    segment_gap_thresholds,
)

st.set_page_config(page_title="Business Value", layout="wide")

st.title("Business Value")
st.markdown(
    """
This page:

1) **Count how many repeat customers are currently at risk of churn** (late vs typical repeat timing).  
2) **Assume we target a portion of them** with a recommendation message.  
3) Use offline **HitRate@10** improvement (Baseline 4% → Model 23%) to estimate **extra purchases**.
"""
)

# -----------------------------
# Fixed model quality (from offline evaluation)
# -----------------------------
BASELINE_HIT10 = 0.04
MODEL_HIT10 = 0.23

st.sidebar.header("Assumptions (minimal)")
target_rate = st.sidebar.slider(
    "Targeting rate (share of at-risk customers contacted)",
    0.0, 1.0, 0.60, 0.05
)

# One business assumption we cannot avoid:
baseline_conv = st.sidebar.slider(
    "Baseline conversion of targeted at-risk customers",
    0.0, 0.10, 0.05, 0.005,
    help="Default 5% is a simple scenario input for at-risk customers."
)

# One conservative mapping constant (keep fixed to remove another slider)
BETA = 0.20  # only 20% of relevance gain translates into conversion lift (conservative)

# -----------------------------
# Build at-risk pool (data-driven)
# -----------------------------
txe = load_enriched_transactions()

date_col = None
for c in ["SaleTransactionDate", "Date", "TransactionDate"]:
    if c in txe.columns:
        date_col = c
        break

if date_col is None or "ClientID" not in txe.columns:
    st.error("Missing ClientID or transaction date column in transactions.")
    st.stop()

seg_col = "ClientSegment" if "ClientSegment" in txe.columns else None

t = txe[["ClientID", date_col] + ([seg_col] if seg_col else [])].copy()
t[date_col] = pd.to_datetime(t[date_col], errors="coerce", utc=True).dt.tz_convert(None)
t = t.dropna(subset=["ClientID", date_col])

today = dataset_today()
st.caption(f"Timeline note: treating **{today.date()}** as 'today' (max transaction date in dataset).")

# Eligible repeat customers (>=2 purchases)
counts = t["ClientID"].value_counts()
eligible_ids = counts[counts >= 2].index
eligible_clients = int(len(eligible_ids))

# Last purchase + days since last (eligible only)
last_purchase = (
    t.groupby("ClientID")[date_col].max()
    .rename("last_purchase_date")
    .reset_index()
)
last_purchase = last_purchase[last_purchase["ClientID"].isin(eligible_ids)].copy()
last_purchase["days_since_last"] = (today - last_purchase["last_purchase_date"].dt.normalize()).dt.days

# Segment p75 thresholds
thr = segment_gap_thresholds()  # expected columns: ClientSegment, p75_gap_days

# Global p75 fallback (from purchase gaps)
ts = t.sort_values(["ClientID", date_col]).copy()
ts["prev_date"] = ts.groupby("ClientID")[date_col].shift(1)
ts["gap_days"] = (ts[date_col] - ts["prev_date"]).dt.days
gaps = ts.dropna(subset=["gap_days"])
gaps = gaps[gaps["gap_days"] >= 0]
global_p75 = float(gaps["gap_days"].quantile(0.75)) if not gaps.empty else 30.0

if seg_col:
    seg_map = (
        t.dropna(subset=[seg_col])
         .groupby("ClientID")[seg_col].first()
         .rename("ClientSegment")
         .reset_index()
    )
    last_purchase = last_purchase.merge(seg_map, on="ClientID", how="left")
else:
    last_purchase["ClientSegment"] = None

if thr is not None and not thr.empty and "p75_gap_days" in thr.columns:
    last_purchase = last_purchase.merge(thr, on="ClientSegment", how="left")
    last_purchase["p75_gap_days"] = last_purchase["p75_gap_days"].fillna(global_p75)
else:
    last_purchase["p75_gap_days"] = global_p75

# At-risk definition
last_purchase["at_risk"] = last_purchase["days_since_last"] > last_purchase["p75_gap_days"]
n_at_risk = int(last_purchase["at_risk"].sum())
pct_at_risk = (n_at_risk / eligible_clients * 100) if eligible_clients else 0.0

# -----------------------------
# Simple impact math using HitRate@10 improvement
# -----------------------------
n_targeted = int(round(n_at_risk * target_rate))

# Relevance gain as relative improvement vs baseline
# (23% vs 4% means the model is ~5.75x as likely to include the eventual purchase in the Top-10)
relevance_gain_rel = (MODEL_HIT10 - BASELINE_HIT10) / BASELINE_HIT10  # e.g., 4.75

# Convert a fraction of that relevance improvement into conversion lift
rel_lift = max(0.0, BETA * relevance_gain_rel)  # conservative

p0 = baseline_conv
p1 = min(1.0, p0 * (1.0 + rel_lift))

extra_products_sold = int(round(n_targeted * (p1 - p0)))

# Interpret extra purchases as "saved at-risk customers" (avoided churn)
saved_customers = extra_products_sold  # 1 extra purchase ~= 1 saved customer in this simple story
saved_rate_of_targeted = (saved_customers / n_targeted * 100) if n_targeted else 0.0
saved_rate_of_at_risk = (saved_customers / n_at_risk * 100) if n_at_risk else 0.0


# -----------------------------
# Output (3 headline metrics + impact)
# -----------------------------
st.subheader("Headline numbers")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Eligible repeat clients", f"{eligible_clients:,}")
    st.caption("Clients with **≥ 2 purchases** (cold-start excluded).")
with c2:
    st.metric("Clients at risk of churn", f"{n_at_risk:,}")
    st.caption(f"Late vs typical repeat timing (**segment p75**, fallback global p75). ({pct_at_risk:.1f}% of eligible)")
with c3:
    st.metric("Clients targeted", f"{n_targeted:,}")
    st.caption("At-risk × targeting rate.")

st.divider()

st.subheader("Churn avoided (scenario)")

c4, c5, c6 = st.columns(3)
with c4:
    st.metric("At-risk customers saved", f"{saved_customers:,}")
    st.caption("Estimated number of at-risk clients who make a repeat purchase because of recommendations.")
with c5:
    st.metric("% of targeted saved", f"{saved_rate_of_targeted:.2f}%")
    st.caption("Saved ÷ targeted clients.")
with c6:
    st.metric("% of all at-risk saved", f"{saved_rate_of_at_risk:.2f}%")
    st.caption("Saved ÷ all at-risk clients.")

st.success(f"Estimated additional products sold: **{extra_products_sold:,}**")
st.caption("Simple interpretation: one extra repeat purchase from an at-risk client ≈ one avoided churn event.")

st.divider()

st.subheader("How this was estimated (simple)")
st.write(f"- Baseline HitRate@10: **{BASELINE_HIT10*100:.0f}%**")
st.write(f"- Model HitRate@10: **{MODEL_HIT10*100:.0f}%**")
st.write(f"- Baseline conversion (targeted at-risk): **{p0*100:.2f}%**")
st.write(f"- Scenario conversion with recommender: **{p1*100:.2f}%**")

st.caption(
    "Methodology: (1) define churn-risk as 'late' vs segment p75 repeat timing, "
    "(2) target a share of at-risk clients, "
    "(3) use HitRate@10 improvement to estimate a conservative conversion lift. "
    "This is a scenario estimate, not causal proof."
)

