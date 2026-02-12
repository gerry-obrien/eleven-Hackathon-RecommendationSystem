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

st.title("Business Value (Simple)")
st.markdown(
    """
We keep this simple:

1) Size the **at-risk churn pool** from real purchase timing data.  
2) Apply a **targeting rate** (how many we contact).  
3) Use HitRate@10 improvement to estimate **extra purchases** (simple scenario).
"""
)

# -----------------------------
# Inputs
# -----------------------------
st.sidebar.header("Inputs (Top-10)")
baseline_hit = st.sidebar.slider("Baseline HitRate@10 (%)", 0.0, 100.0, 15.0, 0.5)
model_hit = st.sidebar.slider("Recommender HitRate@10 (%)", 0.0, 100.0, 25.0, 0.5)

target_rate = st.sidebar.slider(
    "Targeting rate (share of at-risk clients contacted)",
    0.0, 1.0, 0.60, 0.05
)

baseline_conv = st.sidebar.slider(
    "Baseline conversion of targeted clients (buy anyway)",
    0.0, 0.20, 0.03, 0.005
)

# Optional: simple mapping coefficient (kept hidden-ish but adjustable)
alpha = st.sidebar.slider(
    "How strongly HitRate gain converts to conversion lift (α)",
    0.0, 1.0, 0.20, 0.05,
    help="Conservative knob. 0.2 means we assume only 20% of HitRate gain translates into conversion lift."
)

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

# Eligible repeat customers (>=2 purchases)
counts = t["ClientID"].value_counts()
eligible_ids = counts[counts >= 2].index
eligible_clients = int(len(eligible_ids))

# Last purchase + days since last
last_purchase = t.groupby("ClientID")[date_col].max().rename("last_purchase_date").reset_index()
last_purchase["days_since_last"] = (today - last_purchase["last_purchase_date"].dt.normalize()).dt.days
last_purchase = last_purchase[last_purchase["ClientID"].isin(eligible_ids)].copy()

# Segment p75 thresholds
thr = segment_gap_thresholds()  # ClientSegment, p75_gap_days

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

# -----------------------------
# Simple business impact math
# -----------------------------
n_targeted = int(round(n_at_risk * target_rate))

hit_gain = max(0.0, (model_hit - baseline_hit) / 100.0)  # 0..1
# Conversion lift is proportional to HitRate gain (simple scenario)
conv_lift = alpha * hit_gain

p0 = baseline_conv
p1 = min(1.0, p0 * (1.0 + conv_lift))

extra_products_sold = int(round(n_targeted * (p1 - p0)))

# -----------------------------
# Output (simple storyline)
# -----------------------------
st.subheader("Results")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Eligible repeat clients", f"{eligible_clients:,}")
    st.caption("Clients with ≥2 purchases (we exclude cold-start).")
with c2:
    st.metric("Clients at risk of churn", f"{n_at_risk:,}")
    st.caption("Late vs typical repeat timing (segment p75).")
with c3:
    st.metric("Clients targeted", f"{n_targeted:,}")
    st.caption("At-risk × targeting rate.")

st.divider()

st.subheader("Estimated impact (scenario)")
st.write(f"- **Baseline HitRate@10:** {baseline_hit:.1f}%")
st.write(f"- **Recommender HitRate@10:** {model_hit:.1f}%")
st.write(f"- **HitRate gain:** {(model_hit - baseline_hit):+.1f} pp")
st.write(f"- **Assumed baseline conversion (targeted clients):** {p0*100:.2f}%")
st.write(f"- **Scenario conversion with recommender:** {p1*100:.2f}%")

st.success(f"**Estimated additional products sold (extra purchases): {extra_products_sold:,}**")
st.caption(
    "This is a simple scenario: we assume only a small fraction (α) of the HitRate gain translates into conversion lift."
)

st.caption(f"Demo timeline: using {today.date()} as 'today' (max date in dataset).")
