import sys
from pathlib import Path
import streamlit as st
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_eda.components.bootstrap import add_repo_root_to_path  # keeps consistent path behavior
add_repo_root_to_path()

from app_demo.components.demo_data import (
    load_demo_recs,
    get_client_rec_ids,
    build_product_lookup,
    get_client_row,
    get_recent_purchases,
    get_days_since_last_purchase,
    annotate_stock,
    risk_label,
    repeat_recs_from_predict_py,
    purchase_count_for_client,
    cold_start_recs_from_cli,
)


st.set_page_config(page_title="Client Demo", layout="wide")

st.title("Client Demo")
st.markdown(
    """
**Purpose:** Show a single client’s Top recommendations.  
Focus: **reduce churn** by showing relevant items at the right time, and filtering to **in-stock** items.
"""
)

demo_recs = load_demo_recs()
product_lu = build_product_lookup()

st.sidebar.header("Demo controls")
st.sidebar.header("Demo controls")

# Keep one known-good example ID for easy copy/paste during a demo
example_id = "4508698145640552159"

client_id_str = st.sidebar.text_input(
    "ClientID (type/paste)",
    value=str(example_id),
    help="Type a ClientID to test cold-start (0 purchases) vs repeat customers."
).strip()

# Basic validation
if not client_id_str.isdigit():
    st.sidebar.error("ClientID must be numeric.")
    st.stop()

client_id = int(client_id_str)

k = st.sidebar.slider("Top-K", 5, 10, 10, 1)
stock_only = st.sidebar.checkbox("Only show in-stock items (if stock coverage exists)", value=True)

purchase_count = purchase_count_for_client(client_id)
is_cold_start = (purchase_count == 0)

# Client metadata
crow = get_client_row(client_id)
client_country = crow["ClientCountry"].iloc[0] if (len(crow) and "ClientCountry" in crow.columns) else None
client_segment = crow["ClientSegment"].iloc[0] if (len(crow) and "ClientSegment" in crow.columns) else None

days_since = get_days_since_last_purchase(client_id)
label, help_text = risk_label(days_since, client_segment)

from app_demo.components.demo_data import dataset_today
st.caption(f"Demo timeline: treating **{dataset_today().date()}** as 'today' (max transaction date in dataset).")

# Recommendations
if is_cold_start:
    rec_ids = cold_start_recs_from_cli(client_id, k=k)
else:
    rec_ids = repeat_recs_from_predict_py(client_id, k=k)

if not rec_ids:
    st.warning("No recommendations were returned for this client.")

rec_df = pd.DataFrame({"rank": range(1, len(rec_ids) + 1), "ProductID": rec_ids})

# Normalize dtypes to avoid int/str merge failures
rec_df["ProductID"] = rec_df["ProductID"].astype(str)
product_lu = product_lu.copy()
product_lu["ProductID"] = product_lu["ProductID"].astype(str)

rec_df = rec_df.merge(product_lu, on="ProductID", how="left")

rec_df = annotate_stock(rec_df, client_country)

if stock_only and "StockStatus" in rec_df.columns:
    # Keep Unknown (no coverage) so the UI doesn't go empty; only drop explicit out-of-stock.
    rec_df = rec_df[rec_df["StockStatus"].isin(["In stock", "Unknown"])].reset_index(drop=True)

# Layout
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Client snapshot")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Country", client_country if client_country else "—")
    with c2:
        st.metric("Segment", client_segment if client_segment else "—")
        
    st.metric("Purchase count", f"{purchase_count}")

    c3, c4 = st.columns(2)
    with c3:
        st.metric("Days since last purchase", "—" if days_since is None else f"{days_since} days")
    with c4:
        st.metric("Risk", label)

    st.caption(help_text)

    st.subheader("Recent purchases (last 5)")
    if is_cold_start:
        st.write("**No previous purchases.**")
    else:
        st.dataframe(get_recent_purchases(client_id, n=5), use_container_width=True)


with right:
    st.subheader(f"Top recommendations")
    # show most useful columns first if present
    # For recommendations, show the taxonomy columns directly (no long ProductLabel)
    preferred = [
        "rank",
        "ProductID",
        "Universe",
        "FamilyLevel1",
        "FamilyLevel2",
        "Category",
        "StockStatus",
    ]

    cols = [c for c in preferred if c in rec_df.columns]
    st.dataframe(rec_df[cols], use_container_width=True)


    st.caption(
        "Demo notes: 'Risk' is a simple rule using segment-level repeat timing (p75 of interpurchase gaps). "
        "Stock status uses country-level stock coverage where available."
    )
