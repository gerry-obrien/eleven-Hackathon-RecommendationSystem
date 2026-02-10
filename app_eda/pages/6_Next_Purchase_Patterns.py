from __future__ import annotations

import streamlit as st
import pandas as pd

from app_eda.components.bootstrap import add_repo_root_to_path
repo_root = add_repo_root_to_path()

from app_eda.components.data import load_data, enrich_transactions
from app_eda.components.filters import render_sidebar_filters, apply_filters
from app_eda.components.charts import bar

st.title("Next Purchase Patterns")
st.markdown(
    """
**Purpose:** demonstrate that “next purchase” is not random.  
If we can show repeat and transition patterns (e.g., category A → B), we justify a recommender that increases cross-sell and conversion.
"""
)

data = load_data(repo_root / "data" / "raw")
txe = enrich_transactions(data.transactions, data.clients, data.products, data.stores)

filters = render_sidebar_filters(txe)
txf = apply_filters(txe, filters)

if "Category" not in txf.columns:
    st.warning("Category not found. Check that products.csv has Category and enrichment is working.")
    st.stop()

st.subheader("Repeat rate: same category on next purchase")
tx_sorted = txf.sort_values(["ClientID", "SaleTransactionDate"])
tx_sorted["next_category"] = tx_sorted.groupby("ClientID")["Category"].shift(-1)
tx_sorted["is_repeat_category"] = (tx_sorted["Category"] == tx_sorted["next_category"])
repeat_rate = tx_sorted["is_repeat_category"].dropna().mean()
st.metric("Repeat rate (category → same category next time)", f"{repeat_rate*100:.1f}%")

st.subheader("Top category transitions (A → B)")
pairs = tx_sorted.dropna(subset=["Category", "next_category"])
trans = (
    pairs.groupby(["Category", "next_category"])
    .size()
    .rename("count")
    .reset_index()
    .sort_values("count", ascending=False)
)
topn = st.slider("Number of transitions to show", 10, 50, 20, 5)
top_trans = trans.head(topn).copy()
top_trans["transition"] = top_trans["Category"].astype(str) + " → " + top_trans["next_category"].astype(str)

st.plotly_chart(bar(top_trans, "transition", "count", "Top category transitions"), use_container_width=True)
st.dataframe(top_trans[["Category", "next_category", "count"]], use_container_width=True)

st.subheader("Client-level diversity (how many categories each client buys)")
cat_per_client = txf.groupby("ClientID")["Category"].nunique().rename("n_categories")
st.metric("Median categories per client", int(cat_per_client.median()) if len(cat_per_client) else 0)

st.caption("Business value: predictable transitions and repeat behavior justify recommendation models (co-purchase / sequence-based), improving cross-sell and retention.")
