from __future__ import annotations

import streamlit as st
import pandas as pd

from app_eda.components.bootstrap import add_repo_root_to_path
repo_root = add_repo_root_to_path()

from app_eda.components.data import load_data, enrich_transactions
from app_eda.components.filters import render_sidebar_filters, apply_filters

st.title("Data Healthcheck")
st.markdown(
    """
**Purpose:** establish trust in the dataset and confirm we can safely join tables.
This supports the business case that the system is **feasible** and results are **credible**.
"""
)

data = load_data(repo_root / "data" / "raw")
txe = enrich_transactions(data.transactions, data.clients, data.products, data.stores)

filters = render_sidebar_filters(txe)
txf = apply_filters(txe, filters)

st.subheader("Missing values (filtered enriched transactions)")
miss = txf.isna().mean().sort_values(ascending=False).to_frame("missing_rate")
st.dataframe(miss.head(30), use_container_width=True)

st.subheader("Join coverage checks (full transactions)")
tx = data.transactions
clients = data.clients
products = data.products
stores = data.stores

coverage = {
    "transactions": len(tx),
    "tx_with_client": tx["ClientID"].isin(set(clients["ClientID"])).mean(),
    "tx_with_product": tx["ProductID"].isin(set(products["ProductID"])).mean(),
    "tx_with_store": tx["StoreID"].isin(set(stores["StoreID"])).mean(),
}
st.dataframe(pd.DataFrame([coverage]), use_container_width=True)

st.subheader("Basic validity checks (clients)")
c = data.clients.copy()
invalid_age = c["Age"].notna() & ((c["Age"] < 0) | (c["Age"] > 110))
out = pd.DataFrame(
    [{
        "n_clients": len(c),
        "age_missing_rate": float(c["Age"].isna().mean()) if "Age" in c.columns else None,
        "invalid_age_count": int(invalid_age.sum()) if "Age" in c.columns else None,
        "optin_email_rate": float(c["ClientOptINEmail"].mean()) if "ClientOptINEmail" in c.columns else None,
        "optin_phone_rate": float(c["ClientOptINPhone"].mean()) if "ClientOptINPhone" in c.columns else None,
    }]
)
st.dataframe(out, use_container_width=True)

st.caption("Business value: clean joins and known data limitations let us propose a robust, defensible recommender design.")
