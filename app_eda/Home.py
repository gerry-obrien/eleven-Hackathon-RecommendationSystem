# app_eda/Home.py
from __future__ import annotations

import streamlit as st

from app_eda.components.bootstrap import add_repo_root_to_path

repo_root = add_repo_root_to_path()

from pathlib import Path
from app_eda.components.data import load_data, enrich_transactions
from app_eda.components.filters import render_sidebar_filters, apply_filters

st.set_page_config(page_title="EDA Explorer - Next Purchase", layout="wide")

st.title("EDA Explorer — Your Next Purchase (Recommendation Case)")
st.markdown(
    """
This internal app helps the team **understand the data** and build a strong case for a **stock-aware recommendation system**.
Use the sidebar filters to focus on specific time windows, countries, or customer segments.
"""
)

raw_dir = repo_root / "data" / "raw"
data = load_data(raw_dir)

txe = enrich_transactions(data.transactions, data.clients, data.products, data.stores)
filters = render_sidebar_filters(txe)
txf = apply_filters(txe, filters)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions (filtered)", f"{len(txf):,}")
c2.metric("Clients (filtered)", f"{txf['ClientID'].nunique():,}")
c3.metric("Products (filtered)", f"{txf['ProductID'].nunique():,}")
c4.metric("Date range", f"{txf['SaleTransactionDate'].min().date()} → {txf['SaleTransactionDate'].max().date()}")

st.divider()

st.subheader("How to use the pages")
st.markdown(
    """
- **Data Healthcheck**: verify quality and joins (trust & feasibility).
- **Customers & Sparsity**: quantify cold-start and customer frequency (drives model strategy).
- **Products & Long Tail**: show catalog skew (why “top sellers” isn’t enough).
- **Country & Segment Differences**: prove heterogeneity (why personalization adds value).
- **Stock & Actionability**: ensure recommendations are sellable (operational constraint).
- **Next Purchase Patterns**: show predictability (why recommenders can work).
"""
)

st.info("Tip: If the app feels slow, reduce the date range in the sidebar.")
