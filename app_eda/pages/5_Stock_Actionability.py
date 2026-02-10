from __future__ import annotations

import streamlit as st
import pandas as pd

from app_eda.components.bootstrap import add_repo_root_to_path
repo_root = add_repo_root_to_path()

from app_eda.components.data import load_data, enrich_transactions, stock_map
from app_eda.components.filters import render_sidebar_filters, apply_filters
from app_eda.components.charts import bar

st.title("Stock & Actionability")
st.markdown(
    """
**Purpose:** show whether recommendations can be executed operationally.  
Even a perfect predictor is useless if it recommends out-of-stock items. This page quantifies the **stock constraint**.
"""
)

data = load_data(repo_root / "data" / "raw")
txe = enrich_transactions(data.transactions, data.clients, data.products, data.stores)

filters = render_sidebar_filters(txe)
txf = apply_filters(txe, filters)

stocks = data.stocks.copy()
stocks_pos = stocks[stocks["Quantity"] > 0].copy()

st.subheader("Stock coverage by country")
cov = (
    stocks_pos.groupby("StoreCountry")["ProductID"]
    .nunique()
    .rename("products_with_stock")
    .reset_index()
    .sort_values("products_with_stock", ascending=False)
)
st.plotly_chart(bar(cov, "StoreCountry", "products_with_stock", "Products with positive stock by country"), use_container_width=True)
st.dataframe(cov, use_container_width=True)

st.subheader("Overlap: purchased products vs in-stock products (by country)")
if "ClientCountry" in txf.columns:
    purchased_by_country = (
        txf.groupby("ClientCountry")["ProductID"]
        .apply(lambda s: set(s.unique()))
    )
    smap = stock_map(data.stocks)

    rows = []
    for country, purchased_set in purchased_by_country.items():
        country = str(country)
        instock_set = smap.get(country, set())
        overlap = purchased_set & instock_set
        rows.append(
            {
                "ClientCountry": country,
                "purchased_unique_products": len(purchased_set),
                "in_stock_unique_products": len(instock_set),
                "overlap_unique_products": len(overlap),
                "overlap_rate_vs_purchased": (len(overlap) / len(purchased_set)) if len(purchased_set) else None,
            }
        )
    overlap_df = pd.DataFrame(rows).sort_values("overlap_rate_vs_purchased", ascending=False)
    st.dataframe(overlap_df, use_container_width=True)
else:
    st.info("ClientCountry missing; cannot compute overlap by client country.")

st.subheader("How much does stock filtering shrink a Top-K list?")
k = st.slider("K (Top-K)", 5, 30, 10, 1)

top_global = txf["ProductID"].value_counts().head(k).index.tolist()
if "ClientCountry" in txf.columns:
    client_country = txf.groupby("ClientID")["ClientCountry"].first()
    smap = stock_map(data.stocks)

    kept = []
    for cid, ctry in client_country.items():
        allowed = smap.get(str(ctry))
        if allowed is None:
            kept.append(k)  # if no stock info, assume unchanged (conservative)
        else:
            kept.append(sum(1 for pid in top_global if pid in allowed))

    st.write(f"Average items remaining after stock filter (global Top-{k}): **{sum(kept)/len(kept):.2f}**")
    st.write(f"% clients with at least {k} items available from global Top-{k}: **{(sum(1 for x in kept if x==k)/len(kept))*100:.1f}%**")
else:
    st.info("ClientCountry missing; cannot simulate stock filter impact per client.")

st.caption("Business value: quantifies operational readiness and motivates stock-aware candidate filtering to avoid wasted impressions and poor customer experience.")
