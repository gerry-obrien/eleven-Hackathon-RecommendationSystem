from __future__ import annotations

import streamlit as st
import pandas as pd

from app_eda.components.bootstrap import add_repo_root_to_path
repo_root = add_repo_root_to_path()

from app_eda.components.data import load_data, enrich_transactions
from app_eda.components.filters import render_sidebar_filters, apply_filters
from app_eda.components.charts import stacked_bar_share, bar

st.title("Country & Segment Differences")
st.markdown(
    """
**Purpose:** demonstrate that purchase preferences differ across countries and segments.  
This justifies personalization: a single global campaign wastes impressions and misses local/segment tastes.
"""
)

data = load_data(repo_root / "data" / "raw")
txe = enrich_transactions(data.transactions, data.clients, data.products, data.stores)

filters = render_sidebar_filters(txe)
txf = apply_filters(txe, filters)

st.subheader("Category share by client country (top 8 categories + Other)")
if "ClientCountry" in txf.columns and "Category" in txf.columns:
    st.plotly_chart(
        stacked_bar_share(txf.dropna(subset=["ClientCountry", "Category"]), "ClientCountry", "Category", top_n=8,
                          title="Category share by client country"),
        use_container_width=True
    )
else:
    st.info("Need ClientCountry and Category columns.")

st.subheader("Category share by client segment (top 8 categories + Other)")
if "ClientSegment" in txf.columns and "Category" in txf.columns:
    st.plotly_chart(
        stacked_bar_share(txf.dropna(subset=["ClientSegment", "Category"]), "ClientSegment", "Category", top_n=8,
                          title="Category share by client segment"),
        use_container_width=True
    )
else:
    st.info("Need ClientSegment and Category columns.")

st.subheader("Average basket value proxy (SalesNetAmountEuro) by segment/country")
if "SalesNetAmountEuro" in txf.columns:
    dims = []
    if "ClientCountry" in txf.columns:
        dims.append("ClientCountry")
    if "ClientSegment" in txf.columns:
        dims.append("ClientSegment")

    if dims:
        agg = (
            txf.groupby(dims)["SalesNetAmountEuro"]
            .mean()
            .rename("avg_sales_net")
            .reset_index()
            .sort_values("avg_sales_net", ascending=False)
            .head(25)
        )
        x = dims[0]
        st.plotly_chart(bar(agg, x=x, y="avg_sales_net", title="Avg SalesNetAmountEuro (top groups)"), use_container_width=True)
        st.dataframe(agg, use_container_width=True)
    else:
        st.info("No segment/country columns found for grouping.")
else:
    st.info("SalesNetAmountEuro not found.")

st.caption("Business value: segment/country heterogeneity supports targeted messaging and local assortments—key drivers of conversion and retention.")
