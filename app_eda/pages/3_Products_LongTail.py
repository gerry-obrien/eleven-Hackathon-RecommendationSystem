import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import streamlit as st
import pandas as pd

from app_eda.components.bootstrap import add_repo_root_to_path
repo_root = add_repo_root_to_path()

from app_eda.components.data import load_data, enrich_transactions
from app_eda.components.filters import render_sidebar_filters, apply_filters
from app_eda.components.charts import hist, bar

st.title("Products & Long Tail")
st.markdown(
    """
**Purpose:** show catalog scale and popularity skew (long tail).  
This explains why “top sellers to everyone” is suboptimal and why ranking + relevance matter.
"""
)

data = load_data(repo_root / "data" / "raw")
txe = enrich_transactions(data.transactions, data.clients, data.products, data.stores)

filters = render_sidebar_filters(txe)
txf = apply_filters(txe, filters)

per_product = txf.groupby("ProductID").size()

st.plotly_chart(hist(per_product.clip(upper=200), "Transactions per product (clipped at 200)"), use_container_width=True)

st.subheader("Top products (by transaction count)")
topn = 20
top_products = (
    txf["ProductID"].value_counts().head(topn).rename("tx_count").reset_index().rename(columns={"index": "ProductID"})
)
st.dataframe(top_products, use_container_width=True)

st.subheader("Category mix (share)")
if "Category" in txf.columns:
    cat = txf["Category"].astype(str).value_counts(normalize=True).head(15).rename("share").reset_index()
    cat.columns = ["Category", "share"]
    cat["share_pct"] = (cat["share"] * 100).round(2)
    st.dataframe(cat, use_container_width=True)
else:
    st.info("Category column not available (check products enrichment).")

st.subheader("Revenue by category (top 15)")
if "SalesNetAmountEuro" in txf.columns and "Category" in txf.columns:
    rev = (
        txf.groupby("Category")["SalesNetAmountEuro"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .rename("revenue")
        .reset_index()
    )
    st.plotly_chart(bar(rev, "Category", "revenue", "Revenue by category (top 15)"), use_container_width=True)

st.caption("Business value: long-tail structure motivates personalized retrieval+ranking, improving conversion vs generic best-seller pushes.")

import streamlit as st

st.divider()

st.markdown(
    """
### Conclusion (Retention-focused)
Product demand is strongly **long-tailed**: a small number of items are bought very frequently, while most products are purchased only rarely.  
This directly supports the need for a recommender: in a large catalog with a long tail, customers will not naturally discover the “right” items, and generic best-seller lists quickly become repetitive and irrelevant for repeat shoppers. To reduce churn, we should rank items by **personal relevance**—matching each client’s demonstrated interests (e.g., categories/brands/price band) to maximize their **probability of buying**—with a small, controlled amount of exploration to keep recommendations fresh and avoid fatigue. The goal is **repeat purchase likelihood and retention**, not revenue.
"""
)
