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

# --- 3rd figure: True basket value proxy by country + segment (with legend) ---
import plotly.express as px

st.subheader("Average basket value proxy (sum of SalesNetAmountEuro per shopping trip) by country & segment")

if "SalesNetAmountEuro" not in txf.columns:
    st.info("SalesNetAmountEuro not found.")
elif not all(c in txf.columns for c in ["ClientID", "StoreID", "SaleTransactionDate"]):
    st.info("Need ClientID, StoreID, and SaleTransactionDate to build a basket proxy.")
elif "ClientCountry" not in txf.columns or "ClientSegment" not in txf.columns:
    st.info("Need ClientCountry and ClientSegment to compare basket value by group.")
else:
    t = txf.copy()

    # Ensure datetime and create a basket key
    t["SaleTransactionDate"] = pd.to_datetime(t["SaleTransactionDate"], errors="coerce")
    t["basket_date"] = t["SaleTransactionDate"].dt.date  # proxy for one shopping trip/day

    # Drop unusable rows
    t = t.dropna(subset=["ClientID", "StoreID", "basket_date", "SalesNetAmountEuro", "ClientCountry", "ClientSegment"])

    # Build pseudo-baskets: one basket per (ClientID, StoreID, date)
    baskets = (
        t.groupby(["ClientID", "StoreID", "basket_date"], as_index=False)
         .agg(
             basket_value=("SalesNetAmountEuro", "sum"),
             n_lines=("ProductID", "nunique") if "ProductID" in t.columns else ("SalesNetAmountEuro", "size"),
             n_items=("Quantity", "sum") if "Quantity" in t.columns else ("SalesNetAmountEuro", "size"),
         )
    )

    # Attach segment + country (client-level attributes)
    meta = (
        t[["ClientID", "ClientCountry", "ClientSegment"]]
        .drop_duplicates(subset=["ClientID"])
    )
    baskets = baskets.merge(meta, on="ClientID", how="left").dropna(subset=["ClientCountry", "ClientSegment"])

    # Aggregate basket value by country + segment
    agg = (
        baskets.groupby(["ClientCountry", "ClientSegment"], as_index=False)
               .agg(
                   avg_basket_value=("basket_value", "mean"),
                   median_basket_value=("basket_value", "median"),
                   n_baskets=("basket_value", "size"),
               )
    )

    # Keep chart readable: top 10 countries by basket count
    top_countries = (
        baskets["ClientCountry"].value_counts().head(10).index.tolist()
    )
    agg_plot = agg[agg["ClientCountry"].isin(top_countries)].copy()

    # Plot with legend (one color per segment)
    fig = px.bar(
        agg_plot,
        x="ClientCountry",
        y="avg_basket_value",
        color="ClientSegment",
        barmode="group",
        title="Avg basket value (proxy) by country and segment (top countries)",
        hover_data={"n_baskets": True, "median_basket_value": True},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table for transparency
    st.dataframe(
        agg.sort_values(["ClientCountry", "avg_basket_value"], ascending=[True, False]),
        use_container_width=True
    )

st.caption(
    "Basket proxy groups line-items into one shopping trip using (ClientID, StoreID, date). "
    "This is a practical approximation when no BasketID/OrderID exists."
)

st.info(
    "Slide export (exec comprehensive): "
    "`.venv\\Scripts\\python.exe scripts/export_slide_plots.py --chart country_segment_comprehensive --preset country_segment_exec_wide` "
    "-> `plots/country_segment_comprehensive_exec.png`"
)

st.divider()

st.markdown(
    """
### Conclusion
These charts show clear differences in buying behaviour across **countries** and **customer segments**.  
The category mix changes by country, so “one global list” will not fit every market. The category mix also changes by segment (TOP customers buy different types of products than LOYAL and INACTIVE_1Y). Finally, the basket-value proxy shows that **TOP customers spend much more per shopping trip** in every country.  
Overall, this supports using **country- and segment-aware recommendations** to keep items relevant and increase repeat purchases (reduce churn).
"""
)

