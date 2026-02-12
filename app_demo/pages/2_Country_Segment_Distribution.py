import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_eda.components.bootstrap import add_repo_root_to_path
add_repo_root_to_path()

from app_demo.components.demo_data import load_demo_recs, build_product_lookup, load_private_data

st.set_page_config(page_title="Country & Segment Distribution", layout="wide")

st.title("Country / Segment Distribution")
st.markdown(
    """
**Purpose:** Show that recommendations are not one-size-fits-all by comparing the **category mix** of recommended items
across **countries** and **segments**.
"""
)

demo_recs = load_demo_recs()
prod_lu = build_product_lookup()
data = load_private_data()

# Build long format recommendations
REC_COLS = [f"rec_{i}" for i in range(1, 11)]
long_recs = (
    demo_recs.melt(id_vars=["ClientID"], value_vars=REC_COLS, var_name="rank_col", value_name="ProductID")
    .dropna(subset=["ProductID"])
)
long_recs["rank"] = long_recs["rank_col"].str.replace("rec_", "", regex=False).astype(int)

# Attach client metadata
clients = data.clients.copy()
clients["ClientID"] = clients["ClientID"].astype(str)
cols = ["ClientID"]
for c in ["ClientCountry", "ClientSegment"]:
    if c in clients.columns:
        cols.append(c)
client_meta = clients[cols].drop_duplicates(subset=["ClientID"])

long_recs = long_recs.merge(client_meta, on="ClientID", how="left")
long_recs = long_recs.merge(prod_lu, on="ProductID", how="left")

if "Category" not in long_recs.columns:
    st.error("No 'Category' column found in products.csv, so we cannot plot category distributions.")
    st.stop()

# Helper: top-N categories + Other
def topn_other(df: pd.DataFrame, group_col: str, cat_col: str, top_n: int = 8) -> pd.DataFrame:
    top = df[cat_col].value_counts().head(top_n).index.tolist()
    out = df.copy()
    out[cat_col] = out[cat_col].where(out[cat_col].isin(top), other="Other")
    return out

st.sidebar.header("Controls")
top_n = st.sidebar.slider("Top N categories (+ Other)", 5, 12, 8, 1)

# Plot by country
st.subheader("Recommended category share by country")
if "ClientCountry" not in long_recs.columns:
    st.info("ClientCountry not available in clients.csv.")
else:
    dfc = long_recs.dropna(subset=["ClientCountry", "Category"])
    dfc = topn_other(dfc, "ClientCountry", "Category", top_n=top_n)

    # compute shares
    grp = dfc.groupby(["ClientCountry", "Category"]).size().rename("n").reset_index()
    grp["share"] = grp.groupby("ClientCountry")["n"].transform(lambda s: s / s.sum())

    fig = px.bar(
        grp,
        x="ClientCountry",
        y="share",
        color="Category",
        title="Category share of recommendations by country",
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# Plot by segment
st.subheader("Recommended category share by segment")
if "ClientSegment" not in long_recs.columns:
    st.info("ClientSegment not available in clients.csv.")
else:
    dfs = long_recs.dropna(subset=["ClientSegment", "Category"])
    dfs = topn_other(dfs, "ClientSegment", "Category", top_n=top_n)

    grp = dfs.groupby(["ClientSegment", "Category"]).size().rename("n").reset_index()
    grp["share"] = grp.groupby("ClientSegment")["n"].transform(lambda s: s / s.sum())

    fig = px.bar(
        grp,
        x="ClientSegment",
        y="share",
        color="Category",
        title="Category share of recommendations by segment",
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown(
    """
### Conclusion
If category mixes differ across **countries** and **segments**, then one global list is inefficient.
A churn-focused recommender should be **context-aware**, so each client sees items aligned with their group patterns
and personal history—maximizing the chance of a repeat purchase.
"""
)
