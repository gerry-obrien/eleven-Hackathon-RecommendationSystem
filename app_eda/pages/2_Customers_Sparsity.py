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

st.title("Customers & Sparsity")
st.markdown(
    """
**Purpose:** quantify how much history customers have.  
This directly informs the recommender strategy:
- many low-history customers → need strong **cold-start fallbacks** (popular in country/segment)
- richer histories → enable **personalized ranking / sequence models**
"""
)

data = load_data(repo_root / "data" / "raw")
txe = enrich_transactions(data.transactions, data.clients, data.products, data.stores)

filters = render_sidebar_filters(txe)
txf = apply_filters(txe, filters)

per_client = txf.groupby("ClientID").size()

c1, c2, c3 = st.columns(3)
c1.metric("Median tx / client", int(per_client.median()) if len(per_client) else 0)
c2.metric("Mean tx / client", float(per_client.mean()) if len(per_client) else 0.0)
c3.metric("Clients (filtered)", txf["ClientID"].nunique())

st.plotly_chart(hist(per_client.clip(upper=50), "Transactions per client (clipped at 50)"), use_container_width=True)

st.subheader("Cold-start buckets")
bins = [-1, 1, 2, 5, 10, 999999]
labels = ["1", "2", "3–5", "6–10", "10+"]
bucket = pd.cut(per_client, bins=bins, labels=labels)
bucket_table = bucket.value_counts(normalize=True).sort_index().rename("share").to_frame()
bucket_table["share_pct"] = (bucket_table["share"] * 100).round(2)
st.dataframe(bucket_table, use_container_width=True)

st.subheader("Opt-in reachability (clients)")
c = data.clients
if "ClientOptINEmail" in c.columns and "ClientOptINPhone" in c.columns:
    opt = pd.DataFrame(
        {
            "Email opt-in rate": [float(c["ClientOptINEmail"].mean())],
            "Phone opt-in rate": [float(c["ClientOptINPhone"].mean())],
        }
    )
    st.dataframe(opt, use_container_width=True)
else:
    st.info("Opt-in columns not found in clients table.")

st.caption("Business value: cold-start share tells us how much value comes from generic vs personalized approaches, and opt-in rates define reachable marketing volume.")


st.divider()

st.markdown(
    """
### Conclusion
Customer history is sparse: the median customer has only **2 transactions**, and **~65%** have **1–2** purchases.  
This means “cold start” is the *default*, not the exception—so the recommender should be **hybrid/tiered**:
- **1–2 transactions:** popularity (by country/segment) + simple item-to-item similarity
- **3–5 transactions:** stronger “customers also bought” personalization
- **6+ transactions:** fully personalized ranking approaches

Opt-in rates (~**44% email**, **40% phone**) also cap how many customers we can reach via CRM activation, so impact should be reported by **history bucket** and **reachable volume**.
"""
)