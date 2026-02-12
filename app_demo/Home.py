import sys
from pathlib import Path
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

st.set_page_config(page_title="Churn Reduction Demo", layout="wide")

st.title("Churn Reduction Recommendation Prototype")
st.markdown(
    """
Demo App: it shows how a recommendation system can reduce churn by
serving **personalized**, **country/segment-aware**, and optionally **in-stock** recommendations.

Use the pages in the left sidebar:
1. **Client Demo** — the main live demo
2. **Country / Segment Distribution** — shows recommendations differ across groups
3. **Business Value** — simple churn-focused KPIs + scenario calculator
"""
)

st.info(
    "Data source: private CSVs in `data/raw/` (gitignored) + demo recommendations in `data/demo_test/`."
)
