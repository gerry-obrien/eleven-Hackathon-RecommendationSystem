import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Dict, Set, Tuple, Optional

from app_eda.components.data import load_data, enrich_transactions


# ---------- Paths ----------
def get_repo_root() -> Path:
    # app_demo/components/demo_data.py -> repo root
    return Path(__file__).resolve().parents[2]


def demo_csv_path() -> Path:
    return get_repo_root() / "data" / "demo_test" / "ui_client_recommendations_sample.csv"


def raw_dir_path() -> Path:
    return get_repo_root() / "data" / "raw"


# ---------- Loaders ----------
@st.cache_data
def load_private_data():
    """
    Loads private raw data from data/raw using your existing loader.
    Returns the same object structure as app_eda uses (data.clients, data.products, etc.).
    """
    return load_data(raw_dir_path())


@st.cache_data
def load_enriched_transactions():
    data = load_private_data()
    txe = enrich_transactions(data.transactions, data.clients, data.products, data.stores)
    return txe


@st.cache_data
def load_demo_recs() -> pd.DataFrame:
    path = demo_csv_path()
    df = pd.read_csv(path)
    df["ClientID"] = df["ClientID"].astype(str)
    return df


REC_COLS = [f"rec_{i}" for i in range(1, 11)]


def get_client_rec_ids(demo_recs: pd.DataFrame, client_id: str, k: int = 10) -> list:
    row = demo_recs.loc[demo_recs["ClientID"] == str(client_id)].iloc[0]
    rec_ids = [row[c] for c in REC_COLS][:k]
    return rec_ids


@st.cache_data
def dataset_today() -> pd.Timestamp:
    """
    Use the most recent transaction date in the dataset as "today" for demo purposes.
    This prevents days-since-last being inflated when the dataset is historical.
    """
    txe = load_enriched_transactions()
    date_col = _safe_col(txe, ["SaleTransactionDate", "Date", "TransactionDate"])
    if date_col is None:
        return pd.Timestamp.today().normalize()

    d = pd.to_datetime(txe[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    d = d.dropna()
    if d.empty:
        return pd.Timestamp.today().normalize()

    return d.max().normalize()


# ---------- Metadata helpers ----------
def _safe_col(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


@st.cache_data
def build_product_lookup() -> pd.DataFrame:
    data = load_private_data()
    p = data.products.copy()

    # Your products.csv has these columns (no ProductName):
    # ProductID, Category, FamilyLevel1, FamilyLevel2, Universe
    keep = ["ProductID"]
    for c in ["Universe", "FamilyLevel1", "FamilyLevel2", "Category"]:
        if c in p.columns:
            keep.append(c)

    out = p[keep].copy()

    # Create a UI-friendly label (acts like "product name")
    # Example: "Universe > FamilyLevel1 > FamilyLevel2"
    label_parts = [c for c in ["Universe", "FamilyLevel1", "FamilyLevel2"] if c in out.columns]
    if label_parts:
        out["ProductLabel"] = (
            out[label_parts]
            .astype(str)
            .agg(" > ".join, axis=1)
        )
    else:
        out["ProductLabel"] = out["ProductID"].astype(str)

    return out



def get_client_row(client_id: int) -> pd.DataFrame:
    data = load_private_data()
    return data.clients[data.clients["ClientID"] == client_id].head(1)


def get_recent_purchases(client_id: int, n: int = 5) -> pd.DataFrame:
    txe = load_enriched_transactions()
    t = txe[txe["ClientID"] == client_id].copy()

    date_col = _safe_col(t, ["SaleTransactionDate", "Date", "TransactionDate"])
    if date_col is not None:
        t[date_col] = pd.to_datetime(t[date_col], errors="coerce", utc=True).dt.tz_convert(None)
        t = t.sort_values(date_col, ascending=False)

    # Add product name if available
    prod_lu = build_product_lookup()
    t = t.merge(prod_lu, on="ProductID", how="left")

    # Choose display columns
    show_cols = []
    for c in [date_col, "ProductID", "ProductLabel", "Category", "SalesNetAmountEuro", "Quantity"]:
        if c and c in t.columns and c not in show_cols:
            show_cols.append(c)

    return t[show_cols].head(n)


def get_days_since_last_purchase(client_id: int) -> Optional[int]:
    txe = load_enriched_transactions()
    t = txe[txe["ClientID"] == client_id].copy()
    date_col = _safe_col(t, ["SaleTransactionDate", "Date", "TransactionDate"])
    if date_col is None:
        return None

    t[date_col] = pd.to_datetime(t[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    t = t.dropna(subset=[date_col])
    if t.empty:
        return None

    last_dt = t[date_col].max().normalize()
    today = dataset_today()
    return int((today - last_dt).days)




# ---------- Stock map + stock checks ----------
@st.cache_data
def stock_map() -> Dict[str, Set[int]]:
    """
    Returns {CountryCode: set(ProductID)} for items with positive stock quantity if possible.
    Robust to different column names.
    """
    data = load_private_data()
    s = data.stocks.copy()

    country_col = _safe_col(s, ["Country", "StoreCountry", "ClientCountry"])
    pid_col = _safe_col(s, ["ProductID", "product_id", "SKU"])
    qty_col = _safe_col(s, ["Quantity", "Qty", "StockQuantity", "OnHand", "stock"])

    if country_col is None or pid_col is None:
        return {}

    s[country_col] = s[country_col].astype(str)
    s[pid_col] = pd.to_numeric(s[pid_col], errors="coerce")

    if qty_col is not None:
        s[qty_col] = pd.to_numeric(s[qty_col], errors="coerce").fillna(0)
        s = s[s[qty_col] > 0]

    smap: Dict[str, Set[int]] = {}
    for ctry, grp in s.dropna(subset=[pid_col]).groupby(country_col):
        smap[str(ctry)] = set(grp[pid_col].astype(int).tolist())

    return smap


def annotate_stock(rec_df: pd.DataFrame, client_country: Optional[str]) -> pd.DataFrame:
    """
    Adds a StockStatus column: In stock / Out of stock / Unknown (no coverage).
    """
    out = rec_df.copy()
    smap = stock_map()

    if not client_country:
        out["StockStatus"] = "Unknown"
        return out

    allowed = smap.get(str(client_country))
    if allowed is None:
        out["StockStatus"] = "Unknown"
        return out

    out["StockStatus"] = out["ProductID"].apply(lambda pid: "In stock" if int(pid) in allowed else "Out of stock")
    return out


# ---------- Risk label (simple demo rule) ----------
@st.cache_data
def segment_gap_thresholds() -> pd.DataFrame:
    """
    Computes segment-level p75 of interpurchase gap (days) from historical transactions.
    Used to define a simple "late / at risk" threshold.
    """
    txe = load_enriched_transactions()
    if "ClientID" not in txe.columns:
        return pd.DataFrame(columns=["ClientSegment", "p75_gap_days"])

    seg_col = "ClientSegment" if "ClientSegment" in txe.columns else None
    date_col = _safe_col(txe, ["SaleTransactionDate", "Date", "TransactionDate"])
    if seg_col is None or date_col is None:
        return pd.DataFrame(columns=["ClientSegment", "p75_gap_days"])

    t = txe[["ClientID", seg_col, date_col]].copy()
    t[date_col] = pd.to_datetime(t[date_col], errors="coerce")
    t = t.dropna(subset=["ClientID", seg_col, date_col]).sort_values(["ClientID", date_col])

    t["prev_date"] = t.groupby("ClientID")[date_col].shift(1)
    t["gap_days"] = (t[date_col] - t["prev_date"]).dt.days
    gaps = t.dropna(subset=["gap_days"])
    gaps = gaps[gaps["gap_days"] >= 0]

    if gaps.empty:
        return pd.DataFrame(columns=["ClientSegment", "p75_gap_days"])

    out = (
        gaps.groupby(seg_col)["gap_days"]
        .quantile(0.75)
        .rename("p75_gap_days")
        .reset_index()
        .rename(columns={seg_col: "ClientSegment"})
    )
    return out


def risk_label(days_since_last: Optional[int], segment: Optional[str]) -> Tuple[str, str]:
    """
    Returns (label, help_text). Very simple for demo.
    """
    if days_since_last is None or not segment:
        return ("Unknown", "Insufficient data to compute risk label.")

    thr = segment_gap_thresholds()
    row = thr[thr["ClientSegment"] == segment]
    if row.empty:
        # fallback heuristic
        if days_since_last <= 7:
            return ("On track", "Recently active.")
        if days_since_last <= 30:
            return ("Getting late", "No purchase recently.")
        return ("At risk", "Long time since last purchase.")
    p75 = float(row["p75_gap_days"].iloc[0])

    if days_since_last <= p75:
        return ("On track", f"Within typical repeat window for {segment} (p75 ≈ {p75:.0f} days).")
    if days_since_last <= max(p75 * 2, p75 + 14):
        return ("Getting late", f"Past typical window for {segment}; intervention may help.")
    return ("At risk", f"Well past typical window for {segment}; churn risk higher.")
