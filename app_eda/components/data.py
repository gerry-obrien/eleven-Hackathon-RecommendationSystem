# app_eda/components/data.py
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class DataBundle:
    clients: pd.DataFrame
    products: pd.DataFrame
    stocks: pd.DataFrame
    stores: pd.DataFrame
    transactions: pd.DataFrame


@st.cache_data(show_spinner=True)
def load_data(raw_dir: Path) -> DataBundle:
    """
    Cached CSV loader for the EDA app.
    raw_dir should point to data/raw.
    """
    clients = pd.read_csv(raw_dir / "clients.csv")
    products = pd.read_csv(raw_dir / "products.csv")
    stocks = pd.read_csv(raw_dir / "stocks.csv")
    stores = pd.read_csv(raw_dir / "stores.csv")
    transactions = pd.read_csv(
        raw_dir / "transactions.csv",
        parse_dates=["SaleTransactionDate"],
    )
    return DataBundle(clients=clients, products=products, stocks=stocks, stores=stores, transactions=transactions)


@st.cache_data(show_spinner=False)
def enrich_transactions(
    transactions: pd.DataFrame,
    clients: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cached enrichment: transactions + store country + client attrs + product taxonomy.
    """
    tx = transactions.merge(stores, on="StoreID", how="left")
    tx = tx.merge(clients, on="ClientID", how="left")
    tx = tx.merge(products, on="ProductID", how="left")
    return tx


@st.cache_data(show_spinner=False)
def stock_map(stocks: pd.DataFrame) -> dict[str, set]:
    """
    stock_map[country] = set(product_ids in stock (Quantity > 0))
    """
    s = stocks[stocks["Quantity"] > 0].copy()
    out = {}
    for c, g in s.groupby("StoreCountry"):
        out[str(c)] = set(g["ProductID"].unique())
    return out
