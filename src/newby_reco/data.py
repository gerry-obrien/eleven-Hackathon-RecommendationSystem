# src/recommender/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.load_data import load_all


@dataclass
class RecoData:
    clients: pd.DataFrame
    transactions: pd.DataFrame
    products: pd.DataFrame
    stocks: pd.DataFrame
    stores: pd.DataFrame

    existing_clients: pd.DataFrame   # clients with >=1 transaction (only needed cols)
    new_clients: pd.DataFrame        # clients with 0 transaction (only needed cols)

    client_products: pd.Series       # index=ClientID -> list(ProductID)
    stock_map: dict[str, set]        # country -> set(ProductID in stock)
    client_country: pd.Series        # index=ClientID -> country


CLIENT_FEATURE_COLS_WITH_ID = [
    "ClientID",
    "ClientCountry",
    "ClientOptINEmail",
    "ClientOptINPhone",
    "ClientGender",
    "Age",
]

CLIENT_FEATURE_COLS = [
    "ClientCountry",
    "ClientOptINEmail",
    "ClientOptINPhone",
    "ClientGender",
    "Age",
]


def load_reco_data(raw_dir: Path) -> RecoData:
    data = load_all(raw_dir)

    clients = data.clients.copy()
    transactions = data.transactions.copy()
    products = data.products.copy()
    stocks = data.stocks.copy()
    stores = data.stores.copy()

    # clientIDs with at least one transaction
    clients_with_tx = transactions["ClientID"].dropna().unique()

    client_filtered = clients[CLIENT_FEATURE_COLS_WITH_ID].copy()

    existing_clients = client_filtered[client_filtered["ClientID"].isin(clients_with_tx)].copy()
    new_clients = client_filtered[~client_filtered["ClientID"].isin(clients_with_tx)].copy()

    # list of purchased products per client
    client_products = (
        transactions.groupby("ClientID")["ProductID"]
        .apply(list)
    )

    # stock_map[country] = set(product_ids with Quantity > 0)
    stock_map = (
        stocks.loc[stocks["Quantity"] > 0]
        .groupby("StoreCountry")["ProductID"]
        .apply(set)
        .to_dict()
    )

    # country per client
    client_country = clients.set_index("ClientID")["ClientCountry"]

    return RecoData(
        clients=clients,
        transactions=transactions,
        products=products,
        stocks=stocks,
        stores=stores,
        existing_clients=existing_clients,
        new_clients=new_clients,
        client_products=client_products,
        stock_map=stock_map,
        client_country=client_country,
    )
