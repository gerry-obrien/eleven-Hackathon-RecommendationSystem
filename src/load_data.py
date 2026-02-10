from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .config import RAW_DIR


@dataclass(frozen=True)
class DataBundle:
    clients: pd.DataFrame
    products: pd.DataFrame
    stocks: pd.DataFrame
    stores: pd.DataFrame
    transactions: pd.DataFrame


def load_all(
    raw_dir=RAW_DIR,
    parse_dates: bool = True,
) -> DataBundle:
    """
    Load all hackathon CSVs from data/raw/.

    Expected files:
      - clients.csv
      - products.csv
      - stocks.csv
      - stores.csv
      - transactions.csv

    Returns a DataBundle of pandas DataFrames.
    """
    clients = pd.read_csv(raw_dir / "clients.csv")
    products = pd.read_csv(raw_dir / "products.csv")
    stocks = pd.read_csv(raw_dir / "stocks.csv")
    stores = pd.read_csv(raw_dir / "stores.csv")

    if parse_dates:
        transactions = pd.read_csv(
            raw_dir / "transactions.csv",
            parse_dates=["SaleTransactionDate"],
        )
    else:
        transactions = pd.read_csv(raw_dir / "transactions.csv")

    return DataBundle(
        clients=clients,
        products=products,
        stocks=stocks,
        stores=stores,
        transactions=transactions,
    )


def smoke_test() -> None:
    """Quick check: load data and print shapes + key columns presence."""
    data = load_all()
    print("Loaded:")
    print("  clients      ", data.clients.shape)
    print("  products     ", data.products.shape)
    print("  stocks       ", data.stocks.shape)
    print("  stores       ", data.stores.shape)
    print("  transactions ", data.transactions.shape)

    required_tx_cols = {
        "ClientID",
        "ProductID",
        "SaleTransactionDate",
        "StoreID",
        "Quantity",
        "SalesNetAmountEuro",
    }
    missing = required_tx_cols - set(data.transactions.columns)
    if missing:
        raise ValueError(f"transactions.csv missing columns: {sorted(missing)}")

    print("Smoke test OK")
