from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class EDASummary:
    n_transactions: int
    n_clients: int
    n_products: int
    n_stores: int
    date_min: str
    date_max: str


def basic_overview(transactions: pd.DataFrame) -> EDASummary:
    tx = transactions
    return EDASummary(
        n_transactions=int(len(tx)),
        n_clients=int(tx["ClientID"].nunique()),
        n_products=int(tx["ProductID"].nunique()),
        n_stores=int(tx["StoreID"].nunique()),
        date_min=str(tx["SaleTransactionDate"].min()),
        date_max=str(tx["SaleTransactionDate"].max()),
    )


def describe_series(s: pd.Series) -> pd.DataFrame:
    """Deck-friendly describe table including percentiles."""
    desc = s.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    return desc.to_frame(name="value")


def top_n_by_group(df: pd.DataFrame, group_col: str, item_col: str, n: int = 20) -> pd.DataFrame:
    """
    For each group (e.g., country/segment), return top-N items by count.
    Output: group_col, item_col, count, rank
    """
    counts = (
        df.groupby([group_col, item_col])
        .size()
        .rename("count")
        .reset_index()
        .sort_values([group_col, "count"], ascending=[True, False])
    )
    counts["rank"] = counts.groupby(group_col)["count"].rank(method="first", ascending=False).astype(int)
    return counts[counts["rank"] <= n].copy()


def next_item_transitions(transactions: pd.DataFrame, item_col: str) -> pd.DataFrame:
    """
    Compute next-item transitions per client by time order.
    Returns: item, next_item, count
    """
    tx = transactions.sort_values(["ClientID", "SaleTransactionDate"])
    cur_item = tx.groupby("ClientID")[item_col].shift(0)
    next_item = tx.groupby("ClientID")[item_col].shift(-1)

    pairs = pd.DataFrame({"item": cur_item, "next_item": next_item}).dropna()

    out = (
        pairs.groupby(["item", "next_item"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return out
