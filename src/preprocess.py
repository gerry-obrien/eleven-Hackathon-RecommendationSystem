import pandas as pd

def enrich_transactions(transactions: pd.DataFrame,
                        clients: pd.DataFrame,
                        products: pd.DataFrame,
                        stores: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a transaction table enriched with:
      - client attributes (segment, country, etc.)
      - product taxonomy
      - store country
    """
    tx = transactions.merge(stores, on="StoreID", how="left")
    tx = tx.merge(clients, on="ClientID", how="left")
    tx = tx.merge(products, on="ProductID", how="left")
    return tx
