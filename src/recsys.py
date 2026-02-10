import pandas as pd

def popularity_recs(train_tx: pd.DataFrame, k: int = 10) -> dict:
    """Recommend the globally most purchased products to everyone."""
    top_products = (
        train_tx["ProductID"]
        .value_counts()
        .head(k)
        .index
        .tolist()
    )
    client_ids = train_tx["ClientID"].unique()
    return {cid: top_products for cid in client_ids}

def build_stock_map(stocks: pd.DataFrame) -> dict[str, set]:
    """stock_map[country] = set(product_ids with Quantity > 0)."""
    s = stocks[stocks["Quantity"] > 0]
    return {str(c): set(g["ProductID"].unique()) for c, g in s.groupby("StoreCountry")}

def filter_recs_by_stock(
    recs: dict,
    client_country: pd.Series,
    stock_map: dict[str, set],
) -> dict:
    """Filter each client's rec list to products in-stock in their country (if stock exists)."""
    filtered = {}
    for cid, items in recs.items():
        country = client_country.get(cid)
        if pd.isna(country):
            filtered[cid] = items  # can't determine, keep as-is
            continue
        allowed = stock_map.get(str(country))
        if allowed is None:
            filtered[cid] = items  # no stock info for that country, keep as-is
            continue
        filtered[cid] = [pid for pid in items if pid in allowed]
    return filtered
