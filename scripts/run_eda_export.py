from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.load_data import load_all
from src.preprocess import enrich_transactions
from src.eda import basic_overview, describe_series, top_n_by_group, next_item_transitions

EDA_DIR = Path("eda_outputs")
PLOTS_DIR = EDA_DIR / "plots"
TABLES_DIR = EDA_DIR / "tables"


def save_hist(series: pd.Series, title: str, xlabel: str, outpath: Path, clip_upper: int | None = None) -> None:
    s = series.copy()
    if clip_upper is not None:
        s = s.clip(upper=clip_upper)

    plt.figure()
    s.hist(bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    data = load_all()
    tx = data.transactions

    # 0) Basic overview
    overview = basic_overview(tx)
    pd.DataFrame([overview.__dict__]).to_csv(TABLES_DIR / "overview.csv", index=False)

    # 1) Sparsity: purchases per client / product
    per_client = tx.groupby("ClientID").size()
    describe_series(per_client).to_csv(TABLES_DIR / "transactions_per_client_describe.csv")
    save_hist(
        per_client,
        title="Transactions per client (clipped at 50)",
        xlabel="# transactions",
        outpath=PLOTS_DIR / "transactions_per_client_hist.png",
        clip_upper=50,
    )

    per_product = tx.groupby("ProductID").size()
    describe_series(per_product).to_csv(TABLES_DIR / "transactions_per_product_describe.csv")
    save_hist(
        per_product,
        title="Transactions per product (clipped at 200)",
        xlabel="# transactions",
        outpath=PLOTS_DIR / "transactions_per_product_hist.png",
        clip_upper=200,
    )

    # 2) Stock coverage by country
    stocks = data.stocks.copy()
    stock_cov = (
        stocks[stocks["Quantity"] > 0]
        .groupby("StoreCountry")["ProductID"]
        .nunique()
        .rename("products_with_stock")
        .reset_index()
        .sort_values("products_with_stock", ascending=False)
    )
    stock_cov.to_csv(TABLES_DIR / "stock_coverage_by_country.csv", index=False)

    # 3) Enrich transactions for category/segment analysis
    txe = enrich_transactions(data.transactions, data.clients, data.products, data.stores)

    # Top categories by client country / segment (if columns exist)
    if "ClientCountry" in txe.columns and "Category" in txe.columns:
        top_cat_country = top_n_by_group(txe, "ClientCountry", "Category", n=20)
        top_cat_country.to_csv(TABLES_DIR / "top_categories_by_client_country.csv", index=False)

    if "ClientSegment" in txe.columns and "Category" in txe.columns:
        top_cat_seg = top_n_by_group(txe, "ClientSegment", "Category", n=20)
        top_cat_seg.to_csv(TABLES_DIR / "top_categories_by_client_segment.csv", index=False)

    # 4) Next-purchase patterns: category transitions
    if "Category" in txe.columns:
        trans = next_item_transitions(txe.dropna(subset=["Category"]), item_col="Category")
        trans.head(200).to_csv(TABLES_DIR / "top_category_transitions.csv", index=False)

    print("✅ EDA exports written to:")
    print(f"  - {PLOTS_DIR.resolve()}")
    print(f"  - {TABLES_DIR.resolve()}")


if __name__ == "__main__":
    main()
