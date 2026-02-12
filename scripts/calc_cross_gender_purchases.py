from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CLIENT_GENDER_MAP = {
    "M": "Male",
    "MALE": "Male",
    "F": "Female",
    "FEMALE": "Female",
}

PRODUCT_GENDER_MAP = {
    "MEN": "Male",
    "MAN": "Male",
    "MALE": "Male",
    "WOMEN": "Female",
    "WOMAN": "Female",
    "FEMALE": "Female",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate cross-gender purchasing patterns from raw CSV files."
    )
    parser.add_argument(
        "--transactions-path",
        type=Path,
        default=Path("data/raw/transactions.csv"),
        help="Path to transactions.csv",
    )
    parser.add_argument(
        "--products-path",
        type=Path,
        default=Path("data/raw/products.csv"),
        help="Path to products.csv",
    )
    parser.add_argument(
        "--clients-path",
        type=Path,
        default=Path("data/raw/clients.csv"),
        help="Path to clients.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for calculation outputs.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, expected: list[str], table_name: str) -> None:
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def normalize_client_gender(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().map(CLIENT_GENDER_MAP)


def normalize_product_gender(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().map(PRODUCT_GENDER_MAP)


def calculate_cross_gender_metrics(
    transactions: pd.DataFrame, products: pd.DataFrame, clients: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    require_columns(transactions, ["ClientID", "ProductID", "Quantity", "SalesNetAmountEuro"], "transactions.csv")
    require_columns(products, ["ProductID", "Universe"], "products.csv")
    require_columns(clients, ["ClientID", "ClientGender"], "clients.csv")

    tx = transactions[["ClientID", "ProductID", "Quantity", "SalesNetAmountEuro"]].copy()
    prod = products[["ProductID", "Universe"]].copy()
    cli = clients[["ClientID", "ClientGender"]].copy()

    prod["product_gender"] = normalize_product_gender(prod["Universe"])
    cli["client_gender"] = normalize_client_gender(cli["ClientGender"])

    merged = (
        tx.merge(cli[["ClientID", "client_gender"]], on="ClientID", how="left")
        .merge(prod[["ProductID", "product_gender"]], on="ProductID", how="left")
    )

    known = merged[
        merged["client_gender"].isin(["Male", "Female"])
        & merged["product_gender"].isin(["Male", "Female"])
    ].copy()

    tx_matrix = (
        known.groupby(["client_gender", "product_gender"], dropna=False)
        .agg(
            transactions=("ProductID", "size"),
            quantity=("Quantity", "sum"),
            sales_eur=("SalesNetAmountEuro", "sum"),
        )
        .reset_index()
    )

    tx_matrix["cohort_transactions"] = tx_matrix.groupby("client_gender")["transactions"].transform("sum")
    tx_matrix["share_within_client_gender_pct"] = (
        tx_matrix["transactions"] / tx_matrix["cohort_transactions"] * 100
    ).round(2)
    tx_matrix["cross_gender"] = tx_matrix["client_gender"] != tx_matrix["product_gender"]

    client_product = known.drop_duplicates(subset=["ClientID", "client_gender", "product_gender"]).copy()
    client_counts = (
        client_product.groupby(["client_gender", "product_gender"], dropna=False)["ClientID"]
        .nunique()
        .reset_index(name="clients")
    )
    transacting_clients = (
        known.drop_duplicates(subset=["ClientID", "client_gender"])
        .groupby("client_gender")["ClientID"]
        .nunique()
        .rename("cohort_clients")
        .reset_index()
    )

    client_counts = client_counts.merge(transacting_clients, on="client_gender", how="left")
    client_counts["share_of_clients_pct"] = (
        client_counts["clients"] / client_counts["cohort_clients"] * 100
    ).round(2)
    client_counts["cross_gender"] = client_counts["client_gender"] != client_counts["product_gender"]

    diagnostics = {
        "total_transactions": int(len(merged)),
        "known_gender_transactions": int(len(known)),
        "unknown_or_unmapped_transactions": int(len(merged) - len(known)),
    }

    return tx_matrix, client_counts, diagnostics


def main() -> None:
    args = parse_args()

    transactions = pd.read_csv(args.transactions_path)
    products = pd.read_csv(args.products_path)
    clients = pd.read_csv(args.clients_path)

    tx_summary, client_summary, diagnostics = calculate_cross_gender_metrics(
        transactions=transactions,
        products=products,
        clients=clients,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tx_path = args.out_dir / "cross_gender_transactions_summary.csv"
    client_path = args.out_dir / "cross_gender_client_incidence_summary.csv"

    tx_summary.to_csv(tx_path, index=False)
    client_summary.to_csv(client_path, index=False)

    male_to_female_tx = tx_summary[
        (tx_summary["client_gender"] == "Male") & (tx_summary["product_gender"] == "Female")
    ]
    female_to_male_tx = tx_summary[
        (tx_summary["client_gender"] == "Female") & (tx_summary["product_gender"] == "Male")
    ]

    male_to_female_clients = client_summary[
        (client_summary["client_gender"] == "Male") & (client_summary["product_gender"] == "Female")
    ]
    female_to_male_clients = client_summary[
        (client_summary["client_gender"] == "Female") & (client_summary["product_gender"] == "Male")
    ]

    print(f"Saved: {tx_path.resolve()}")
    print(f"Saved: {client_path.resolve()}")
    print()
    print("Cross-gender purchase frequency (transactions):")
    if not male_to_female_tx.empty:
        row = male_to_female_tx.iloc[0]
        print(
            f"Male -> Female products: {int(row['transactions']):,} tx "
            f"({row['share_within_client_gender_pct']:.2f}% of male-client transactions)"
        )
    else:
        print("Male -> Female products: no matched rows")
    if not female_to_male_tx.empty:
        row = female_to_male_tx.iloc[0]
        print(
            f"Female -> Male products: {int(row['transactions']):,} tx "
            f"({row['share_within_client_gender_pct']:.2f}% of female-client transactions)"
        )
    else:
        print("Female -> Male products: no matched rows")

    print()
    print("Cross-gender incidence (unique clients):")
    if not male_to_female_clients.empty:
        row = male_to_female_clients.iloc[0]
        print(
            f"Male clients buying >=1 Female product: {int(row['clients']):,} "
            f"({row['share_of_clients_pct']:.2f}% of transacting male clients)"
        )
    else:
        print("Male clients buying >=1 Female product: no matched rows")
    if not female_to_male_clients.empty:
        row = female_to_male_clients.iloc[0]
        print(
            f"Female clients buying >=1 Male product: {int(row['clients']):,} "
            f"({row['share_of_clients_pct']:.2f}% of transacting female clients)"
        )
    else:
        print("Female clients buying >=1 Male product: no matched rows")

    print()
    print("Diagnostics:")
    print(f"Total transactions loaded: {diagnostics['total_transactions']:,}")
    print(f"Transactions with mapped client+product gender: {diagnostics['known_gender_transactions']:,}")
    print(f"Transactions excluded (unknown/unmapped): {diagnostics['unknown_or_unmapped_transactions']:,}")


if __name__ == "__main__":
    main()
