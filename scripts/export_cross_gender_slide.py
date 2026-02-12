from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


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
        description="Export a presentation-ready slide for cross-gender purchasing insights."
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
        "--out-path",
        type=Path,
        default=Path("plots/cross_gender_purchase_insights_exec.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=16.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=9.0,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="Output DPI.",
    )
    parser.add_argument(
        "--top-n-categories",
        type=int,
        default=7,
        help="Top N categories per cross-gender direction in the bottom panels.",
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


def extract_cross_gender_dataset(
    transactions_path: Path, products_path: Path, clients_path: Path
) -> tuple[pd.DataFrame, dict[str, float]]:
    transactions = pd.read_csv(
        transactions_path, usecols=["ClientID", "ProductID", "Quantity", "SalesNetAmountEuro"]
    )
    products = pd.read_csv(products_path, usecols=["ProductID", "Universe", "Category"])
    clients = pd.read_csv(clients_path, usecols=["ClientID", "ClientGender", "ClientCountry"])

    require_columns(transactions, ["ClientID", "ProductID", "Quantity", "SalesNetAmountEuro"], "transactions.csv")
    require_columns(products, ["ProductID", "Universe", "Category"], "products.csv")
    require_columns(clients, ["ClientID", "ClientGender", "ClientCountry"], "clients.csv")

    products["product_gender"] = normalize_product_gender(products["Universe"])
    clients["client_gender"] = normalize_client_gender(clients["ClientGender"])

    merged = (
        transactions.merge(clients[["ClientID", "client_gender", "ClientCountry"]], on="ClientID", how="left")
        .merge(products[["ProductID", "product_gender", "Category"]], on="ProductID", how="left")
    )

    known = merged[
        merged["client_gender"].isin(["Male", "Female"])
        & merged["product_gender"].isin(["Male", "Female"])
    ].copy()
    known["is_cross_gender"] = known["client_gender"] != known["product_gender"]

    diagnostics = {
        "total_transactions": float(len(merged)),
        "mapped_transactions": float(len(known)),
        "mapped_share_pct": float((len(known) / len(merged) * 100) if len(merged) else 0.0),
        "cross_tx_share_pct": float(known["is_cross_gender"].mean() * 100 if len(known) else 0.0),
        "cross_sales_share_pct": float(
            (known.loc[known["is_cross_gender"], "SalesNetAmountEuro"].sum() / known["SalesNetAmountEuro"].sum() * 100)
            if len(known) and known["SalesNetAmountEuro"].sum() > 0
            else 0.0
        ),
        "cross_quantity_share_pct": float(
            (known.loc[known["is_cross_gender"], "Quantity"].sum() / known["Quantity"].sum() * 100)
            if len(known) and known["Quantity"].sum() > 0
            else 0.0
        ),
    }

    return known, diagnostics


def build_client_incidence(known: pd.DataFrame) -> pd.DataFrame:
    client_product = known.drop_duplicates(subset=["ClientID", "client_gender", "product_gender"]).copy()
    counts = (
        client_product.groupby(["client_gender", "product_gender"])["ClientID"]
        .nunique()
        .reset_index(name="clients")
    )
    cohorts = (
        known.drop_duplicates(subset=["ClientID", "client_gender"])
        .groupby("client_gender")["ClientID"]
        .nunique()
        .rename("cohort_clients")
        .reset_index()
    )
    out = counts.merge(cohorts, on="client_gender", how="left")
    out["share_of_clients_pct"] = np.where(
        out["cohort_clients"] > 0, out["clients"] / out["cohort_clients"] * 100, 0.0
    )
    return out


def get_scalar(
    df: pd.DataFrame,
    client_gender: str,
    product_gender: str,
    value_col: str,
) -> float:
    row = df[(df["client_gender"] == client_gender) & (df["product_gender"] == product_gender)]
    if row.empty:
        return 0.0
    return float(row.iloc[0][value_col])


def export_cross_gender_slide(
    known: pd.DataFrame,
    diagnostics: dict[str, float],
    out_path: Path,
    fig_width: float,
    fig_height: float,
    dpi: int,
    top_n_categories: int,
) -> dict[str, float]:
    tx_matrix = (
        known.groupby(["client_gender", "product_gender"])
        .agg(
            transactions=("ProductID", "size"),
            quantity=("Quantity", "sum"),
            sales_eur=("SalesNetAmountEuro", "sum"),
        )
        .reset_index()
    )
    tx_matrix["cohort_transactions"] = tx_matrix.groupby("client_gender")["transactions"].transform("sum")
    tx_matrix["row_share_pct"] = np.where(
        tx_matrix["cohort_transactions"] > 0,
        tx_matrix["transactions"] / tx_matrix["cohort_transactions"] * 100,
        0.0,
    )

    incidence = build_client_incidence(known)

    male_to_female_tx_pct = get_scalar(tx_matrix, "Male", "Female", "row_share_pct")
    female_to_male_tx_pct = get_scalar(tx_matrix, "Female", "Male", "row_share_pct")
    male_to_female_client_pct = get_scalar(incidence, "Male", "Female", "share_of_clients_pct")
    female_to_male_client_pct = get_scalar(incidence, "Female", "Male", "share_of_clients_pct")

    matrix_plot = (
        tx_matrix.pivot(index="client_gender", columns="product_gender", values="transactions")
        .reindex(index=["Male", "Female"], columns=["Male", "Female"])
        .fillna(0.0)
    )
    matrix_row_share = (
        tx_matrix.pivot(index="client_gender", columns="product_gender", values="row_share_pct")
        .reindex(index=["Male", "Female"], columns=["Male", "Female"])
        .fillna(0.0)
    )

    country_cross = (
        known.dropna(subset=["ClientCountry"])
        .groupby("ClientCountry")
        .agg(
            transactions=("ProductID", "size"),
            cross_tx=("is_cross_gender", "sum"),
        )
        .reset_index()
    )
    country_cross["cross_tx_pct"] = np.where(
        country_cross["transactions"] > 0,
        country_cross["cross_tx"] / country_cross["transactions"] * 100,
        0.0,
    )
    country_cross = country_cross.sort_values("cross_tx_pct", ascending=False)

    cross_only = known[known["is_cross_gender"]].copy()
    cat_male_to_female = (
        cross_only[
            (cross_only["client_gender"] == "Male")
            & (cross_only["product_gender"] == "Female")
        ]
        .groupby("Category")
        .size()
        .sort_values(ascending=False)
        .head(top_n_categories)
    )
    cat_female_to_male = (
        cross_only[
            (cross_only["client_gender"] == "Female")
            & (cross_only["product_gender"] == "Male")
        ]
        .groupby("Category")
        .size()
        .sort_values(ascending=False)
        .head(top_n_categories)
    )

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor("white")

    # Use spacer columns between A/B/C to prevent axis-label collisions.
    gs = fig.add_gridspec(3, 14, height_ratios=[0.95, 2.45, 2.45], hspace=0.58, wspace=0.30)
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_a = fig.add_subplot(gs[1, 0:4])
    ax_b = fig.add_subplot(gs[1, 5:9])
    ax_c = fig.add_subplot(gs[1, 10:14])
    ax_d = fig.add_subplot(gs[2, 0:7])
    ax_e = fig.add_subplot(gs[2, 7:14])

    fig.suptitle("Cross-Gender Purchase Behavior: recommendation implications", fontsize=24, y=0.985)
    fig.text(
        0.5,
        0.946,
        "Joined from transactions + products (Universe) + clients (ClientGender) to quantify cross-gender purchasing.",
        ha="center",
        va="top",
        fontsize=12.5,
        color="#2f2f2f",
    )

    ax_kpi.axis("off")
    kpis = [
        ("Transactions analyzed", f"{int(diagnostics['mapped_transactions']):,} / {int(diagnostics['total_transactions']):,}"),
        ("Mapped coverage", f"{diagnostics['mapped_share_pct']:.1f}%"),
        ("Cross-gender tx share", f"{diagnostics['cross_tx_share_pct']:.2f}%"),
        ("Male -> Female tx share", f"{male_to_female_tx_pct:.2f}%"),
        ("Female -> Male tx share", f"{female_to_male_tx_pct:.2f}%"),
    ]
    kpi_x = [0.01, 0.22, 0.38, 0.57, 0.77]
    kpi_w = [0.20, 0.14, 0.17, 0.18, 0.18]
    for (label, value), x, width in zip(kpis, kpi_x, kpi_w):
        card = FancyBboxPatch(
            (x, 0.12),
            width,
            0.74,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            transform=ax_kpi.transAxes,
            linewidth=1.0,
            edgecolor="#d7dfeb",
            facecolor="#f8fbff",
        )
        ax_kpi.add_patch(card)
        ax_kpi.text(x + 0.012, 0.78, label, transform=ax_kpi.transAxes, ha="left", va="top", fontsize=10.5, color="#3f4d68")
        ax_kpi.text(x + 0.012, 0.50, value, transform=ax_kpi.transAxes, ha="left", va="top", fontsize=13.5, color="#1f2f4f", fontweight="bold")

    ax_a.imshow(matrix_plot.values, cmap="Blues", aspect="auto")
    ax_a.set_title("A) Client gender x product gender matrix", fontsize=12.5, pad=8)
    ax_a.set_xticks([0, 1], labels=["Male products", "Female products"])
    ax_a.set_yticks([0, 1], labels=["Male clients", "Female clients"])
    ax_a.tick_params(axis="x", labelsize=10)
    ax_a.tick_params(axis="y", labelsize=10)
    for i in range(2):
        for j in range(2):
            count = int(matrix_plot.values[i, j])
            pct = matrix_row_share.values[i, j]
            txt_color = "white" if matrix_plot.values[i, j] > matrix_plot.values.max() * 0.55 else "#1f2f4f"
            ax_a.text(j, i, f"{count:,}\n{pct:.1f}%", ha="center", va="center", fontsize=10, color=txt_color)
    ax_a.text(
        0.02,
        -0.16,
        "Cell text shows tx count and row share (%)",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#4a4a4a",
    )

    inc_plot = incidence.copy()
    same_vals = [
        get_scalar(inc_plot, "Male", "Male", "share_of_clients_pct"),
        get_scalar(inc_plot, "Female", "Female", "share_of_clients_pct"),
    ]
    cross_vals = [male_to_female_client_pct, female_to_male_client_pct]
    x = np.arange(2)
    width = 0.34
    same_bars = ax_b.bar(x - width / 2, same_vals, width, label="Same-gender product", color="#5c8bc3")
    cross_bars = ax_b.bar(x + width / 2, cross_vals, width, label="Cross-gender product", color="#9ebee2")
    ax_b.set_title("B) Client incidence: same vs cross-gender buying", fontsize=12.5, pad=8)
    ax_b.set_xticks(x, ["Male clients", "Female clients"])
    ax_b.set_ylabel("")
    ax_b.set_ylim(0, 100)
    ax_b.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax_b.grid(axis="x", visible=False)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.tick_params(axis="both", labelsize=10)
    ax_b.legend(fontsize=9.5, loc="upper right")
    for bars in [same_bars, cross_bars]:
        for bar in bars:
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#2f2f2f",
            )

    ax_c.bar(country_cross["ClientCountry"], country_cross["cross_tx_pct"], color="#6f99cc", edgecolor="white")
    ax_c.set_title("C) Cross-gender transaction share by country", fontsize=12.5, pad=8)
    ax_c.set_ylabel("cross-gender %", fontsize=10.5, labelpad=3)
    ax_c.set_ylim(0, max(60.0, country_cross["cross_tx_pct"].max() + 8))
    ax_c.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax_c.grid(axis="x", visible=False)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.tick_params(axis="x", labelsize=10)
    ax_c.tick_params(axis="y", labelsize=10)
    for _, row in country_cross.iterrows():
        ax_c.text(
            row["ClientCountry"],
            row["cross_tx_pct"] + 0.5,
            f"{row['cross_tx_pct']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.8,
            color="#2f2f2f",
        )

    male_cat = cat_male_to_female.sort_values(ascending=True)
    ax_d.barh(male_cat.index.astype(str), male_cat.values, color="#7fa7d4")
    ax_d.set_title("D) Male -> Female: top cross-purchased categories", fontsize=12.5, pad=8)
    ax_d.set_xlabel("transactions", fontsize=10.5)
    ax_d.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    ax_d.grid(axis="y", visible=False)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.tick_params(axis="both", labelsize=10)

    female_cat = cat_female_to_male.sort_values(ascending=True)
    ax_e.barh(female_cat.index.astype(str), female_cat.values, color="#5c8bc3")
    ax_e.set_title("E) Female -> Male: top cross-purchased categories", fontsize=12.5, pad=8)
    ax_e.set_xlabel("transactions", fontsize=10.5)
    ax_e.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    ax_e.grid(axis="y", visible=False)
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)
    ax_e.tick_params(axis="both", labelsize=10)

    fig.text(
        0.01,
        0.012,
        (
            "Notes: Product gender is inferred from products.Universe (Men/Women). "
            f"Cross-gender sales share: {diagnostics['cross_sales_share_pct']:.2f}%, "
            f"quantity share: {diagnostics['cross_quantity_share_pct']:.2f}%."
        ),
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#4a4a4a",
    )

    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.08, hspace=0.58, wspace=0.30)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)

    return {
        "mapped_transactions": diagnostics["mapped_transactions"],
        "mapped_share_pct": diagnostics["mapped_share_pct"],
        "cross_tx_share_pct": diagnostics["cross_tx_share_pct"],
        "male_to_female_tx_pct": male_to_female_tx_pct,
        "female_to_male_tx_pct": female_to_male_tx_pct,
        "male_to_female_client_pct": male_to_female_client_pct,
        "female_to_male_client_pct": female_to_male_client_pct,
    }


def main() -> None:
    args = parse_args()
    if args.fig_width <= 0 or args.fig_height <= 0 or args.dpi <= 0:
        raise ValueError("Figure size and DPI must be positive.")
    if args.top_n_categories <= 0:
        raise ValueError("--top-n-categories must be positive.")

    known, diagnostics = extract_cross_gender_dataset(
        transactions_path=args.transactions_path,
        products_path=args.products_path,
        clients_path=args.clients_path,
    )

    stats = export_cross_gender_slide(
        known=known,
        diagnostics=diagnostics,
        out_path=args.out_path,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        dpi=args.dpi,
        top_n_categories=args.top_n_categories,
    )

    print(f"Saved slide: {args.out_path.resolve()}")
    print(f"Mapped transactions: {int(stats['mapped_transactions']):,} ({stats['mapped_share_pct']:.2f}% of total)")
    print(f"Cross-gender tx share: {stats['cross_tx_share_pct']:.2f}%")
    print(f"Male -> Female tx share: {stats['male_to_female_tx_pct']:.2f}%")
    print(f"Female -> Male tx share: {stats['female_to_male_tx_pct']:.2f}%")
    print(f"Male clients with >=1 Female product: {stats['male_to_female_client_pct']:.2f}%")
    print(f"Female clients with >=1 Male product: {stats['female_to_male_client_pct']:.2f}%")


if __name__ == "__main__":
    main()
## IGNORE-END