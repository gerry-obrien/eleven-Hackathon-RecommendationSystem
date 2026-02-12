from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.load_data import load_all
from src.preprocess import enrich_transactions

PRESETS = {
    # Left-card Canva placeholder fit used in this repo's deck drafts.
    "slide_fit_145": {
        "fig_width": 11.6,
        "fig_height": 8.0,
        "dpi": 220,
        "color": "#5c8bc3",
        "out_name_suffix": "_slide_fit_145",
    },
    # Wide center-panel Canva frame for the comprehensive strategy chart.
    "comprehensive_wide_220": {
        "fig_width": 16.0,
        "fig_height": 7.2,
        "dpi": 220,
        "color": "#5c8bc3",
        "out_name_suffix": "_comprehensive_wide_220",
    },
    # Exec-focused full-slide country/segment view.
    "country_segment_exec_wide": {
        "fig_width": 18.0,
        "fig_height": 8.6,
        "dpi": 220,
        "color": "#5c8bc3",
        "out_name_suffix": "_country_segment_exec",
    },
    "stock_actionability_exec": {
        "fig_width": 18.0,
        "fig_height": 9.0,
        "dpi": 220,
        "color": "#5c8bc3",
        "out_name_suffix": "_exec",
    },
    "stock_actionability_slide_wide": {
        "fig_width": 20.0,
        "fig_height": 7.8,
        "dpi": 220,
        "color": "#5c8bc3",
        "out_name_suffix": "_slide_wide",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export slide-ready static plots from raw data."
    )
    parser.add_argument(
        "--clip-upper",
        type=int,
        default=50,
        help="Upper clipping threshold for transactions-per-client histogram.",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="#5c8bc3",
        help="Hex color for histogram bars.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where plot files will be written.",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="",
        help="Optional output filename (e.g., heavy_tail_slide.png).",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=50,
        help="Maximum number of histogram bins (matches Streamlit helper default).",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=13.33,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=7.5,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="",
        choices=sorted(PRESETS.keys()),
        help="Optional named export preset.",
    )
    parser.add_argument(
        "--chart",
        type=str,
        default="heavy_tail",
        choices=["heavy_tail", "past_purchase_strategy", "country_segment_comprehensive", "stock_actionability"],
        help="Chart type to export.",
    )
    return parser.parse_args()


def validate_transactions_columns(transactions: pd.DataFrame) -> None:
    if "ClientID" not in transactions.columns:
        raise ValueError("transactions.csv missing required column: ClientID")


def validate_country_segment_columns(txe: pd.DataFrame) -> None:
    required = [
        "ClientCountry",
        "ClientSegment",
        "ClientID",
        "SaleTransactionDate",
        "Category",
        "SalesNetAmountEuro",
    ]
    missing = [c for c in required if c not in txe.columns]
    if missing:
        raise ValueError(f"Missing required columns for country-segment chart: {missing}")


def build_bins(clip_upper: int, nbins: int) -> np.ndarray:
    # Plotly's nbins is a "max bins" hint. For discrete transaction counts,
    # one bin per integer closely matches Streamlit readability.
    if nbins >= clip_upper:
        return np.arange(0.5, clip_upper + 1.5, 1.0)
    return np.linspace(0.5, clip_upper + 0.5, nbins + 1)


def export_heavy_tail_plot(
    transactions: pd.DataFrame,
    clip_upper: int,
    color: str,
    dpi: int,
    out_dir: Path,
    nbins: int,
    fig_width: float,
    fig_height: float,
    out_name: str,
) -> tuple[Path, dict[str, float]]:
    validate_transactions_columns(transactions)

    per_client = transactions.groupby("ClientID").size()
    clipped = per_client.clip(upper=clip_upper)

    stats = {
        "n_clients": float(per_client.shape[0]),
        "median_tx": float(per_client.median()),
        "mean_tx": float(per_client.mean()),
        "share_within_clip": float((per_client <= clip_upper).mean() * 100),
        "max_tx": float(per_client.max()),
    }

    bins = build_bins(clip_upper=clip_upper, nbins=nbins)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.hist(
        clipped,
        bins=bins,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.95,
        rwidth=0.95,
    )

    fig.suptitle(
        f"Transactions per client (Heavy Tail, clipped at {clip_upper})",
        fontsize=22,
        y=0.98,
    )
    ax.set_xlabel("Transactions per client", fontsize=12)
    ax.set_ylabel("Number of clients", fontsize=12)
    ax.set_xlim(0.5, clip_upper + 0.5)

    stat_text = (
        f"Clients: {int(stats['n_clients']):,} | "
        f"Median: {stats['median_tx']:.0f} | "
        f"Mean: {stats['mean_tx']:.2f} | "
        f"Max: {int(stats['max_tx']):,}"
    )
    fig.text(
        0.5,
        0.93,
        stat_text,
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )

    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Reserve top band for title + centered stats line to avoid overlap.
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    out_dir.mkdir(parents=True, exist_ok=True)
    if out_name:
        out_path = out_dir / out_name
    else:
        out_path = out_dir / f"heavy_tail_transactions_per_client_clip{clip_upper}.png"
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)

    non_empty_bins = int((np.histogram(clipped, bins=bins)[0] > 0).sum())
    stats["requested_nbins"] = float(nbins)
    stats["effective_bins"] = float(len(bins) - 1)
    stats["non_empty_bins"] = float(non_empty_bins)

    return out_path, stats


def export_past_purchase_strategy_plot(
    transactions: pd.DataFrame,
    clip_upper: int,
    color: str,
    dpi: int,
    out_dir: Path,
    nbins: int,
    fig_width: float,
    fig_height: float,
    out_name: str,
) -> tuple[Path, dict[str, float]]:
    validate_transactions_columns(transactions)

    per_client = transactions.groupby("ClientID").size()
    clipped = per_client.clip(upper=clip_upper)
    n_clients = len(per_client)

    stats = {
        "n_clients": float(n_clients),
        "median_tx": float(per_client.median()),
        "mean_tx": float(per_client.mean()),
        "share_within_clip": float((per_client <= clip_upper).mean() * 100),
        "max_tx": float(per_client.max()),
    }

    bucket_labels = ["1", "2", "3-5", "6-10", "10+"]
    bucket_bins = [0, 1, 2, 5, 10, np.inf]
    buckets = pd.cut(
        per_client,
        bins=bucket_bins,
        labels=bucket_labels,
        include_lowest=True,
        right=True,
    )
    bucket_share = (
        buckets.value_counts(normalize=True)
        .reindex(bucket_labels)
        .fillna(0.0)
        .mul(100)
    )
    share_1_2 = float(bucket_share.loc["1"] + bucket_share.loc["2"])

    bins = build_bins(clip_upper=clip_upper, nbins=nbins)
    x_vals = np.arange(1, clip_upper + 1)
    freq = per_client.value_counts().sort_index()
    cum_pct = (
        (freq.cumsum() / n_clients * 100)
        .reindex(x_vals, method="ffill")
        .fillna(0.0)
    )

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], hspace=0.40, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.hist(
        clipped,
        bins=bins,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.95,
        rwidth=0.95,
    )
    ax1.set_title("A) Purchase frequency is highly skewed", fontsize=12, pad=8)
    ax1.set_xlabel("Transactions per client", fontsize=10)
    ax1.set_ylabel("Number of clients", fontsize=10)
    ax1.set_xlim(0.5, clip_upper + 0.5)
    ax1.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax1.grid(axis="x", visible=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax1b = ax1.twinx()
    ax1b.plot(x_vals, cum_pct.values, color="#1f3d6d", linewidth=2.2)
    ax1b.set_ylim(0, 100)
    ax1b.set_ylabel("Cumulative % clients", fontsize=10, color="#1f3d6d")
    ax1b.yaxis.set_major_formatter(PercentFormatter())
    ax1b.tick_params(axis="y", labelsize=9, colors="#1f3d6d")
    ax1b.spines["top"].set_visible(False)

    bucket_colors = ["#5c8bc3", "#6a97cc", "#7fa7d4", "#97b8dd", "#b5cde8"]
    bars = ax2.bar(bucket_labels, bucket_share.values, color=bucket_colors, edgecolor="white")
    ax2.set_title("B) Cold-start segments dominate", fontsize=12, pad=8)
    ax2.set_xlabel("Transactions per client bucket", fontsize=10)
    ax2.set_ylabel("Share of clients (%)", fontsize=10)
    ax2.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax2.grid(axis="x", visible=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylim(0, max(bucket_share.max() * 1.25, 5))
    for rect, value in zip(bars, bucket_share.values):
        ax2.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.7,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1f1f1f",
        )

    ax3.axis("off")
    ax3.set_title("C) Recommendation strategy implied by data", fontsize=12, pad=6)

    box_specs = [
        {
            "x": 0.02,
            "w": 0.30,
            "title": f"1-2 tx clients ({share_1_2:.1f}%)",
            "body": "Model focus:\n- Popularity by country/segment\n- Category/brand affinity\nExecution:\n- Higher exploration and broad recall",
        },
        {
            "x": 0.35,
            "w": 0.30,
            "title": "3-5 tx clients",
            "body": "Model focus:\n- Item-item co-purchase signals\n- Personalized reranking\nExecution:\n- Balance relevance and novelty",
        },
        {
            "x": 0.68,
            "w": 0.30,
            "title": "6+ tx clients",
            "body": "Model focus:\n- Sequence/recency features\n- Strong personalized ranking\nExecution:\n- Low exploration, precision-first",
        },
    ]

    for spec in box_specs:
        rect = FancyBboxPatch(
            (spec["x"], 0.06),
            spec["w"],
            0.80,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            transform=ax3.transAxes,
            linewidth=1.0,
            edgecolor="#d6dce8",
            facecolor="#f7f9fc",
        )
        ax3.add_patch(rect)
        ax3.text(
            spec["x"] + 0.015,
            0.80,
            spec["title"],
            transform=ax3.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            color="#1f3d6d",
            fontweight="bold",
        )
        ax3.text(
            spec["x"] + 0.015,
            0.73,
            spec["body"],
            transform=ax3.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            color="#1f1f1f",
            linespacing=1.25,
        )

    fig.suptitle(
        "Distribution of past purchases -> recommendation strategy",
        fontsize=17,
        y=0.985,
    )
    fig.text(
        0.5,
        0.95,
        (
            f"Clients: {n_clients:,} | Median tx/client: {stats['median_tx']:.0f} | "
            f"Mean tx/client: {stats['mean_tx']:.2f} | "
            f"Share in 1-2 tx: {share_1_2:.1f}%"
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )

    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.06, top=0.88, wspace=0.28, hspace=0.40)

    out_dir.mkdir(parents=True, exist_ok=True)
    if out_name:
        out_path = out_dir / out_name
    else:
        out_path = out_dir / f"past_purchase_strategy_clip{clip_upper}.png"
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)

    non_empty_bins = int((np.histogram(clipped, bins=bins)[0] > 0).sum())
    stats["requested_nbins"] = float(nbins)
    stats["effective_bins"] = float(len(bins) - 1)
    stats["non_empty_bins"] = float(non_empty_bins)
    stats["share_1_2"] = share_1_2

    return out_path, stats


def export_country_segment_comprehensive_plot(
    txe: pd.DataFrame,
    color: str,
    dpi: int,
    out_dir: Path,
    fig_width: float,
    fig_height: float,
    out_name: str,
) -> tuple[Path, dict[str, float]]:
    validate_country_segment_columns(txe)

    seg_order = ["TOP", "LOYAL", "INACTIVE_1Y"]
    low_sample_threshold = 100

    t = txe.copy()
    t["SaleTransactionDate"] = pd.to_datetime(t["SaleTransactionDate"], errors="coerce")
    t = t.dropna(
        subset=["ClientCountry", "ClientSegment", "ClientID", "SaleTransactionDate", "Category"]
    ).copy()
    t["ClientCountry"] = t["ClientCountry"].astype(str)
    t["ClientSegment"] = t["ClientSegment"].astype(str)
    t["Category"] = t["Category"].astype(str)
    t = t[t["ClientSegment"].isin(seg_order)].copy()

    if t.empty:
        raise ValueError("No rows available for country-segment comprehensive chart after filtering.")

    countries = (
        t.groupby("ClientCountry")
        .size()
        .sort_values(ascending=False)
        .index.tolist()
    )

    total_tx = int(len(t))
    total_clients = int(t["ClientID"].nunique())

    client_seg = t[["ClientID", "ClientSegment"]].drop_duplicates(subset=["ClientID"])
    seg_client_share = (
        client_seg["ClientSegment"]
        .value_counts(normalize=True)
        .reindex(seg_order)
        .fillna(0.0)
        .mul(100)
    )

    tx_per_client = float(total_tx / total_clients) if total_clients else 0.0

    t_sorted = t.sort_values(["ClientID", "SaleTransactionDate"])
    t_sorted["next_date"] = t_sorted.groupby("ClientID")["SaleTransactionDate"].shift(-1)
    t_sorted["gap_days"] = (t_sorted["next_date"] - t_sorted["SaleTransactionDate"]).dt.days
    gaps = t_sorted.dropna(subset=["gap_days"]).copy()
    gaps = gaps[gaps["gap_days"] >= 0]
    overall_repeat_30 = float((gaps["gap_days"] <= 30).mean() * 100) if len(gaps) else 0.0

    idx = pd.MultiIndex.from_product([countries, seg_order], names=["ClientCountry", "ClientSegment"])
    agg = (
        t.groupby(["ClientCountry", "ClientSegment"])
        .agg(tx_count=("ClientID", "size"), clients=("ClientID", "nunique"))
        .reindex(idx, fill_value=0)
        .reset_index()
    )
    agg["tx_per_client"] = np.where(agg["clients"] > 0, agg["tx_count"] / agg["clients"], 0.0)
    agg["country_segment_tx_share_pct"] = np.where(total_tx > 0, agg["tx_count"] / total_tx * 100, 0.0)

    if len(gaps):
        repeat = (
            gaps.groupby(["ClientCountry", "ClientSegment"])["gap_days"]
            .apply(lambda s: float((s <= 30).mean() * 100))
            .reindex(idx, fill_value=0.0)
            .reset_index(name="repeat_within_30d_pct")
        )
    else:
        repeat = pd.DataFrame(
            [(c, s, 0.0) for c, s in idx],
            columns=["ClientCountry", "ClientSegment", "repeat_within_30d_pct"],
        )
    agg = agg.merge(repeat, on=["ClientCountry", "ClientSegment"], how="left")
    agg["repeat_within_30d_pct"] = agg["repeat_within_30d_pct"].fillna(0.0)

    cat_source = t.groupby(["ClientSegment", "Category"]).size().rename("tx").reset_index()
    top_n = 5
    seg_top_categories: dict[str, list[str]] = {}
    for seg in seg_order:
        top_cats = (
            cat_source[cat_source["ClientSegment"] == seg]
            .sort_values("tx", ascending=False)
            .head(top_n)["Category"]
            .tolist()
        )
        seg_top_categories[seg] = top_cats

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(
        3,
        15,
        height_ratios=[0.85, 4.95, 0.62],
        hspace=0.30,
        wspace=0.40,
    )
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_heat = fig.add_subplot(gs[1, 0:5])
    ax_ret = fig.add_subplot(gs[1, 5:10])
    c_sub = gs[1, 10:15].subgridspec(3, 1, hspace=0.55)
    c_axes = [fig.add_subplot(c_sub[i, 0]) for i in range(3)]
    ax_takeaways = fig.add_subplot(gs[2, :])

    fig.suptitle(
        "Country & Segment Differences: where retention strategy should diverge",
        fontsize=28,
        y=0.985,
    )
    fig.text(
        0.5,
        0.918,
        (
            "Purchase behavior is heterogeneous across countries and segments; "
            "one global recommendation policy underperforms."
        ),
        ha="center",
        va="top",
        fontsize=14,
        color="#2b2b2b",
    )

    ax_kpi.axis("off")
    kpi_items = [
        ("Transactions", f"{total_tx:,}"),
        ("Clients", f"{total_clients:,}"),
        (
            "Segment client mix",
            (
                f"TOP {seg_client_share.get('TOP', 0.0):.1f}% | "
                f"LOYAL {seg_client_share.get('LOYAL', 0.0):.1f}% | "
                f"INACTIVE_1Y {seg_client_share.get('INACTIVE_1Y', 0.0):.1f}%"
            ),
        ),
        ("Tx per client", f"{tx_per_client:.2f}"),
        ("Repeat <=30d", f"{overall_repeat_30:.1f}%"),
    ]
    kpi_x = [0.02, 0.20, 0.39, 0.70, 0.84]
    kpi_w = [0.16, 0.16, 0.29, 0.13, 0.14]
    for (label, value), x, w in zip(kpi_items, kpi_x, kpi_w):
        rect = FancyBboxPatch(
            (x, 0.10),
            w,
            0.78,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            transform=ax_kpi.transAxes,
            linewidth=1.0,
            edgecolor="#d7dfeb",
            facecolor="#f8fbff",
        )
        ax_kpi.add_patch(rect)
        ax_kpi.text(
            x + 0.01,
            0.77,
            label,
            transform=ax_kpi.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color="#3f4d68",
        )
        ax_kpi.text(
            x + 0.01,
            0.47,
            value,
            transform=ax_kpi.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            color="#1f2f4f",
            fontweight="bold",
        )

    heat = (
        agg.pivot(index="ClientCountry", columns="ClientSegment", values="country_segment_tx_share_pct")
        .reindex(index=countries, columns=seg_order)
        .fillna(0.0)
    )
    client_mat = (
        agg.pivot(index="ClientCountry", columns="ClientSegment", values="clients")
        .reindex(index=countries, columns=seg_order)
        .fillna(0.0)
    )
    im = ax_heat.imshow(heat.values, aspect="auto", cmap="Blues")
    ax_heat.set_title("A) Country x Segment volume map (% of total transactions)", fontsize=15, pad=10)
    ax_heat.set_xticks(np.arange(len(seg_order)))
    ax_heat.set_xticklabels(seg_order, fontsize=11)
    ax_heat.set_yticks(np.arange(len(countries)))
    ax_heat.set_yticklabels(countries, fontsize=11)
    for i, c in enumerate(countries):
        for j, s in enumerate(seg_order):
            pct = heat.loc[c, s]
            txc = int(agg[(agg["ClientCountry"] == c) & (agg["ClientSegment"] == s)]["tx_count"].iloc[0])
            clients_n = int(client_mat.loc[c, s])
            star = "*" if clients_n < low_sample_threshold else ""
            ax_heat.text(
                j,
                i,
                f"{pct:.1f}%\n{txc:,}{star}",
                ha="center",
                va="center",
                fontsize=9,
                color="#102040" if pct < heat.values.max() * 0.55 else "white",
            )

    ax_ret.set_title("B) Retention behavior matrix", fontsize=15, pad=10)
    x_map = {c: i for i, c in enumerate(countries)}
    marker_map = {"TOP": "o", "LOYAL": "s", "INACTIVE_1Y": "^"}
    segment_colors = {"TOP": "#2d4f8b", "LOYAL": "#5c8bc3", "INACTIVE_1Y": "#8db0d6"}

    for seg in seg_order:
        sdat = agg[agg["ClientSegment"] == seg].copy()
        sdat["x"] = sdat["ClientCountry"].map(x_map).astype(float)
        offset = {"TOP": -0.18, "LOYAL": 0.0, "INACTIVE_1Y": 0.18}[seg]
        sizes = np.clip(np.sqrt(sdat["clients"].values + 1) * 3.0, 20, 360)
        ax_ret.scatter(
            sdat["x"] + offset,
            sdat["tx_per_client"],
            s=sizes,
            c=segment_colors[seg],
            marker=marker_map[seg],
            alpha=0.9,
            edgecolor="#2f2f2f",
            linewidth=0.3,
            label=seg,
        )
        for _, row in sdat.iterrows():
            ax_ret.text(
                row["x"] + offset,
                row["tx_per_client"] + 1.2,
                f"{row['repeat_within_30d_pct']:.0f}%",
                fontsize=8.8,
                color="#3a3a3a",
                ha="center",
                va="bottom",
            )
            if row["clients"] < low_sample_threshold:
                ax_ret.text(
                    row["x"] + offset + 0.02,
                    row["tx_per_client"] + 2.4,
                    "*",
                    fontsize=12,
                    color="#8a3b12",
                    fontweight="bold",
                )

    ax_ret.set_xticks(np.arange(len(countries)))
    ax_ret.set_xticklabels(countries, fontsize=11)
    ax_ret.set_ylim(-1, max(agg["tx_per_client"].max() + 10, 20))
    ax_ret.set_ylabel("tx_per_client", fontsize=13, labelpad=4)
    ax_ret.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax_ret.grid(axis="x", visible=False)
    ax_ret.tick_params(axis="y", labelsize=11)
    ax_ret.spines["top"].set_visible(False)
    ax_ret.spines["right"].set_visible(False)
    handles = [
        Line2D([0], [0], marker=marker_map[s], color="w", markerfacecolor=segment_colors[s], markeredgecolor="#2f2f2f", markersize=7, label=s)
        for s in seg_order
    ]
    ax_ret.legend(handles=handles, title="Segment", fontsize=10, title_fontsize=10, loc="upper right")

    palette = ["#5c8bc3", "#6f99cc", "#86abd7", "#9bbce0", "#b4cee9", "#cfdff1"]
    for i, seg in enumerate(seg_order):
        ax = c_axes[i]
        seg_data = t[t["ClientSegment"] == seg].copy()
        top_cats = seg_top_categories[seg]
        seg_data["CatPlot"] = seg_data["Category"].where(seg_data["Category"].isin(top_cats), other="Other")
        share = (
            seg_data.groupby(["ClientCountry", "CatPlot"])
            .size()
            .rename("tx")
            .reset_index()
        )
        denom = seg_data.groupby("ClientCountry").size().rename("denom")
        share = share.merge(denom, on="ClientCountry", how="left")
        share["pct"] = np.where(share["denom"] > 0, share["tx"] / share["denom"] * 100, 0.0)
        pivot = (
            share.pivot(index="ClientCountry", columns="CatPlot", values="pct")
            .reindex(index=countries)
            .fillna(0.0)
        )
        ordered_cols = [c for c in top_cats if c in pivot.columns] + ["Other"]
        ordered_cols = [c for c in ordered_cols if c in pivot.columns]
        bottom = np.zeros(len(countries))
        for j, cat in enumerate(ordered_cols):
            vals = pivot[cat].values
            ax.bar(countries, vals, bottom=bottom, color=palette[j % len(palette)], edgecolor="white", linewidth=0.2, label=cat)
            bottom += vals
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_ylabel("")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=10, rotation=0)
        ax.tick_params(axis="y", labelsize=8.8)
        ax.text(
            0.01,
            0.90,
            seg,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            color="#203960",
            fontweight="bold",
        )
        if i == 0:
            ax.set_title("C) Preference heterogeneity (category mix by country)", fontsize=15, pad=10)
        if i < 2:
            ax.set_xticklabels([])
    c_axes[-1].set_xlabel("")
    ax_takeaways.axis("off")
    fig.subplots_adjust(left=0.035, right=0.99, top=0.90, bottom=0.06, wspace=0.28, hspace=0.30)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name if out_name else out_dir / "country_segment_comprehensive_exec.png"
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)

    stats = {
        "n_transactions": float(total_tx),
        "n_clients": float(total_clients),
        "overall_tx_per_client": tx_per_client,
        "overall_repeat_within_30d_pct": overall_repeat_30,
        "n_countries": float(len(countries)),
        "n_segments": float(len(seg_order)),
    }
    return out_path, stats


def export_stock_actionability_plot(
    txe: pd.DataFrame,
    clients: pd.DataFrame,
    stocks: pd.DataFrame,
    color: str,
    dpi: int,
    out_dir: Path,
    fig_width: float,
    fig_height: float,
    out_name: str,
) -> tuple[Path, dict[str, float]]:
    required = ["ClientID", "ProductID", "ClientCountry"]
    missing = [col for col in required if col not in txe.columns]
    if missing:
        raise ValueError(f"Missing required columns for stock actionability chart: {missing}")

    if "Quantity" not in stocks.columns or "StoreCountry" not in stocks.columns or "ProductID" not in stocks.columns:
        raise ValueError("stocks.csv must include StoreCountry, ProductID, Quantity")

    tx = txe.dropna(subset=["ClientID", "ProductID"]).copy()
    tx["ClientCountry"] = tx["ClientCountry"].astype(str)

    stocks_pos = stocks[stocks["Quantity"] > 0].copy()
    stock_cov = (
        stocks_pos.groupby("StoreCountry")["ProductID"]
        .nunique()
        .rename("products_with_stock")
        .reset_index()
        .sort_values("products_with_stock", ascending=False)
    )

    purchased_by_country = (
        tx.dropna(subset=["ClientCountry"])
        .groupby("ClientCountry")["ProductID"]
        .apply(lambda series: set(series.unique()))
    )
    stock_map = {
        str(country): set(group["ProductID"].unique())
        for country, group in stocks_pos.groupby("StoreCountry")
    }

    overlap_rows = []
    for country, purchased_set in purchased_by_country.items():
        stock_set = stock_map.get(str(country), set())
        overlap_set = purchased_set & stock_set
        overlap_rows.append(
            {
                "country": str(country),
                "purchased_unique_products": len(purchased_set),
                "in_stock_unique_products": len(stock_set),
                "overlap_unique_products": len(overlap_set),
                "overlap_rate_vs_purchased": (len(overlap_set) / len(purchased_set)) if purchased_set else 0.0,
            }
        )
    overlap_df = pd.DataFrame(overlap_rows).sort_values("overlap_rate_vs_purchased", ascending=False)

    k = 10
    top_global = tx["ProductID"].value_counts().head(k).index.tolist()
    client_country = clients.set_index("ClientID")["ClientCountry"] if "ClientCountry" in clients.columns else pd.Series(dtype=object)

    kept_counts = []
    for client_id, country in client_country.items():
        allowed = stock_map.get(str(country))
        if allowed is None:
            kept_counts.append(k)
        else:
            kept_counts.append(sum(1 for product_id in top_global if product_id in allowed))

    avg_remaining = float(np.mean(kept_counts)) if kept_counts else 0.0
    pct_full_k = float((sum(1 for value in kept_counts if value == k) / len(kept_counts)) * 100) if kept_counts else 0.0
    pct_loss = float((1 - avg_remaining / k) * 100) if k > 0 else 0.0
    countries_without_stock = sorted(set(purchased_by_country.index.astype(str)) - set(stock_map.keys()))

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 12, height_ratios=[0.95, 4.75], hspace=0.45, wspace=0.58)

    ax_kpi = fig.add_subplot(gs[0, :])
    ax_a = fig.add_subplot(gs[1, 0:6])
    ax_b = fig.add_subplot(gs[1, 6:12])

    fig.suptitle("Stock & Actionability: operational constraint on recommendations", fontsize=28, y=0.982)
    fig.text(
        0.5,
        0.928,
        "Even accurate models fail if recommended products are unavailable in market.",
        ha="center",
        va="top",
        fontsize=15,
        color="#2b2b2b",
    )

    ax_kpi.axis("off")
    kpis = [
        ("Countries with stock data", f"{stock_cov['StoreCountry'].nunique():,}"),
        ("Avg items remaining (Top-10)", f"{avg_remaining:.2f}"),
        ("% clients with full Top-10", f"{pct_full_k:.1f}%"),
        ("Avg Top-10 shrinkage", f"{pct_loss:.1f}%"),
        ("Countries w/o stock data", ", ".join(countries_without_stock) if countries_without_stock else "None"),
    ]
    kpi_x = [0.01, 0.22, 0.43, 0.64, 0.82]
    kpi_w = [0.18, 0.18, 0.18, 0.16, 0.17]
    for (label, value), x, width in zip(kpis, kpi_x, kpi_w):
        rect = FancyBboxPatch(
            (x, 0.12),
            width,
            0.75,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            transform=ax_kpi.transAxes,
            linewidth=1.0,
            edgecolor="#d7dfeb",
            facecolor="#f8fbff",
        )
        ax_kpi.add_patch(rect)
        ax_kpi.text(x + 0.012, 0.78, label, transform=ax_kpi.transAxes, ha="left", va="top", fontsize=11, color="#3f4d68")
        ax_kpi.text(x + 0.012, 0.50, value, transform=ax_kpi.transAxes, ha="left", va="top", fontsize=14, color="#1f2f4f", fontweight="bold")

    cov_plot = stock_cov.copy()
    ax_a.bar(cov_plot["StoreCountry"], cov_plot["products_with_stock"], color=color, edgecolor="white")
    ax_a.set_title("A) Stock coverage by country", fontsize=15, pad=10)
    ax_a.set_ylabel("products_with_stock", fontsize=12)
    ax_a.tick_params(axis="x", labelsize=11)
    ax_a.tick_params(axis="y", labelsize=11)
    ax_a.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax_a.grid(axis="x", visible=False)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    overlap_plot = overlap_df.sort_values("overlap_rate_vs_purchased", ascending=False).copy()
    ax_b.bar(overlap_plot["country"], overlap_plot["overlap_rate_vs_purchased"] * 100, color="#6f99cc", edgecolor="white")
    ax_b.set_title("B) Purchased vs in-stock overlap rate", fontsize=15, pad=10)
    ax_b.set_ylabel("overlap_rate_vs_purchased (%)", fontsize=12)
    ax_b.set_ylim(0, 35)
    ax_b.tick_params(axis="x", labelsize=11)
    ax_b.tick_params(axis="y", labelsize=11)
    ax_b.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax_b.grid(axis="x", visible=False)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    for idx, row in overlap_plot.iterrows():
        ax_b.text(
            row["country"],
            row["overlap_rate_vs_purchased"] * 100 + 0.6,
            f"{row['overlap_rate_vs_purchased'] * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#2f2f2f",
        )

    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.10, wspace=0.58, hspace=0.45)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name if out_name else out_dir / "stock_actionability_exec.png"
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)

    stats = {
        "n_clients": float(len(client_country)),
        "avg_remaining_topk": avg_remaining,
        "pct_full_topk": pct_full_k,
        "n_countries_with_stock": float(stock_cov["StoreCountry"].nunique()),
    }
    return out_path, stats


def main() -> None:
    args = parse_args()

    if args.preset:
        preset = PRESETS[args.preset]
        args.fig_width = float(preset["fig_width"])
        args.fig_height = float(preset["fig_height"])
        args.dpi = int(preset["dpi"])
        args.color = str(preset["color"])
        if not args.out_name:
            if args.chart == "country_segment_comprehensive":
                args.out_name = "country_segment_comprehensive_exec.png"
            elif args.chart == "stock_actionability":
                if args.preset == "stock_actionability_exec":
                    args.out_name = "stock_actionability_exec.png"
                elif args.preset == "stock_actionability_slide_wide":
                    args.out_name = "stock_actionability_slide_wide.png"
                else:
                    args.out_name = f"stock_actionability{preset['out_name_suffix']}.png"
            elif args.chart == "heavy_tail":
                base_name = "heavy_tail_transactions_per_client"
            elif args.chart == "past_purchase_strategy":
                base_name = "past_purchase_strategy"
            else:
                base_name = "country_segment_comprehensive"
            if args.chart not in {"country_segment_comprehensive", "stock_actionability"}:
                args.out_name = f"{base_name}_clip{args.clip_upper}{preset['out_name_suffix']}.png"

    if args.clip_upper <= 0:
        raise ValueError("--clip-upper must be a positive integer")
    if args.dpi <= 0:
        raise ValueError("--dpi must be a positive integer")
    if args.nbins <= 0:
        raise ValueError("--nbins must be a positive integer")
    if args.fig_width <= 0:
        raise ValueError("--fig-width must be a positive number")
    if args.fig_height <= 0:
        raise ValueError("--fig-height must be a positive number")

    data = load_all()
    if args.chart == "heavy_tail":
        out_path, stats = export_heavy_tail_plot(
            transactions=data.transactions,
            clip_upper=args.clip_upper,
            color=args.color,
            dpi=args.dpi,
            out_dir=args.out_dir,
            nbins=args.nbins,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            out_name=args.out_name,
        )
    elif args.chart == "past_purchase_strategy":
        out_path, stats = export_past_purchase_strategy_plot(
            transactions=data.transactions,
            clip_upper=args.clip_upper,
            color=args.color,
            dpi=args.dpi,
            out_dir=args.out_dir,
            nbins=args.nbins,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            out_name=args.out_name,
        )
    elif args.chart == "stock_actionability":
        txe = enrich_transactions(data.transactions, data.clients, data.products, data.stores)
        out_path, stats = export_stock_actionability_plot(
            txe=txe,
            clients=data.clients,
            stocks=data.stocks,
            color=args.color,
            dpi=args.dpi,
            out_dir=args.out_dir,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            out_name=args.out_name,
        )
    else:
        txe = enrich_transactions(data.transactions, data.clients, data.products, data.stores)
        out_path, stats = export_country_segment_comprehensive_plot(
            txe=txe,
            color=args.color,
            dpi=args.dpi,
            out_dir=args.out_dir,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            out_name=args.out_name,
        )

    print(f"Saved plot: {out_path.resolve()}")
    if "median_tx" in stats:
        print(f"Clients: {int(stats['n_clients']):,}")
        print(f"Median tx/client: {stats['median_tx']:.0f}")
        print(f"Mean tx/client: {stats['mean_tx']:.2f}")
        print(f"Max tx/client: {int(stats['max_tx']):,}")
        print(f"Share <= clip ({args.clip_upper}): {stats['share_within_clip']:.2f}%")
        print(f"Requested nbins: {int(stats['requested_nbins'])}")
        print(f"Effective bins drawn: {int(stats['effective_bins'])}")
        print(f"Non-empty bins: {int(stats['non_empty_bins'])}")
    else:
        if "n_transactions" in stats:
            print(f"Transactions: {int(stats['n_transactions']):,}")
            print(f"Clients: {int(stats['n_clients']):,}")
            print(f"tx_per_client: {stats['overall_tx_per_client']:.2f}")
            print(f"repeat_within_30d: {stats['overall_repeat_within_30d_pct']:.1f}%")
            print(f"Countries: {int(stats['n_countries'])}")
            print(f"Segments: {int(stats['n_segments'])}")
        else:
            print(f"Clients: {int(stats['n_clients']):,}")
            print(f"Avg remaining Top-10: {stats['avg_remaining_topk']:.2f}")
            print(f"% full Top-10: {stats['pct_full_topk']:.1f}%")
            print(f"Countries with stock data: {int(stats['n_countries_with_stock'])}")


if __name__ == "__main__":
    main()
## IGNORE-END