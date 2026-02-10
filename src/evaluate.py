# src/evaluate.py
from __future__ import annotations

import numpy as np
import pandas as pd


def last_purchase_split(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-client leakage-safe split for next-purchase evaluation:
      - Train: all but last transaction per client (by SaleTransactionDate)
      - Test:  last transaction per client

    Requires: ClientID, SaleTransactionDate
    """
    tx = transactions.sort_values(["ClientID", "SaleTransactionDate"])
    idx_last = tx.groupby("ClientID").tail(1).index
    test = tx.loc[idx_last].copy()
    train = tx.drop(index=idx_last).copy()
    return train, test


def hitrate_at_k(recs: dict, truth: pd.Series, k: int = 10) -> float:
    """
    HitRate@K (same as Recall@K when each user has one true item).
    recs: dict client_id -> ranked list of ProductID
    truth: Series indexed by client_id with the true next ProductID
    """
    hits, total = 0, 0
    for cid, true_pid in truth.items():
        items = recs.get(cid)
        if not items:
            continue
        total += 1
        if true_pid in items[:k]:
            hits += 1
    return hits / total if total else 0.0


def mrr_at_k(recs: dict, truth: pd.Series, k: int = 10) -> float:
    """
    Mean Reciprocal Rank@K.
    If true item is at rank r (1-indexed), score = 1/r. If not in top K, score = 0.
    """
    scores = []
    for cid, true_pid in truth.items():
        items = recs.get(cid)
        if not items:
            continue
        topk = items[:k]
        if true_pid in topk:
            scores.append(1.0 / (topk.index(true_pid) + 1))
        else:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


def customer_coverage_at_k(recs: dict, truth_index: pd.Index, k: int = 10) -> float:
    """
    % of evaluated customers who received at least K recommendations.
    Useful when stock filtering may shorten lists.
    """
    ids = list(truth_index)
    if not ids:
        return 0.0
    ok = 0
    for cid in ids:
        items = recs.get(cid, [])
        if items is not None and len(items) >= k:
            ok += 1
    return ok / len(ids)


def summarize_metrics(recs: dict, truth: pd.Series, k: int = 10) -> pd.DataFrame:
    """
    Convenience: returns a 1-row dataframe with the three core metrics.
    """
    return pd.DataFrame(
        [{
            "K": k,
            "HitRate@K": hitrate_at_k(recs, truth, k=k),
            "MRR@K": mrr_at_k(recs, truth, k=k),
            "CustomerCoverage@K": customer_coverage_at_k(recs, truth.index, k=k),
        }]
    )
