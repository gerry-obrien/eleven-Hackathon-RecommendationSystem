# src/recommender/model.py
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer


FEATURE_COLS = [
    "ClientCountry",
    "ClientOptINEmail",
    "ClientOptINPhone",
    "ClientGender",
    "Age",
]


@dataclass
class KNNRecoModel:
    preprocessor: ColumnTransformer
    knn: NearestNeighbors
    X_train: np.ndarray             # training matrix aligned with ids
    ids: np.ndarray                 # ClientID aligned with X_train rows

    client_products: pd.Series      # index=ClientID -> list(ProductID)
    client_country: pd.Series       # index=ClientID -> country
    stock_map: dict[str, set]       # country -> set(ProductID in stock)
    products: pd.DataFrame          # product catalog dataframe


def fit_knn_model(
    existing_clients: pd.DataFrame,     # must contain ClientID + FEATURE_COLS
    preprocessor: ColumnTransformer,
    client_products: pd.Series,
    client_country: pd.Series,
    stock_map: dict[str, set],
    products: pd.DataFrame,
    n_neighbors_fit: int = 50,          # internal fit; query will choose k
) -> KNNRecoModel:
    df = existing_clients.copy()

    ids = df["ClientID"].to_numpy()
    X_df = df[FEATURE_COLS].copy()

    X = preprocessor.fit_transform(X_df)
    X = np.asarray(X, dtype=np.float32)

    knn = NearestNeighbors(n_neighbors=n_neighbors_fit, metric="cosine", algorithm="brute")
    knn.fit(X)

    return KNNRecoModel(
        preprocessor=preprocessor,
        knn=knn,
        X_train=X,
        ids=ids,
        client_products=client_products,
        client_country=client_country,
        stock_map=stock_map,
        products=products,
    )


def _neighbors_from_vector(model: KNNRecoModel, x_query: np.ndarray, k_neighbors: int = 5):
    # ask for k+1 then drop self if present (for existing clients)
    distances, indices = model.knn.kneighbors(x_query, n_neighbors=k_neighbors + 1)
    return indices.ravel(), distances.ravel()


def _remove_self_if_present(query_idx: int, indices: np.ndarray, distances: np.ndarray, k_neighbors: int):
    mask = indices != query_idx
    indices = indices[mask][:k_neighbors]
    distances = distances[mask][:k_neighbors]
    return indices, distances


def build_candidates_from_neighbor_ids(
    model: KNNRecoModel,
    neighbor_client_ids: np.ndarray,
    topn: int = 200,
) -> tuple[list[int], dict[int, int]]:
    """
    Returns:
      - candidates: sorted ProductIDs by frequency among neighbors
      - counts_map: ProductID -> neighbor_count
    """
    big_list = []
    for nid in neighbor_client_ids:
        big_list.extend(model.client_products.get(nid, []))

    counts = Counter(big_list)
    top_pairs = counts.most_common(topn)
    candidates = [pid for pid, _ in top_pairs]
    counts_map = dict(top_pairs)
    return candidates, counts_map


def filter_candidates_for_client(
    model: KNNRecoModel,
    query_client_id: int | None,        # None for anonymous profile
    candidates: list[int],
    country: str | None,
    k: int = 10,
) -> list[int]:
    allowed = model.stock_map.get(str(country), None) if country is not None else None
    already = set(model.client_products.get(query_client_id, [])) if query_client_id is not None else set()

    out = []
    for pid in candidates:
        if pid in already:
            continue
        if allowed is not None and pid not in allowed:
            continue
        out.append(pid)
        if len(out) >= k:
            break
    return out


def describe_products(
    model: KNNRecoModel,
    product_ids: list[int],
    country: str | None,
    counts_map: dict[int, int] | None = None
) -> pd.DataFrame:
    df = pd.DataFrame({"ProductID": product_ids})
    if counts_map is not None:
        df["NeighborCount"] = df["ProductID"].map(counts_map).fillna(0).astype(int)

    df = df.merge(model.products, on="ProductID", how="left")

    if country is None:
        df["InStockInCountry"] = np.nan
    else:
        allowed = model.stock_map.get(str(country), set())
        df["InStockInCountry"] = df["ProductID"].apply(lambda x: x in allowed)

    if "NeighborCount" in df.columns:
        df = df.sort_values("NeighborCount", ascending=False)

    return df.reset_index(drop=True)


def recommend_for_client_id(
    model: KNNRecoModel,
    clients_df: pd.DataFrame,
    query_client_id: int,
    k_neighbors: int = 5,
    topn_candidates: int = 200,
    k_reco: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mode A:
      - Find client row in clients table (newby or existing)
      - Compute neighbors in existing population
      - Build candidates from neighbor purchases
      - Filter by stock for the client's country + remove already bought if any
    Returns:
      reco_df, neighbors_df
    """
    row = clients_df.loc[clients_df["ClientID"] == query_client_id, FEATURE_COLS]
    if row.empty:
        raise ValueError(f"ClientID {query_client_id} not found in clients table.")
    country = str(clients_df.loc[clients_df["ClientID"] == query_client_id, "ClientCountry"].iloc[0])

    x_query = model.preprocessor.transform(row)
    x_query = np.asarray(x_query, dtype=np.float32)

    indices, distances = _neighbors_from_vector(model, x_query, k_neighbors=k_neighbors)

    # if this client is part of training ids, drop self
    pos = np.where(model.ids == query_client_id)[0]
    if len(pos) > 0:
        query_idx = int(pos[0])
        indices, distances = _remove_self_if_present(query_idx, indices, distances, k_neighbors)

    neighbor_ids = model.ids[indices]

    candidates, counts_map = build_candidates_from_neighbor_ids(model, neighbor_ids, topn=topn_candidates)
    reco_ids = filter_candidates_for_client(model, query_client_id, candidates, country=country, k=k_reco)

    reco_df = describe_products(model, reco_ids, country=country, counts_map=counts_map)
    neighbors_df = pd.DataFrame({
        "NeighborClientID": neighbor_ids,
        "CosineDistance": distances[:len(neighbor_ids)]
    })
    return reco_df, neighbors_df


def recommend_for_profile(
    model: KNNRecoModel,
    profile: dict,
    k_neighbors: int = 5,
    topn_candidates: int = 200,
    k_reco: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mode B:
      - profile is a dict with keys in FEATURE_COLS
      - no 'already bought' removal (anonymous / no history)
      - stock filter uses profile['ClientCountry'] if provided
    """
    row = pd.DataFrame([profile])
    for col in FEATURE_COLS:
        if col not in row.columns:
            row[col] = np.nan
    row = row[FEATURE_COLS]

    country = profile.get("ClientCountry")
    country = None if country is None or (isinstance(country, float) and np.isnan(country)) else str(country)

    x_query = model.preprocessor.transform(row)
    x_query = np.asarray(x_query, dtype=np.float32)

    indices, distances = _neighbors_from_vector(model, x_query, k_neighbors=k_neighbors)
    # Here we don't have a "self" row, so keep first k
    indices = indices[:k_neighbors]
    distances = distances[:k_neighbors]

    neighbor_ids = model.ids[indices]

    candidates, counts_map = build_candidates_from_neighbor_ids(model, neighbor_ids, topn=topn_candidates)
    reco_ids = filter_candidates_for_client(model, query_client_id=None, candidates=candidates, country=country, k=k_reco)

    reco_df = describe_products(model, reco_ids, country=country, counts_map=counts_map)
    neighbors_df = pd.DataFrame({
        "NeighborClientID": neighbor_ids,
        "CosineDistance": distances
    })
    return reco_df, neighbors_df
