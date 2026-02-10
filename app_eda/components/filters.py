# app_eda/components/filters.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class Filters:
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    client_countries: list[str]
    store_countries: list[str]
    segments: list[str]


def _safe_list(series: pd.Series) -> list[str]:
    vals = series.dropna().astype(str).unique().tolist()
    vals.sort()
    return vals


def init_filters_state(txe: pd.DataFrame) -> None:
    """
    Initialize session_state defaults once.
    """
    if "filters_initialized" in st.session_state:
        return

    dmin = txe["SaleTransactionDate"].min()
    dmax = txe["SaleTransactionDate"].max()

    st.session_state["f_date_min"] = dmin
    st.session_state["f_date_max"] = dmax
    st.session_state["f_client_countries"] = []   # empty means "all"
    st.session_state["f_store_countries"] = []
    st.session_state["f_segments"] = []
    st.session_state["filters_initialized"] = True


def render_sidebar_filters(txe: pd.DataFrame) -> Filters:
    """
    Render sidebar controls and return active filters.
    Filters apply to enriched transactions (txe).
    """
    init_filters_state(txe)

    st.sidebar.header("Global filters")

    dmin = txe["SaleTransactionDate"].min()
    dmax = txe["SaleTransactionDate"].max()

    date_range = st.sidebar.date_input(
        "Transaction date range",
        value=(st.session_state["f_date_min"].date(), st.session_state["f_date_max"].date()),
        min_value=dmin.date(),
        max_value=dmax.date(),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        st.session_state["f_date_min"] = pd.to_datetime(date_range[0])
        st.session_state["f_date_max"] = pd.to_datetime(date_range[1])

    client_country_opts = _safe_list(txe.get("ClientCountry", pd.Series(dtype=object)))
    store_country_opts = _safe_list(txe.get("StoreCountry", pd.Series(dtype=object)))
    seg_opts = _safe_list(txe.get("ClientSegment", pd.Series(dtype=object)))

    st.session_state["f_client_countries"] = st.sidebar.multiselect(
        "Client country",
        options=client_country_opts,
        default=st.session_state["f_client_countries"],
        help="Filters transactions by the client's country attribute.",
    )
    st.session_state["f_store_countries"] = st.sidebar.multiselect(
        "Store country",
        options=store_country_opts,
        default=st.session_state["f_store_countries"],
        help="Filters transactions by the store country (StoreID → StoreCountry).",
    )
    st.session_state["f_segments"] = st.sidebar.multiselect(
        "Client segment",
        options=seg_opts,
        default=st.session_state["f_segments"],
        help="Filters transactions by client segment.",
    )

    return Filters(
        date_min=st.session_state["f_date_min"],
        date_max=st.session_state["f_date_max"],
        client_countries=st.session_state["f_client_countries"],
        store_countries=st.session_state["f_store_countries"],
        segments=st.session_state["f_segments"],
    )


def apply_filters(txe: pd.DataFrame, f: Filters) -> pd.DataFrame:
    """
    Apply filter object to enriched transactions (txe).
    """
    out = txe
    out = out[(out["SaleTransactionDate"] >= f.date_min) & (out["SaleTransactionDate"] <= f.date_max)]

    if f.client_countries and "ClientCountry" in out.columns:
        out = out[out["ClientCountry"].astype(str).isin(f.client_countries)]

    if f.store_countries and "StoreCountry" in out.columns:
        out = out[out["StoreCountry"].astype(str).isin(f.store_countries)]

    if f.segments and "ClientSegment" in out.columns:
        out = out[out["ClientSegment"].astype(str).isin(f.segments)]

    return out
