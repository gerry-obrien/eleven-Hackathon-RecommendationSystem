# app_eda/components/charts.py
import pandas as pd
import plotly.express as px


def hist(series: pd.Series, title: str, nbins: int = 50):
    s = series.dropna()
    fig = px.histogram(s, nbins=nbins, title=title)
    fig.update_layout(bargap=0.05)
    return fig


def bar(df: pd.DataFrame, x: str, y: str, title: str):
    fig = px.bar(df, x=x, y=y, title=title)
    return fig


def stacked_bar_share(df: pd.DataFrame, group_col: str, item_col: str, top_n: int = 8, title: str = ""):
    """
    Build a stacked share bar: within each group, show share of top categories/items.
    """
    counts = (
        df.groupby([group_col, item_col])
        .size()
        .rename("count")
        .reset_index()
    )
    # keep top_n items overall to avoid clutter
    top_items = counts.groupby(item_col)["count"].sum().sort_values(ascending=False).head(top_n).index
    counts[item_col] = counts[item_col].where(counts[item_col].isin(top_items), other="Other")

    counts2 = counts.groupby([group_col, item_col])["count"].sum().reset_index()
    totals = counts2.groupby(group_col)["count"].sum().rename("total").reset_index()
    merged = counts2.merge(totals, on=group_col, how="left")
    merged["share"] = merged["count"] / merged["total"]

    fig = px.bar(merged, x=group_col, y="share", color=item_col, title=title, barmode="stack")
    fig.update_yaxes(tickformat=".0%")
    return fig
