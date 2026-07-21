"""Reusable Plotly chart builders with modernized dark aesthetic."""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from components.theme import COLORS, PIN_COLORS, base_layout as _base_layout

STYLE = {
    "bg": COLORS["canvas"],
    "positive": COLORS["positive"],
    "negative": COLORS["negative"],
    "warning": COLORS["warning"],
    "text": COLORS["text"],
    "muted": COLORS["text_secondary"],
    "accent": COLORS["primary"],
    "template": "alpha",
}


def cumulative_wealth_chart(
    returns_dict: dict[str, pd.Series],
    start_val: float = 10000,
    cash_flow: float = 0,
) -> go.Figure:
    fig = go.Figure()

    for i, (name, rets) in enumerate(returns_dict.items()):
        values = []
        bal = start_val
        for r in rets:
            bal = (bal + cash_flow) * (1 + r)
            values.append(bal)
        wealth = pd.Series(values, index=rets.index)

        color = STYLE["positive"] if i == 0 else (
            COLORS["text_muted"] if name == "SPY" else PIN_COLORS[(i - 1) % len(PIN_COLORS)]
        )
        dash = "dash" if name == "SPY" else None
        fig.add_trace(go.Scatter(
            x=wealth.index, y=wealth.values, name=name,
            line=dict(color=color, width=2, dash=dash),
        ))

    cf_label = f", ${cash_flow:+,}/mo" if cash_flow != 0 else ""
    fig.update_layout(**_base_layout(
        title="Cumulative Wealth",
        yaxis_title=f"Portfolio Value (${start_val:,} start{cf_label})",
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
    ))
    return fig


def drawdown_chart(returns_dict: dict[str, pd.Series], start_val: float = 10000) -> go.Figure:
    fig = go.Figure()

    for i, (name, rets) in enumerate(returns_dict.items()):
        cum = (1 + rets).cumprod() * start_val
        dd = cum / cum.cummax() - 1

        color = STYLE["negative"] if i == 0 else PIN_COLORS[(i - 1) % len(PIN_COLORS)]
        fill = "tozeroy" if i == 0 else None
        fillcolor = "rgba(244,106,106,0.22)" if i == 0 else None

        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=name, fill=fill,
            line=dict(color=color, width=1), fillcolor=fillcolor,
        ))

    fig.update_layout(**_base_layout(
        title="Drawdown", yaxis_title="Drawdown %", yaxis_tickformat=".0%",
    ))
    return fig


def monthly_heatmap(returns: pd.Series) -> go.Figure:
    df = returns.to_frame("ret")
    df["year"] = df.index.str[:4]
    df["month"] = df.index.str[5:7].astype(int)
    pivot = df.pivot_table(values="ret", index="year", columns="month", aggfunc="first")

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_labels[c - 1] for c in pivot.columns]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 100, x=pivot.columns, y=pivot.index,
        colorscale=[[0, STYLE["negative"]], [0.5, COLORS["surface"]], [1, STYLE["positive"]]],
        zmid=0,
        text=np.round(pivot.values * 100, 1),
        texttemplate="%{text:.1f}%",
        textfont=dict(size=10),
        hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(**_base_layout(title="Monthly Returns Heatmap (%)"))
    return fig


def rolling_metric_chart(
    series: pd.Series,
    window: int = 12,
    name: str = "Metric",
    show_bands: bool = True,
) -> go.Figure:
    rolling_mean = series.rolling(window, min_periods=window // 2).mean()
    fig = go.Figure()

    if show_bands:
        rolling_std = series.rolling(window, min_periods=window // 2).std()
        upper = rolling_mean + rolling_std
        lower = rolling_mean - rolling_std
        fig.add_trace(go.Scatter(
            x=upper.index, y=upper.values, mode="lines", name="+1σ",
            line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=lower.index, y=lower.values, mode="lines", name="-1σ",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(79,195,232,0.15)", showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=rolling_mean.index, y=rolling_mean.values, name=f"Rolling {window}m {name}",
        line=dict(color=STYLE["accent"], width=2),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=STYLE["muted"])

    fig.update_layout(**_base_layout(title=f"Rolling {window}-Month {name}", yaxis_title=name))
    return fig


def bar_chart(
    series: pd.Series,
    name: str = "Value",
    mean_line: bool = True,
) -> go.Figure:
    colors = [STYLE["positive"] if v >= 0 else STYLE["negative"] for v in series.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=series.index, y=series.values, name=name,
        marker_color=colors, opacity=0.7,
    ))

    if mean_line:
        fig.add_hline(
            y=series.mean(), line_dash="dash", line_color=STYLE["warning"],
            annotation_text=f"Mean={series.mean():.3f}",
        )

    fig.update_layout(**_base_layout(title=name, yaxis_title=name))
    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu_r", zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text:.2f}",
        textfont=dict(size=8),
    ))
    fig.update_layout(**_base_layout(title="Spearman Correlation Matrix", height=600))
    return fig


def sector_allocation_chart(
    holdings_dict: dict[str, pd.DataFrame],
    strategy_type: str = "long_only",
) -> go.Figure:
    sector_data = {}
    for m, held in holdings_dict.items():
        if "sector" in held.columns:
            if strategy_type == "long_short":
                longs = held[held.get("side", pd.Series("long")) == "long"]
                counts = longs["sector"].value_counts(normalize=True)
            else:
                counts = held["sector"].value_counts(normalize=True)
            sector_data[m] = counts.to_dict()

    if not sector_data:
        return go.Figure()

    df = pd.DataFrame(sector_data).T.fillna(0).sort_index()
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], stackgroup="one", name=col, mode="lines",
        ))
    fig.update_layout(**_base_layout(
        title="Sector Allocation Over Time",
        yaxis_title="Weight", yaxis_tickformat=".0%",
    ))
    return fig


def traffic_light_dashboard(alerts: dict[str, dict]) -> go.Figure:
    names = list(alerts.keys())
    colors_map = {"green": STYLE["positive"], "amber": STYLE["warning"], "red": STYLE["negative"]}
    colors = [colors_map.get(a["status"], STYLE["muted"]) for a in alerts.values()]
    values = [a.get("value", "") for a in alerts.values()]

    fig = go.Figure(data=go.Bar(
        x=names, y=[1] * len(names),
        marker_color=colors,
        text=values, textposition="inside", textfont=dict(size=14, color="white"),
        hovertemplate="%{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title="Alert Dashboard", height=200,
        yaxis=dict(visible=False), xaxis=dict(tickfont=dict(size=11)),
        showlegend=False,
    ))
    return fig


def risk_pie_chart(risk_contributions: pd.Series, top_n: int = 10) -> go.Figure:
    if len(risk_contributions) > top_n:
        top = risk_contributions.nlargest(top_n)
        other = risk_contributions.sum() - top.sum()
        top["Other"] = other
    else:
        top = risk_contributions

    fig = go.Figure(data=go.Pie(
        labels=top.index.astype(str),
        values=top.values,
        hole=0.3,
        textinfo="percent",
        textposition="inside",
        hovertemplate="%{label}<br>Risk contribution: %{percent}<extra></extra>",
        marker=dict(line=dict(color=STYLE["bg"], width=2)),
    ))
    fig.update_layout(**_base_layout(title="Risk Contribution (%)", height=400))
    return fig
