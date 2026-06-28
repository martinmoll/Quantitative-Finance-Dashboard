# Handoff: Alpha Strategy Dashboard — UI Modernization

## Overview
A modernization of the existing Streamlit + Plotly quant dashboard (RMBI3110 "Alpha Strategy
Dashboard"). It keeps the Bloomberg-terminal DNA (dark, information-dense, monospace figures)
but refines it: a **softer dark palette**, an **IBM Plex type system** (Plex Sans for UI/prose,
Plex Mono for tabular figures), reusable **component variants** that replace undifferentiated
`st.metric` rows and verbose banners, and three concrete patterns for **taming the dense pages**.

## About the Design Files
`Dashboard Redesign Board.dc.html` in this bundle is a **design reference created in HTML** — a
prototype of the intended look and component behavior, **not production code to copy directly**.
Your task is to **recreate these designs inside the existing Streamlit app** (`dashboard/` package)
using its established patterns: `st.columns`, `st.sidebar`, `st.markdown(unsafe_allow_html=True)`
inline-HTML cards, `st.tabs` / `st.expander`, and Plotly `go.Figure` objects driven by a shared
dark template. Do **not** try to ship the HTML file itself. The schematic charts in the mock are
placeholders for your real Plotly figures — only their **colors, gridlines, margins, and legend
treatment** are prescriptive.

Open the file in a browser to inspect exact spacing/colors; it is a pannable canvas (four sections:
Foundations, Components, Density treatments, Page redesigns).

## Fidelity
**High-fidelity.** Colors, typography, spacing, and component structure are final. Recreate them
faithfully using Streamlit/Plotly equivalents. Where the mock shows a schematic chart, match the
chart's **style** (colors/grid/margins) but keep your existing Plotly data/logic.

---

## Design Tokens

### Color palette
| Role | Hex | Replaces (old) | Usage |
|------|-----|----------------|-------|
| Canvas / page bg | `#0B121F` | `#0A1628` | Page + Plotly `paper_bgcolor` / `plot_bgcolor` |
| Surface (cards) | `#141E2E` | `#111D2E` | Metric cards, chart containers, table rows |
| Raised | `#1B2738` | — | Hover, inactive stepper circles, pills |
| Hairline border | `rgba(255,255,255,0.07)` | — | Card borders, table row dividers |
| Gridlines (charts) | `rgba(255,255,255,0.06)` | — | Plotly `gridcolor` |
| Text primary | `#E9EEF6` | `#FFFFFF` | Headings, values (softer than pure white) |
| Text secondary | `#93A1B5` | `#8899AA` | Labels, captions |
| Text muted | `#64748B` | — | Hints, disabled, footnotes |
| Primary / blue | `#5B9BFF` | `#3b82f6` | Active stepper, links, primary lines, primary button |
| Positive / green | `#34E0A1` | `#00D26A` | Gains, completed steps, success accents |
| Negative / red | `#F46A6A` | `#FF4444` | Losses, drawdown fill/stroke, errors |
| Warning / amber | `#F5B13D` | `#FFB800` | Thresholds, moderate signals |
| Pins / series | `#A78BFA` `#2DD4BF` `#F472B6` `#FB923C` | same intent | Comparing pinned configs |

Translucent accent fills for banners/cards: use the accent hex at **8% alpha** for backgrounds and
**22% alpha** for borders (e.g. success bg `rgba(52,224,161,0.08)`, border `rgba(52,224,161,0.22)`).

### Typography
- **IBM Plex Sans** — all UI, headings, labels, and interpretation prose. Weights 400/500/600/700.
- **IBM Plex Mono** — all numeric/tabular figures, tickers, coefficients, p-values, t-stats, code
  tokens (`ret_1_xs`, `l1_ratio`). Weights 400/500/600. Always `font-variant-numeric: tabular-nums`.
- Scale: page title 26–30px/700; section heading 17–19px/600; card value 22–26px Mono/500;
  label 10–11px Sans/500 uppercase `letter-spacing:.05em`; body 12.5–13px/400 line-height 1.5.

### Spacing & shape
- Card padding 13–16px; section gap 14–16px; card radius **10px**; frame/container radius 12–14px.
- Card border `1px solid rgba(255,255,255,0.07)`. Accent-bar cards: 4px colored left bar.

---

## Implementing the tokens in Streamlit

### 1. `.streamlit/config.toml`
```toml
[theme]
base = "dark"
primaryColor = "#5B9BFF"
backgroundColor = "#0B121F"
secondaryBackgroundColor = "#141E2E"
textColor = "#E9EEF6"
font = "sans-serif"   # base UI now sans; mono is applied per-element for figures
```

### 2. Load IBM Plex once (e.g. in a shared `inject_theme()` called at the top of every page)
```python
import streamlit as st

def inject_theme():
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
      .mono { font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; }
    </style>
    """, unsafe_allow_html=True)
```

### 3. Shared Plotly template (replace the current `plotly_dark` overrides)
```python
import plotly.graph_objects as go
import plotly.io as pio

T = dict(
    paper_bgcolor="#0B121F", plot_bgcolor="#0B121F",
    font=dict(family="IBM Plex Mono, monospace", size=12, color="#93A1B5"),
    margin=dict(t=40, b=40, l=50, r=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    colorway=["#5B9BFF", "#34E0A1", "#F5B13D", "#A78BFA", "#2DD4BF", "#F472B6"],
)
pio.templates["alpha"] = go.layout.Template(layout=T)
# then: fig.update_layout(template="alpha", height=...)
```
- Strategy line `#34E0A1`, benchmark `#64748B` dashed. Drawdown: stroke `#F46A6A`, fill
  `rgba(244,106,106,0.22)`. Heatmaps: diverging green↔red around 0.

---

## Components (recreate as Python helpers in `dashboard/components/`)

### 1. Metric card — 3 variants (replaces `st.metric` rows)
Build a helper that emits inline-HTML so sentiment is encoded by **color**, not just a word.
```python
def metric_card(label, value, *, accent="#34E0A1", chip=None, variant="bar"):
    # variant "minimal" | "bar" (4px left accent) | "spark" (+ sparkline svg)
    ...
```
- **A · Minimal**: uppercase Sans label `#93A1B5`, big Mono value, optional tiny Mono delta line.
- **B · Accent bar**: 4px left bar colored by sentiment (green positive / blue neutral / red
  negative), optional pill chip top-right (e.g. "Strong" on `rgba(accent,.14)`).
- **C · Sparkline**: value + delta on one row, then a thin Plotly/SVG sparkline below.
- Lay out a row with `st.columns(6)`, one card per column. Sentiment rule (example):
  Sharpe ≥1.0 green / 0.5–1.0 blue / <0.5 amber; any drawdown/loss red.

### 2. Interpretation banner — compact with disclosure
Headline carries the **figure** (Mono, accent-colored) + a short clause; full 3–4 sentence
explainer goes inside a `st.expander("Why this matters")` so power users skip it.
```python
def banner(kind, headline_html, detail=None):  # kind: success|info|warning|error
    # bg = accent@8%, border = accent@22%, round icon chip on the left
    ...
```
Icon chips: ✓ success `#34E0A1`, i info `#5B9BFF`, ! warning `#F5B13D`, ✕ error `#F46A6A`.

### 3. Tables — hairline + conditional formatting
- Render via inline-HTML grid (preferred for control) **or** `st.dataframe` + `pandas Styler`.
- Right-align all numeric columns, Mono `tabular-nums`; row dividers `rgba(255,255,255,0.06)`;
  header row bg `#141E2E`, label `#93A1B5` uppercase.
- Significance: `**` green, `*` green, `—` muted. VIF conditional via `.style.map()`:
  `<5 #34E0A1`, `5–10 #F5B13D`, `>10 #F46A6A` (tint bg + colored left border).

### 4. Workflow stepper — refined (replaces hand-built sidebar HTML)
- **A · Horizontal**: 30px circles, completed = green `#34E0A1` + ✓, active = blue `#5B9BFF` with
  `box-shadow:0 0 0 4px rgba(91,155,255,.18)`, pending = `#1B2738` + hairline border. Connectors
  2px; completed segments green, the completed→active segment a green→blue gradient.
- **B · Compact pill** breadcrumb for tight headers: `✓ Data › ✓ Explore › ● Build › Results …`.

---

## Density treatments (choose one per dense page)
The dense pages (**Backtest Results**, **Portfolio Construction**, **Monitoring**, **VaR & Tail
Risk**) should adopt one of these instead of one long scroll:

- **A · Tabs** — `st.tabs(["Overview","Signal quality","Attribution","Turnover & costs","Compare"])`.
  Best when sub-sections are independent. **Recommended for Backtest Results** (see page redesign).
- **B · Anchored scroll + section rail** — keep one scroll but add a sticky left nav of section
  links (anchor jumps). Best when the page is read top-to-bottom as a report.
- **C · Collapsible sections** — `st.expander` per section, each showing its **headline metric in
  the label while collapsed** (e.g. "Performance — Sharpe 1.50"). Best for reference scanning.

---

## Screens / Page redesigns

### Home / Landing (`app.py`)
- **Header row**: eyebrow (Mono, blue, uppercase "RMBI3110 · Walk-forward backtester") + H1 title,
  with three compact dataset chips on the right (Stocks 101 / Months 37 / Range 2023-06→2026-06)
  as small `#141E2E` cards (`st.columns` for the row).
- **Refined horizontal stepper** below the header.
- **Two-column body** (`st.columns([1.55, 1])`): left = primary "Start here" gradient card
  (`linear-gradient(135deg,#15243c,#101a2c)`, blue border) with two buttons (Explore data → /
  Build a model); right = "Last run" resume card (strategy chip, big Mono wealth value, sparkline,
  Sharpe + MDD row, "View full results →"). Only render the resume card if a run exists in
  `st.session_state`; otherwise show an empty-state hint.

### Backtest Results (`pages/4_Backtest_Results.py`) — the dense page, fully worked
- **Header**: H1 "Backtest Results" + Sans subtitle line describing the config
  (`HGB · Long-only · expanding window · 2024-06 → 2026-06`); right side: run selector + "Pin to
  compare" primary button.
- **`st.tabs`**: Overview / Signal quality / Attribution / Turnover & costs / Compare.
- **Overview tab**:
  - KPI row of 6 accent-bar cards (`st.columns(6)`): Sharpe, Ann. Return, Ann. Vol, Max DD,
    Calmar, FF5 Alpha (α with `**` significance). Accent green/blue/red by sentiment.
  - One compact success banner with the Sharpe interpretation; full text behind the expander.
  - Charts (`st.columns([2,1])`): left = cumulative wealth (strategy green solid + SPY grey dashed)
    above a drawdown area (red); right = monthly-returns heatmap above a horizontal feature-
    importance bar chart (blue bars).
- Move IC bar + cumulative IC and IC stats into **Signal quality**; turnover/cost analysis into
  **Turnover & costs**; the strategy comparison table into **Compare**.

The same KPI-card + banner + tab/expander system applies to the other dense pages below — their
content inventory is unchanged, only the presentation system changes. Each spec keeps your existing
Plotly figures and `st.session_state` logic; you are restructuring layout and restyling, not
re-deriving numbers.

### Portfolio Construction (`pages/5_Portfolio_Construction.py`)
- **Header**: H1 "Portfolio Construction" + Sans subtitle echoing the active config
  (`HGB · Long-only · K=10 · equal_weight`). Right side: construction-method selector.
- **`st.tabs`**: Allocation / Holdings / Risk & factors / Costs / Compare methods.
- **Allocation tab**: a row of 3–4 accent-bar metric cards (e.g. # positions, effective N, top-10
  weight, net exposure), then the **sector allocation stacked area** full-width. Stacked-area
  series use the pin/series palette (`#A78BFA #2DD4BF #F472B6 #FB923C …`) at ~0.8 opacity over
  `#0B121F`; legend horizontal under the title.
- **Holdings tab**: the current-holdings table using the hairline table component — ticker (Sans)
  + Mono columns for weight / sector / score, right-aligned; weight column gets a thin inline bar
  (`#5B9BFF`) behind the number to read as a mini bar chart. Add a search/sort affordance.
- **Risk & factors tab** (`st.columns([1,1])`): left = risk-decomposition donut(s) (`hole=0.6`,
  series palette, center label = total vol in Mono); right = factor-exposure metric cards
  (β_mkt, β_smb, β_hml … colored blue neutral / amber if |exposure| beyond a band) above the
  **rolling factor-exposure line chart**. One info banner interpreting the dominant exposure.
- **Costs tab**: turnover bar chart (`#5B9BFF`, periods that breach a turnover threshold tinted
  `#F5B13D`) + a transaction-cost summary card row (gross vs net return, annual cost drag). A
  warning banner if cost drag is material.
- **Compare methods tab**: method-comparison table (equal_weight / score_weighted / risk_parity …)
  with the best cell per metric highlighted (green tint), Mono figures right-aligned.

### Monitoring & OOD Detection (`pages/6_Monitoring.py`)
This page is **alert-first** — lead with status, then detail.
- **Status header card** (full-width, the hero): overall health as a single large traffic-light
  banner — green "Healthy" `#34E0A1` / amber "Watch" `#F5B13D` / red "Retrain now" `#F46A6A`,
  using the accent@8% bg + 22% border treatment, with a one-line Mono summary
  (`drift KS 0.08 · half-life 14m · 2 stale signals`). The **retraining recommendation** lives here
  as the headline + a primary CTA button ("Trigger retrain →") when red.
- **`st.tabs`**: Drift (KS) / Alpha decay / Signal staleness.
- **Drift tab**: a row of KS metric cards (max D-stat, # features drifting, p-value) above the
  **KS D-statistic heatmap** (feature × time). Heatmap scale: low `#0B121F`→`#5B9BFF` for normal,
  switching to `#F5B13D`→`#F46A6A` past the critical D threshold; annotate the threshold in the
  colorbar title.
- **Alpha decay tab**: the **decay curve** with the half-life marker — curve `#5B9BFF`, a vertical
  dashed `#F5B13D` line at the half-life with a Mono annotation (`t½ = 14 mo`). Info banner
  interpreting whether decay is within tolerance.
- **Signal staleness tab**: horizontal staleness bar chart sorted desc; bars green→amber→red by
  days-since-refresh thresholds; a muted reference line at the staleness limit.

### VaR & Tail Risk (`pages/7_VaR_Tail_Risk.py`)
- **Header**: H1 "VaR & Tail Risk" + subtitle (`95% confidence · 1-month horizon · 10,000 sims`).
  Sidebar keeps confidence level + sim count (consistent control placement).
- **`st.tabs`**: Summary / Distribution / Backtest / Decomposition / Tail stats.
- **Summary tab**: a 4-card row (one accent-bar card per method — Historical / Parametric /
  Cornish-Fisher / Monte-Carlo), each showing VaR as the big Mono value and CVaR as a secondary
  Mono line; all red `#F46A6A` accent (these are loss figures). Below, a method-comparison table.
- **Distribution tab**: the return-distribution histogram (`#5B9BFF` bars) with **VaR and CVaR
  vertical lines** — VaR dashed `#F5B13D`, CVaR solid `#F46A6A`, each Mono-labelled; shade the
  tail beyond VaR `rgba(244,106,106,0.18)`.
- **Backtest tab**: rolling-VaR line (`#F5B13D`) with realized-loss scatter; **breach markers** as
  red `#F46A6A` dots. A Kupiec-test result card row (# breaches expected vs observed, LR stat,
  p-value) + a success/warning banner stating whether the model passes coverage.
- **Decomposition tab**: component-VaR horizontal bar chart (series palette, largest contributor
  emphasized), with a one-line interpretation of concentration.
- **Tail stats tab**: a metric-card row — skewness, excess kurtosis, Jarque-Bera (stat + p),
  Sortino — each card's accent set by whether the value is benign (blue) or concerning (amber/red,
  e.g. fat tails / strong negative skew).

### Apply-everywhere checklist (per page)
1. `inject_theme()` at the top; wrap the page title in the H1 + Sans-subtitle header pattern.
2. Replace every `st.metric` row with `metric_card(...)` in `st.columns`, sentiment-colored.
3. Wrap each `st.success/info/warning/error` block in the compact `banner(...)` + expander.
4. Restyle every `go.Figure` with `template="alpha"` and the prescribed series colors.
5. Restructure the long scroll into the page's chosen `st.tabs` (or `st.expander`) set above.
6. Route page-specific controls to the **sidebar** consistently (don't mix inline + sidebar).

---

## Screenshots
Reference renders of each board section are in `screenshots/` next to this README:
- `01-foundations.png` — palette + typography
- `02-components-cards-banners.png` — metric cards + interpretation banners
- `02-components-tables-stepper.png` — tables + workflow stepper
- `03-density.png` — the three density treatments
- `04-home.png` — Home / Landing redesign
- `04-backtest-left.png` / `04-backtest-right.png` — Backtest Results redesign

For exact spacing/colors, still prefer opening `Dashboard Redesign Board.dc.html` in a browser
(the screenshots are crops of a wider canvas).

---

## Suggested migration order
1. Land the foundations: `config.toml` + `inject_theme()` + the `alpha` Plotly template. (Global
   win, low risk — every page improves immediately.)
2. Build the 4 component helpers in `dashboard/components/` (metric_card, banner, table, stepper).
3. Refactor **Backtest Results** to the tabbed layout using those helpers (highest-impact page).
4. Roll the same treatment across the other dense pages, then Home.

## Assets
No image assets. Fonts load from Google Fonts (IBM Plex Sans + Mono). No icons library required —
the ✓ / i / ! / ✕ glyphs and ▲▼ arrows are plain Unicode in styled spans.

## Files
- `Dashboard Redesign Board.dc.html` — the full pannable design reference (all sections).
