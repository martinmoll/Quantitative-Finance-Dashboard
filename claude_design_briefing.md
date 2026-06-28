# Claude Design Briefing: Alpha Strategy Dashboard

## What This App Is

A quantitative finance dashboard for walk-forward backtesting ML-driven equity alpha models on US equities. Built with **Streamlit** (Python) and **Plotly** for all charts. The target users are finance/quant professionals and university students (RMBI3110 course at HKUST) who expect a Bloomberg-terminal-like aesthetic: information-dense, dark theme, monospace type.

The app is a multipage Streamlit app. All layout uses `st.columns`, `st.sidebar`, and `st.markdown(unsafe_allow_html=True)` for custom HTML. Charts are exclusively Plotly with a shared dark template.

---

## Current Design System

### Color Palette

| Role              | Hex       | Usage                                      |
|-------------------|-----------|---------------------------------------------|
| Background        | `#0A1628` | Page background, chart paper/plot bg        |
| Secondary BG      | `#111D2E` | Sidebar, card backgrounds                   |
| Primary/Accent    | `#3b82f6` | Main chart lines, active stepper, links     |
| CTA Green         | `#00D26A` | Positive values, completed steps, success   |
| Negative/Red      | `#FF4444` | Losses, drawdown fills, errors              |
| Warning/Amber     | `#FFB800` | Alerts, threshold lines, moderate signals   |
| Text              | `#FFFFFF` | Primary text                                |
| Muted Text        | `#8899AA` | Captions, secondary labels, inactive steps  |
| Pin Colors        | `#f59e0b`, `#8b5cf6`, `#ec4899`, `#14b8a6` | Pinned config comparisons |

### Typography

- **Font:** `monospace` everywhere (Streamlit theme + Plotly charts)
- Chart labels: 10-14px
- Stepper labels: 11px
- No custom web fonts loaded

### Streamlit Theme (`.streamlit/config.toml`)

```toml
[theme]
base = "dark"
primaryColor = "#00D26A"
backgroundColor = "#0A1628"
secondaryBackgroundColor = "#111D2E"
textColor = "#FFFFFF"
font = "monospace"
```

---

## Page Map & User Flow

The app follows a **6-step linear workflow pipeline** with a horizontal stepper rendered in the sidebar showing completion state. Later pages gate on earlier steps being completed (e.g., you can't view Backtest Results until you've run a backtest).

### Pages

1. **Home / Landing** (`app.py`) — Welcome message, dataset summary metrics (stocks, months, date range), workflow stepper, "next steps" CTA.

2. **Data Pipeline** (`0_Data_Pipeline.py`) — Manual data refresh from Yahoo Finance, FRED, Ken French. Universe selector, progress bar, pipeline log. Operational, not analytical.

3. **Data Explorer** (`1_Data_Explorer.py`) — Dataset summary stats, universe size over time (line chart), feature distribution viewer (histogram + dropdowns), Spearman correlation heatmap, missing data bar chart.

4. **Factor Analysis** (`2_Factor_Analysis.py`) — CAPM/FF3/FF5 regression tables, rolling beta with confidence bands, VIF table with color-coded thresholds, Wald joint-significance tests, market-neutral hedge calculator. Heavy on statistical output.

5. **Alpha Model Lab** (`3_Alpha_Model_Lab.py`) — The main configuration page. Model selector (6 models), hyperparameter sliders, feature selection (presets + custom multiselect), walk-forward config (OOS start, retrain frequency, window type), portfolio construction config (strategy type, K stocks, construction method). Two action buttons: "Run Backtest" and "Pin Config". Shows pinned config chips at bottom.

6. **Backtest Results** (`4_Backtest_Results.py`) — The densest page. KPI cards (Sharpe, Ann Return, Vol, MDD, Calmar, FF5 Alpha), cumulative wealth + drawdown side-by-side, IC bar chart + cumulative IC, IC stats metrics row, Fundamental Law calculator, R2 OOS, monthly returns heatmap, turnover bar chart, feature importance horizontal bar, strategy comparison table. Each metric section has colored interpretation banners.

7. **Portfolio Construction** (`5_Portfolio_Construction.py`) — Compare construction methods, sector allocation stacked area, current holdings table, risk decomposition pie charts, factor exposure metrics + rolling factor chart, turnover & transaction costs analysis, method comparison table.

8. **Monitoring & OOD Detection** (`6_Monitoring.py`) — KS test distribution shift heatmap, alpha decay curve with half-life marker, signal staleness bar chart, traffic-light alert dashboard (colored bar chart), retraining recommendation.

9. **VaR & Tail Risk** (`7_VaR_Tail_Risk.py`) — VaR/CVaR summary (4 methods), return distribution histogram with VaR/CVaR vertical lines, rolling VaR with breach markers, Kupiec test validation, component VaR bar chart, tail risk statistics (skewness, kurtosis, Jarque-Bera, Sortino).

10. **Theory & Methods** (`8_Theory_Methods.py`) — Reference/educational page. 14 expandable sections covering the quantitative theory behind each dashboard feature. Table of contents in sidebar.

---

## UI Component Inventory

These are the repeating building blocks used across pages:

### 1. KPI Metric Cards
Rows of `st.metric()` widgets in `st.columns()`. Used on almost every page. Show label, value, and optional delta indicator (green "Good" / red "Low"). Typically 3-6 per row.

### 2. Plotly Charts (dark themed)
All charts share: dark background (`#0A1628`), monospace font, `plotly_dark` template, 400px default height, compact margins (t=40, b=40, l=50, r=20). Chart types used:
- **Line charts** — cumulative wealth, rolling metrics, universe size, rolling beta, rolling factor exposure
- **Area charts** — drawdown (red fill), confidence bands (blue fill), sector allocation (stacked)
- **Bar charts** — IC, turnover, missing data, feature importance (horizontal), component VaR
- **Heatmaps** — monthly returns, correlation matrix, KS D-statistic over time
- **Histograms** — feature distributions, return distributions
- **Pie/donut charts** — risk contribution
- **Scatter** — actual losses vs VaR (with breach markers)
- **Traffic light dashboard** — colored bars with text overlay for alerts

### 3. Interpretation Banners
Colored callout boxes (`st.success`, `st.info`, `st.warning`, `st.error`) with bold metric values followed by 2-4 sentences of plain-English interpretation. These appear after every major metric section. The interpretations are context-aware (e.g., different text for Sharpe of 0.3 vs 1.5).

### 4. Theory Sections
Collapsible `st.expander` blocks containing academic explanations. Appear inline on analytical pages and collected on the Theory page. Cover quant finance concepts (CAPM, Fundamental Law, VaR, etc.).

### 5. Data Tables
`st.dataframe()` with `use_container_width=True`. Some use `.style.map()` for conditional coloring (e.g., VIF thresholds: green <5, amber 5-10, red >10). Regression tables show coefficient, HAC SE, t-stat, p-value, significance stars.

### 6. Workflow Stepper
Custom HTML rendered via `st.markdown(unsafe_allow_html=True)` in the sidebar. Shows 6 numbered circles connected by lines. States: completed (green circle + checkmark), active (blue circle), pending (gray circle). Labels below each circle.

### 7. Empty States
When required data is missing: centered icon (microscope emoji), gray heading and description, checklist of required vs available data, primary CTA button to the prerequisite page.

### 8. Sidebar Controls
Used for page-specific settings: stock/sector selectors, confidence levels, rolling window sliders, simulation count inputs. Some pages also use inline controls (sliders, number inputs, select boxes) in `st.columns`.

### 9. Pinned Config Chips
On the Alpha Model Lab page: button-style chips showing pinned strategy configurations with "X" to remove. Limited to 4 pins.

### 10. Progress Indicators
`st.progress()` bars + `st.status()` containers during long-running operations (data pipeline, backtest execution).

---

## Current Layout Patterns

- **Side-by-side charts:** Two charts in `st.columns(2)` is the most common pattern (wealth + drawdown, IC + cumulative IC, heatmap + turnover).
- **Metrics row + interpretation:** A row of 3-6 `st.metric` widgets followed by a colored interpretation banner.
- **Config panel → results:** Left column for controls, right column for output (Data Explorer feature viewer, Fundamental Law calculator).
- **Section separators:** `st.markdown("---")` between every major section, with `st.header()` for section titles.
- **Full-width charts:** Single charts spanning the full page width (correlation heatmap, sector allocation, rolling factor exposure, KS heatmap).

---

## What Each Page Looks Like (Summary for Design Reference)

| Page | Density | Primary Elements |
|------|---------|-----------------|
| Home | Low | 3 metric cards, welcome text, next-steps CTA |
| Data Pipeline | Low | Universe selector, API key input, progress bar, log |
| Data Explorer | Medium | 4 metrics, line chart, histogram with selectors, correlation heatmap, missing data bars |
| Factor Analysis | High | Regression table, rolling beta line chart, VIF table, Wald test buttons, hedge calculator |
| Alpha Model Lab | Medium | Many selectors/sliders organized in columns, two action buttons, pinned config chips |
| Backtest Results | Very High | 5-6 KPI cards, 2 charts, 4 IC metrics, heatmap, turnover, feature importance, comparison table, multiple interpretation banners |
| Portfolio Construction | High | Method comparison, sector stacked area, holdings table, risk pie charts, factor metrics, rolling chart, cost analysis, comparison table |
| Monitoring | High | KS metrics + table, KS heatmap, decay curve, staleness bars, traffic light dashboard, retraining recommendation |
| VaR & Tail Risk | High | 8 metric cards (2 rows), comparison table, histogram with VaR lines, rolling VaR scatter, Kupiec metrics, component VaR bars, tail stats |
| Theory & Methods | Low | Text-heavy, 14 expandable sections, sidebar TOC |

---

## Known Design Pain Points / Things to Consider

- **Information overload:** The Backtest Results and Portfolio Construction pages pack a lot of metrics, charts, interpretation banners, and tables into a single scroll. Could benefit from better visual hierarchy or tabbed sub-sections.
- **Repetitive metric rows:** Many pages have similar-looking rows of `st.metric` cards with no visual distinction between them.
- **Interpretation banners are verbose:** The colored callout boxes contain 3-4 sentences of interpretation text. These are educational but may feel heavy for experienced users.
- **Workflow stepper is custom HTML:** The sidebar stepper is hand-built with inline styles. It works but could be more polished.
- **Limited interactivity between charts:** Charts are static Plotly figures — no cross-filtering or linked brushing between related visualizations.
- **No dark/light mode toggle:** The app is dark-only. The Bloomberg aesthetic is intentional but some users may prefer a lighter theme.
- **Sidebar gets crowded:** Some pages (Factor Analysis, VaR) put controls in the sidebar while others put them inline, creating inconsistency.
- **Tables are plain:** `st.dataframe` with minimal styling. The regression and comparison tables could benefit from better formatting.

---

## Tech Constraints

- **Streamlit** — layout is grid-based (`st.columns`), no CSS Grid/Flexbox beyond what Streamlit provides. Custom styling only via `st.markdown(unsafe_allow_html=True)` with inline styles.
- **Plotly** — all charts are Plotly `go.Figure` objects rendered via `st.plotly_chart`. No D3, no Altair, no Matplotlib.
- **No external CSS/JS** — styling is either Streamlit theme config or inline HTML/CSS.
- **Multipage app** — each page is a separate `.py` file in `dashboard/pages/`. Shared components live in `dashboard/components/`.
- **Session state** — data flows between pages via `st.session_state`. Pages check for required state and show empty states if missing.
