# Dashboard Overhaul — Design Spec

## Goal

Reposition this repository from a class assignment (RMBI3110) to a general-purpose quantitative finance dashboard. Three changes:

1. **Remove assignment framing** — rewrite README, archive assignment-specific docs
2. **Gitignore notebooks** — keep them locally but stop tracking
3. **Add long-short strategy** — new strategy type alongside existing long-only

---

## 1. Repository Cleanup

### README.md

Complete rewrite. New framing as a personal Streamlit backtesting dashboard for ML-driven equity strategies. Retains technical content (dataset, features, models, walk-forward design) but removes all references to RMBI3110, HKUST, grading rubric, instructor, and assignment deadlines.

### .gitignore

Add `Code/*.ipynb` to stop tracking notebooks. They remain in the working directory for reference.

### Docs Archive

Move to `docs/archive/`:
- `progress.md`
- `mdd_reduction_log.md`
- `karpathy_guidelines_summary.md`
- `docs/superpowers/specs/2026-04-25-alpha-strategy-design.md`
- `docs/superpowers/specs/2026-05-09-bloomberg-dashboard-design.md`
- `docs/superpowers/plans/2026-04-25-alpha-strategy-plan.md`
- `docs/superpowers/plans/2026-05-09-bloomberg-dashboard-plan.md`

---

## 2. Long-Short Strategy

### UI Changes (`app.py`)

- New dropdown in top control bar: **Strategy** with options `Long Only`, `Long-Short`
- When `Long-Short` selected: show `K_short` slider (5–50, step 5, default = K value)
- Existing `K` slider label unchanged (serves as K_long in long-short mode)

### Engine Changes (`engine.py` → `build_portfolio()`)

New parameters:
- `strategy_type: str` — `"long_only"` (default) or `"long_short"`
- `K_short: int` — number of stocks in the short leg (default 10)

Behavior when `strategy_type == "long_short"`:
- Long leg: top K stocks by adjusted prediction (same as current)
- Short leg: bottom K_short stocks by adjusted prediction
- Monthly return: `mean(long leg y_raw) - mean(short leg y_raw)`
- Holdings dict stores both legs with a `side` column (`"long"` / `"short"`)
- Turnover computed separately for each leg, reported as combined
- IC unchanged (computed on full universe predictions)

When `strategy_type == "long_only"`: existing behavior, no changes.

### Cache Changes (`cache_manager.py`)

Portfolio cache key includes `strategy_type` and `K_short`. Prediction cache unchanged.

### Chart Updates

- **Sector allocation**: shows both legs when long-short
- **Holdings table**: adds "Side" column when long-short
- **All other charts** (KPIs, wealth, drawdown, rolling SR, IC, heatmap, turnover): work unchanged since they operate on the `monthly_returns` series

---

## 3. Memory Updates

Update project memory files to reflect the new project identity (remove assignment references from memory index).

---

## Out of Scope

- Renaming Python files or directories
- Cleaning up assignment references inside Python docstrings/comments
- Adding tests
- Live data (yfinance) integration
- Deployment
