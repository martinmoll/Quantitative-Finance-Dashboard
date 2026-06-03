"""Compute fundamental features from quarterly financial statements."""

import numpy as np
import pandas as pd


def compute_fundamental_features(
    fundamentals: dict[str, dict],
) -> pd.DataFrame:
    all_rows = []

    for ticker, data in fundamentals.items():
        quarters = data.get("quarters")
        if quarters is None:
            continue

        inc = data.get("income", {})
        bal = data.get("balance", {})
        cf = data.get("cashflow", {})
        mcap = data.get("market_cap", np.nan)

        n_q = len(quarters)
        prev_sue = np.nan

        for i in range(n_q):
            ym = quarters[i].strftime("%Y-%m")
            row = {"ticker": ticker, "ym": ym}

            total_assets = _safe_get(bal, "TotalAssets", i)
            equity = _safe_get(bal, "TotalEquityGrossMinorityInterest", i)
            debt = _safe_get(bal, "TotalDebt", i)
            cash = _safe_get(bal, "CashAndCashEquivalents", i)

            revenue = _safe_get(inc, "TotalRevenue", i)
            gross_profit = _safe_get(inc, "GrossProfit", i)
            op_income = _safe_get(inc, "OperatingIncome", i)
            net_income = _safe_get(inc, "NetIncome", i)
            eps = _safe_get(inc, "DilutedEPS", i)

            ocf = _safe_get(cf, "OperatingCashFlow", i)

            rev_ttm = _ttm(inc, "TotalRevenue", i)
            ni_ttm = _ttm(inc, "NetIncome", i)
            gp_ttm = _ttm(inc, "GrossProfit", i)
            ocf_ttm = _ttm(cf, "OperatingCashFlow", i)

            if mcap and mcap > 0:
                row["bm"] = _safe_div(equity, mcap)
                row["ep"] = _safe_div(ni_ttm, mcap)
                row["cfp"] = _safe_div(ocf_ttm, mcap)
                row["sp"] = _safe_div(rev_ttm, mcap)

            if total_assets and total_assets > 0:
                row["gpa"] = _safe_div(gp_ttm, total_assets)
                row["roa"] = _safe_div(ni_ttm, total_assets)
                row["ato"] = _safe_div(rev_ttm, total_assets)
                row["cash_at"] = _safe_div(cash, total_assets)

                if ocf is not None and net_income is not None:
                    row["acc"] = (net_income - ocf) / total_assets

                if i >= 4:
                    prev_assets = _safe_get(bal, "TotalAssets", i - 4)
                    if prev_assets and prev_assets > 0:
                        row["ag"] = total_assets / prev_assets - 1

            if equity and equity > 0:
                row["roe"] = _safe_div(ni_ttm, equity)
                row["lev"] = _safe_div(debt, equity)

            shares_issued = _safe_get(bal, "ShareIssued", i)
            if i >= 4 and shares_issued is not None:
                prev_shares = _safe_get(bal, "ShareIssued", i - 4)
                if prev_shares and prev_shares > 0:
                    row["nsi"] = shares_issued / prev_shares - 1

            if i >= 4:
                prev_rev = _safe_get(inc, "TotalRevenue", i - 4)
                if prev_rev and prev_rev > 0 and revenue:
                    row["sgr"] = revenue / prev_rev - 1

            if i >= 1:
                prev_rev_qq = _safe_get(inc, "TotalRevenue", i - 1)
                if prev_rev_qq and prev_rev_qq > 0 and revenue:
                    row["rev_growth_qq"] = revenue / prev_rev_qq - 1

            if i >= 4:
                prev_eps = _safe_get(inc, "DilutedEPS", i - 4)
                if prev_eps and prev_eps != 0 and eps:
                    row["earn_growth_yoy"] = eps / prev_eps - 1

            if revenue and revenue > 0:
                row["gm_q"] = _safe_div(gross_profit, revenue)
                row["op_margin_q"] = _safe_div(op_income, revenue)
                row["ato_q"] = _safe_div(revenue, total_assets) if total_assets else np.nan

            if i >= 1:
                prev_gp = _safe_get(inc, "GrossProfit", i - 1)
                prev_rev_1 = _safe_get(inc, "TotalRevenue", i - 1)
                if prev_rev_1 and prev_rev_1 > 0 and prev_gp is not None:
                    prev_gm = prev_gp / prev_rev_1
                    curr_gm = row.get("gm_q", np.nan)
                    if not np.isnan(curr_gm):
                        row["gm_chg"] = curr_gm - prev_gm

                prev_op = _safe_get(inc, "OperatingIncome", i - 1)
                if prev_rev_1 and prev_rev_1 > 0 and prev_op is not None:
                    prev_opm = prev_op / prev_rev_1
                    curr_opm = row.get("op_margin_q", np.nan)
                    if not np.isnan(curr_opm):
                        row["op_margin_chg"] = curr_opm - prev_opm

            if i >= 4 and eps is not None:
                prev_eps_4 = _safe_get(inc, "DilutedEPS", i - 4)
                if prev_eps_4 is not None:
                    diffs = []
                    for j in range(max(0, i - 12), i):
                        e_j = _safe_get(inc, "DilutedEPS", j)
                        e_j4 = _safe_get(inc, "DilutedEPS", j - 4) if j >= 4 else None
                        if e_j is not None and e_j4 is not None:
                            diffs.append(e_j - e_j4)
                    std_diff = np.std(diffs) if len(diffs) > 1 else np.nan
                    if std_diff and std_diff > 0:
                        sue_val = (eps - prev_eps_4) / std_diff
                        row["sue_q"] = sue_val
                        row["sue"] = sue_val
                        if not np.isnan(prev_sue):
                            row["sue_chg"] = sue_val - prev_sue
                        prev_sue = sue_val

            if total_assets and total_assets > 0 and ocf is not None and net_income is not None:
                row["cfo_at"] = ocf / total_assets
                row["earn_quality"] = (ocf - net_income) / total_assets

            sga = _safe_get(inc, "SellingGeneralAndAdministration", i)
            if i >= 1 and sga is not None:
                prev_sga = _safe_get(inc, "SellingGeneralAndAdministration", i - 1)
                if prev_sga and prev_sga > 0:
                    row["sga_chg"] = sga / prev_sga - 1

            if equity and equity > 0 and net_income is not None:
                row["roe_q"] = net_income / equity
                if i >= 1:
                    prev_ni = _safe_get(inc, "NetIncome", i - 1)
                    prev_eq = _safe_get(bal, "TotalEquityGrossMinorityInterest", i - 1)
                    if prev_ni is not None and prev_eq and prev_eq > 0:
                        row["roe_chg"] = (net_income / equity) - (prev_ni / prev_eq)

            if total_assets and total_assets > 0 and ocf is not None and net_income is not None:
                row["acc_q"] = (net_income - ocf) / total_assets

            rd = _safe_get(inc, "ResearchAndDevelopment", i)
            if rd is not None and revenue and revenue > 0:
                row["rd_intensity"] = rd / revenue

            if i >= 4 and op_income is not None:
                prev_oi = _safe_get(inc, "OperatingIncome", i - 4)
                if prev_oi and prev_oi != 0:
                    row["oi_growth_yoy"] = op_income / prev_oi - 1

            inventory = _safe_get(bal, "Inventory", i)
            if i >= 1 and inventory is not None and total_assets and total_assets > 0:
                prev_inv = _safe_get(bal, "Inventory", i - 1)
                if prev_inv is not None:
                    row["inv_chg"] = (inventory - prev_inv) / total_assets

            receivables = _safe_get(bal, "AccountsReceivable", i)
            if receivables is None:
                receivables = _safe_get(bal, "Receivables", i)
            if i >= 1 and receivables is not None and total_assets and total_assets > 0:
                prev_rec = _safe_get(bal, "AccountsReceivable", i - 1)
                if prev_rec is None:
                    prev_rec = _safe_get(bal, "Receivables", i - 1)
                if prev_rec is not None:
                    row["rec_chg"] = (receivables - prev_rec) / total_assets

            all_rows.append(row)

    return pd.DataFrame(all_rows)


def _safe_get(data: dict, field: str, idx: int):
    arr = data.get(field)
    if arr is None or idx < 0 or idx >= len(arr):
        return None
    val = arr[idx]
    if isinstance(val, (int, float)) and np.isnan(val):
        return None
    return float(val)


def _safe_div(num, den):
    if num is None or den is None or den == 0:
        return np.nan
    return num / den


def _ttm(data: dict, field: str, idx: int):
    vals = []
    for j in range(max(0, idx - 3), idx + 1):
        v = _safe_get(data, field, j)
        if v is not None:
            vals.append(v)
    return sum(vals) if len(vals) == 4 else (_safe_get(data, field, idx) or np.nan)
