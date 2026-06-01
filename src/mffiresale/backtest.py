from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import ProjectPaths
from .io import manifest_base, write_json, write_parquet_atomic


def run_backtests(cfg: dict[str, Any], paths: ProjectPaths, sample: str = "smoke") -> dict[str, Any]:
    paths.ensure()
    panel = pd.read_parquet(paths.processed / f"model_panel_{sample}.parquet")
    if panel.empty:
        manifest = manifest_base(paths.root, "backtest", sample=sample, status="empty_panel")
        write_json(paths.manifests / f"backtest_{sample}.json", manifest)
        return manifest
    preds = _load_predictions(paths, sample)
    panel["date"] = pd.to_datetime(panel["date"])
    data = panel.merge(preds, on=["permno", "date", "next_excess_ret"], how="left")
    portfolios = []
    score_cols = [
        "score_pressure_reversal",
        "pred_elasticnet",
        "pred_lightgbm",
        "pred_xgboost",
        "pred_torch_mlp",
        "pred_conditional_deep",
    ]
    for col in score_cols:
        if col in data.columns:
            portfolios.append(_long_short_returns(data, col, cfg, implementation_only=False))
            portfolios.append(_long_short_returns(data, col, cfg, implementation_only=True))
    rets = pd.concat([x for x in portfolios if not x.empty], ignore_index=True) if portfolios else pd.DataFrame()
    ret_path = paths.processed / f"portfolio_returns_{sample}.parquet"
    write_parquet_atomic(rets, ret_path)
    summary = _summarize_portfolios(rets, paths, sample)
    manifest = manifest_base(
        paths.root,
        "backtest",
        sample=sample,
        returns=str(ret_path.relative_to(paths.root)),
        summary=summary.get("path"),
        portfolio_count=int(rets["portfolio"].nunique()) if len(rets) else 0,
    )
    write_json(paths.manifests / f"backtest_{sample}.json", manifest)
    return manifest


def _load_predictions(paths: ProjectPaths, sample: str) -> pd.DataFrame:
    base: pd.DataFrame | None = None
    for name in ["elasticnet", "lightgbm", "xgboost", "torch_mlp", "conditional_deep"]:
        path = paths.processed / f"predictions_{name}_{sample}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        if base is None:
            base = df
        else:
            base = base.merge(df, on=["permno", "date", "next_excess_ret"], how="outer")
    return base if base is not None else pd.DataFrame(columns=["permno", "date", "next_excess_ret"])


def _long_short_returns(df: pd.DataFrame, score_col: str, cfg: dict[str, Any], implementation_only: bool) -> pd.DataFrame:
    q = float(cfg["modeling"]["top_bottom_quantile"])
    adv_cap = float(cfg["universe"].get("max_adv_participation", 0.01))
    max_name_weight = float(cfg["universe"].get("max_name_weight", 1.0))
    turnover_cap = float(cfg["universe"].get("turnover_cap", np.inf))
    data = df[df[score_col].notna() & df["next_ret"].notna()].copy()
    if implementation_only and "implementation_universe" in data:
        data = data[data["implementation_universe"].fillna(False)]
    rows = []
    prev_weights: dict[int, float] = {}
    for date, g in data.groupby("date"):
        if len(g) < 50:
            continue
        g = g.copy()
        g["_rank_score"] = _neutralized_score(g, score_col)
        lo = g["_rank_score"].quantile(q)
        hi = g["_rank_score"].quantile(1.0 - q)
        long = g[g["_rank_score"] >= hi]
        short = g[g["_rank_score"] <= lo]
        if len(long) < 10 or len(short) < 10:
            continue
        target_weights = _candidate_weight_map(long, short, max_name_weight)
        capped_weights = _blend_to_turnover_cap(prev_weights, target_weights, turnover_cap)
        selected, weights, weight_map = _tradable_weights(g, capped_weights)
        if selected.empty:
            continue
        turnover = _turnover_between(prev_weights, weight_map)
        if np.isfinite(turnover_cap) and turnover > turnover_cap:
            capped_weights = _blend_to_turnover_cap(prev_weights, weight_map, turnover_cap)
            selected, weights, weight_map = _tradable_weights(g, capped_weights)
            turnover = _turnover_between(prev_weights, weight_map)
        if selected.empty:
            continue
        long_w = weights[weights.gt(0)]
        short_w = weights[weights.lt(0)]
        long_ret = float((long_w * selected.loc[long_w.index, "next_ret"]).sum()) if len(long_w) else 0.0
        short_ret = float((-short_w * selected.loc[short_w.index, "next_ret"]).sum()) if len(short_w) else 0.0
        long_excess = float((long_w * selected.loc[long_w.index, "next_excess_ret"]).sum()) if len(long_w) else 0.0
        short_excess = float((-short_w * selected.loc[short_w.index, "next_excess_ret"]).sum()) if len(short_w) else 0.0
        prev_weights = weight_map
        capacity_musd = _capacity_proxy_musd(selected, weights, adv_cap)
        rows.append(
            {
                "date": date,
                "portfolio": f"{score_col}_{'implementation' if implementation_only else 'research'}",
                "score": score_col,
                "implementation_only": implementation_only,
                "long_ret": long_ret,
                "short_ret": short_ret,
                "long_short_ret": float((weights * selected["next_ret"]).sum()),
                "long_short_excess": float((weights * selected["next_excess_ret"]).sum()),
                "long_n": len(long),
                "short_n": len(short),
                "net_exposure": float(weights.sum()),
                "gross_exposure": float(weights.abs().sum()),
                "max_abs_weight": float(weights.abs().max()),
                "turnover": turnover,
                "turnover_cap": turnover_cap if np.isfinite(turnover_cap) else np.nan,
                "max_name_weight": max_name_weight,
                "beta_exposure": _weighted_sum(selected, weights, "beta_36m"),
                "size_exposure": _weighted_sum(selected, weights, "log_mktcap"),
                "capacity_musd": capacity_musd,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for bps in cfg["modeling"]["transaction_cost_bps"]:
        out[f"long_short_net_{bps}bps"] = out["long_short_ret"] - out["turnover"] * (bps / 10_000.0)
    return out


def _candidate_weight_map(long: pd.DataFrame, short: pd.DataFrame, max_name_weight: float) -> dict[int, float]:
    weights: dict[int, float] = {}
    cap = max_name_weight if np.isfinite(max_name_weight) and max_name_weight > 0 else 1.0
    for side, sign in [(long, 1.0), (short, -1.0)]:
        n = len(side)
        if n == 0:
            continue
        weight = min(1.0 / n, cap)
        weights.update({int(permno): sign * weight for permno in side["permno"].astype(int)})
    return weights


def _tradable_weights(g: pd.DataFrame, weight_map: dict[int, float]) -> tuple[pd.DataFrame, pd.Series, dict[int, float]]:
    selected = g[g["permno"].astype(int).isin(weight_map)].copy()
    if selected.empty:
        return selected, pd.Series(dtype="float64"), {}
    weights = pd.Series(selected["permno"].astype(int).map(weight_map).to_numpy(dtype="float64"), index=selected.index)
    weights = weights[weights.abs().gt(1e-12)]
    selected = selected.loc[weights.index]
    actual = dict(zip(selected["permno"].astype(int), weights.astype(float)))
    return selected, weights, actual


def _blend_to_turnover_cap(prev: dict[int, float], target: dict[int, float], cap: float) -> dict[int, float]:
    if not np.isfinite(cap) or cap <= 0:
        return target
    turnover = _turnover_between(prev, target)
    if turnover <= cap or turnover <= 1e-12:
        return target
    effective_cap = max(cap - 1e-3, 0.0)
    blend = effective_cap / turnover
    out = {}
    for key in set(prev) | set(target):
        weight = prev.get(key, 0.0) + blend * (target.get(key, 0.0) - prev.get(key, 0.0))
        if abs(weight) > 1e-12:
            out[key] = float(weight)
    return out


def _neutralized_score(g: pd.DataFrame, score_col: str) -> pd.Series:
    y = pd.to_numeric(g[score_col], errors="coerce").astype("float64")
    parts = []
    for col in ["beta_36m", "log_mktcap"]:
        if col in g:
            x = pd.to_numeric(g[col], errors="coerce")
            sd = float(x.std(ddof=0)) if x.notna().any() else np.nan
            if np.isfinite(sd) and sd > 0:
                parts.append(((x - x.mean()) / sd).rename(col))
    if "siccd" in g:
        sector = (pd.to_numeric(g["siccd"], errors="coerce") // 1000).astype("Int64")
        dummies = pd.get_dummies(sector, prefix="sic", dummy_na=False)
        if 1 < dummies.shape[1] <= 12:
            parts.append(dummies.iloc[:, 1:].astype("float64"))
    if not parts:
        return y
    x = pd.concat(parts, axis=1)
    ok = y.notna() & x.notna().all(axis=1)
    if ok.sum() < max(50, x.shape[1] + 5):
        return y
    x_ok = x.loc[ok].astype("float64")
    design = np.column_stack([np.ones(len(x_ok)), x_ok.to_numpy()])
    try:
        beta = np.linalg.lstsq(design, y.loc[ok].to_numpy(dtype="float64"), rcond=None)[0]
    except np.linalg.LinAlgError:
        return y
    resid = y.copy()
    resid.loc[ok] = y.loc[ok] - design @ beta
    return resid


def _weighted_sum(selected: pd.DataFrame, weights: pd.Series, col: str) -> float:
    if col not in selected:
        return np.nan
    x = pd.to_numeric(selected[col], errors="coerce")
    ok = x.notna()
    return float((weights.loc[ok] * x.loc[ok]).sum()) if ok.any() else np.nan


def _capacity_proxy_musd(selected: pd.DataFrame, weights: pd.Series, adv_cap: float) -> float:
    if "dollar_vol_lag1" not in selected:
        return np.nan
    adv = pd.to_numeric(selected["dollar_vol_lag1"], errors="coerce")
    ok = adv.gt(0) & weights.abs().gt(0)
    if not ok.any():
        return np.nan
    return float((adv.loc[ok] * adv_cap / weights.loc[ok].abs()).min())


def _turnover_series(weights_by_date: list[dict[int, float]]) -> list[float]:
    prev: dict[int, float] = {}
    out = []
    for current in weights_by_date:
        out.append(_turnover_between(prev, current))
        prev = current
    return out


def _turnover_between(prev: dict[int, float], current: dict[int, float]) -> float:
    names = set(prev) | set(current)
    return float(0.5 * sum(abs(current.get(k, 0.0) - prev.get(k, 0.0)) for k in names))


def _summarize_portfolios(rets: pd.DataFrame, paths: ProjectPaths, sample: str) -> dict[str, Any]:
    if rets.empty:
        return {"status": "empty"}
    ff = pd.read_parquet(paths.raw / "factors" / "fivefactors_monthly.parquet")
    ff["date"] = pd.to_datetime(ff["date"]).dt.to_period("M").dt.to_timestamp("M")
    for col in ["mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"]:
        if col in ff:
            ff[col] = pd.to_numeric(ff[col], errors="coerce")
            if ff[col].abs().median(skipna=True) > 1:
                ff[col] = ff[col] / 100.0
    rows = []
    cost_rows = []
    for name, g in rets.groupby("portfolio"):
        s = g.sort_values("date")["long_short_ret"].dropna()
        if len(s) < 6:
            continue
        downside = s[s.lt(0)].std(ddof=1)
        boot = _moving_block_mean_ci(s)
        row = {
            "portfolio": name,
            "months": len(s),
            "mean_monthly": s.mean(),
            "annualized_mean": s.mean() * 12,
            "annualized_vol": s.std(ddof=1) * np.sqrt(12),
            "sharpe": (s.mean() / s.std(ddof=1)) * np.sqrt(12) if s.std(ddof=1) else np.nan,
            "sortino": (s.mean() / downside) * np.sqrt(12) if downside and np.isfinite(downside) else np.nan,
            "t_mean": s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if s.std(ddof=1) else np.nan,
            "block_boot_mean_lo": boot[0],
            "block_boot_mean_hi": boot[1],
            "max_drawdown": _max_drawdown(s),
            "avg_turnover": g["turnover"].mean() if "turnover" in g else np.nan,
            "median_capacity_musd": g["capacity_musd"].median() if "capacity_musd" in g else np.nan,
            "avg_abs_beta_exposure": g["beta_exposure"].abs().mean() if "beta_exposure" in g else np.nan,
            "avg_abs_size_exposure": g["size_exposure"].abs().mean() if "size_exposure" in g else np.nan,
        }
        merged = g[["date", "long_short_ret"]].merge(ff, on="date", how="left").dropna(subset=["long_short_ret", "mktrf"])
        if len(merged) >= 12:
            for factors, label in [
                (["mktrf"], "capm"),
                (["mktrf", "smb", "hml"], "ff3"),
                (["mktrf", "smb", "hml", "rmw", "cma"], "ff5"),
                (["mktrf", "smb", "hml", "rmw", "cma", "umd"], "ff5_umd"),
            ]:
                avail = [f for f in factors if f in merged.columns]
                x = merged[avail].apply(pd.to_numeric, errors="coerce")
                y = pd.to_numeric(merged["long_short_ret"], errors="coerce")
                ok = y.notna() & x.notna().all(axis=1)
                if ok.sum() < 12:
                    continue
                x = sm.add_constant(x.loc[ok].astype("float64"), has_constant="add")
                fit = sm.OLS(y.loc[ok].astype("float64"), x).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
                row[f"{label}_alpha_monthly"] = float(fit.params["const"])
                row[f"{label}_alpha_t"] = float(fit.tvalues["const"])
        rows.append(row)
        for col in [c for c in g.columns if c.startswith("long_short_net_") and c.endswith("bps")]:
            net = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(net) < 6:
                continue
            bps = int(col.replace("long_short_net_", "").replace("bps", ""))
            cost_rows.append(
                {
                    "portfolio": name,
                    "cost_bps_per_turnover": bps,
                    "annualized_net_mean": float(net.mean() * 12),
                    "net_sharpe": float((net.mean() / net.std(ddof=1)) * np.sqrt(12)) if net.std(ddof=1) else np.nan,
                    "months": int(len(net)),
                }
            )
    summary = pd.DataFrame(rows)
    out = paths.tables / f"portfolio_summary_{sample}.csv"
    summary.to_csv(out, index=False)
    if cost_rows:
        pd.DataFrame(cost_rows).to_csv(paths.tables / f"transaction_costs_{sample}.csv", index=False)
    return {"path": str(out.relative_to(paths.root)), "rows": int(len(summary))}


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def _moving_block_mean_ci(s: pd.Series, block: int = 6, draws: int = 500) -> tuple[float, float]:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype="float64")
    if len(x) < block + 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(20260601)
    starts = np.arange(0, len(x) - block + 1)
    means = []
    blocks_needed = int(np.ceil(len(x) / block))
    for _ in range(draws):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([x[i : i + block] for i in chosen])[: len(x)]
        means.append(sample.mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return (float(lo), float(hi))
