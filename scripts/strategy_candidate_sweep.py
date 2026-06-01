from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


PREDICTION_FILES = {
    "xgboost": "predictions_xgboost_{sample}.parquet",
    "lightgbm": "predictions_lightgbm_{sample}.parquet",
    "torch_mlp": "predictions_torch_mlp_{sample}.parquet",
    "conditional_deep": "predictions_conditional_deep_{sample}.parquet",
}


@dataclass(frozen=True)
class Candidate:
    score: str
    quantile: float
    neutralization: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Validation-selected implementation strategy sweep.")
    parser.add_argument("--root", default=".", help="project or artifact root containing data/ and reports/")
    parser.add_argument("--sample", default="full")
    parser.add_argument("--validation-start", default="2015-01")
    parser.add_argument("--validation-end", default="2018-12")
    parser.add_argument("--test-start", default="2019-01")
    parser.add_argument("--min-capacity-musd", type=float, default=1.0)
    parser.add_argument("--max-turnover", type=float, default=1.5)
    parser.add_argument("--max-name-weight", type=float, default=0.05)
    parser.add_argument("--adv-participation", type=float, default=0.01)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    sample = args.sample
    panel = _load_panel(root, sample)
    if panel.empty:
        raise SystemExit("empty model panel")
    data = _attach_predictions(root, panel, sample)
    data = _make_scores(data)
    score_cols = [
        col
        for col in [
            "pred_xgboost",
            "pred_lightgbm",
            "pred_torch_mlp",
            "pred_conditional_deep",
            "score_ensemble_all",
            "score_ensemble_gpu",
            "score_ensemble_deep",
            "score_xgboost_pressure_blend",
            "score_ensemble_pressure_blend",
        ]
        if col in data.columns and data[col].notna().any()
    ]
    candidates = [
        Candidate(score=score, quantile=q, neutralization=neutral)
        for score in score_cols
        for q in [0.05, 0.10, 0.20]
        for neutral in ["none", "beta_size_sector"]
    ]
    factors = _load_factors(root)
    rows: list[dict[str, Any]] = []
    returns_by_candidate: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        rets = _candidate_returns(
            data,
            candidate,
            max_name_weight=args.max_name_weight,
            turnover_cap=args.max_turnover,
            adv_participation=args.adv_participation,
        )
        if rets.empty:
            continue
        cid = _candidate_id(candidate)
        returns_by_candidate[cid] = rets
        for window, start, end in [
            ("validation", args.validation_start, args.validation_end),
            ("locked_test", args.test_start, None),
        ]:
            summary = _summarize_window(rets, window, start, end, factors)
            summary.update({"candidate": cid, "score": candidate.score, "quantile": candidate.quantile, "neutralization": candidate.neutralization})
            rows.append(summary)

    out_dir = root / "reports" / "tables"
    fig_dir = root / "reports" / "figures"
    manifest_dir = root / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    sweep = pd.DataFrame(rows)
    sweep_path = out_dir / f"strategy_candidate_sweep_{sample}.csv"
    sweep.to_csv(sweep_path, index=False)

    selected = _select_candidate(sweep, args.min_capacity_musd, args.max_turnover)
    selected_path = out_dir / f"strategy_candidate_selected_{sample}.csv"
    selected.to_csv(selected_path, index=False)
    figure_paths: dict[str, str] = {}
    if not selected.empty:
        cid = str(selected.iloc[0]["candidate"])
        rets = returns_by_candidate.get(cid)
        if rets is not None and not rets.empty:
            figure_paths = _plot_selected(rets, cid, root, sample, args.test_start)

    manifest = {
        "stage": "strategy_candidate_sweep",
        "sample": sample,
        "candidate_count": len(candidates),
        "evaluated_candidate_count": int(sweep["candidate"].nunique()) if not sweep.empty else 0,
        "selection_window": f"{args.validation_start}:{args.validation_end}",
        "locked_test_start": args.test_start,
        "selection_rule": "highest validation net_25bps_sharpe subject to capacity and turnover gates",
        "min_capacity_musd": args.min_capacity_musd,
        "max_turnover": args.max_turnover,
        "sweep_table": str(sweep_path.relative_to(root)),
        "selected_table": str(selected_path.relative_to(root)),
        "figures": figure_paths,
    }
    manifest_path = manifest_dir / f"strategy_candidate_sweep_{sample}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


def _load_panel(root: Path, sample: str) -> pd.DataFrame:
    path = root / "data" / "processed" / f"model_panel_{sample}.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp("M")
    return df


def _attach_predictions(root: Path, panel: pd.DataFrame, sample: str) -> pd.DataFrame:
    out = panel.copy()
    keys = ["permno", "date", "next_excess_ret"]
    for name, template in PREDICTION_FILES.items():
        path = root / "data" / "processed" / template.format(sample=sample)
        if not path.exists():
            continue
        pred = pd.read_parquet(path)
        pred["date"] = pd.to_datetime(pred["date"]).dt.to_period("M").dt.to_timestamp("M")
        pred_cols = [c for c in pred.columns if c.startswith("pred_")]
        out = out.merge(pred[keys + pred_cols], on=keys, how="left")
    return out


def _make_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "pred_xgboost",
        "pred_lightgbm",
        "pred_torch_mlp",
        "pred_conditional_deep",
        "score_pressure_reversal",
    ]:
        if col in out:
            out[col + "_z_cs"] = _z_by_date(out, col)
    groups = {
        "score_ensemble_all": ["pred_xgboost_z_cs", "pred_lightgbm_z_cs", "pred_torch_mlp_z_cs", "pred_conditional_deep_z_cs"],
        "score_ensemble_gpu": ["pred_lightgbm_z_cs", "pred_torch_mlp_z_cs", "pred_conditional_deep_z_cs"],
        "score_ensemble_deep": ["pred_torch_mlp_z_cs", "pred_conditional_deep_z_cs"],
    }
    for name, cols in groups.items():
        have = [c for c in cols if c in out.columns]
        if have:
            out[name] = out[have].mean(axis=1)
    if {"pred_xgboost_z_cs", "score_pressure_reversal_z_cs"}.issubset(out.columns):
        out["score_xgboost_pressure_blend"] = out["pred_xgboost_z_cs"] + 0.25 * out["score_pressure_reversal_z_cs"]
    if {"score_ensemble_all", "score_pressure_reversal_z_cs"}.issubset(out.columns):
        out["score_ensemble_pressure_blend"] = out["score_ensemble_all"] + 0.25 * out["score_pressure_reversal_z_cs"]
    return out


def _z_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    x = pd.to_numeric(df[col], errors="coerce")
    mean = x.groupby(df["date"]).transform("mean")
    std = x.groupby(df["date"]).transform("std").replace(0, np.nan)
    return (x - mean) / std


def _candidate_returns(
    df: pd.DataFrame,
    candidate: Candidate,
    max_name_weight: float,
    turnover_cap: float,
    adv_participation: float,
) -> pd.DataFrame:
    needed = ["date", "permno", candidate.score, "next_ret", "next_excess_ret", "implementation_universe"]
    data = df[[c for c in needed + ["beta_36m", "log_mktcap", "siccd", "dollar_vol_lag1"] if c in df.columns]].copy()
    data = data[data[candidate.score].notna() & data["next_ret"].notna()]
    data = data[data["implementation_universe"].fillna(False)]
    rows = []
    prev_weights: dict[int, float] = {}
    for date, g in data.groupby("date", sort=True):
        if len(g) < 100:
            continue
        g = g.copy()
        g["_score"] = _neutralized_score(g, candidate.score, candidate.neutralization)
        g = g[g["_score"].notna()]
        if len(g) < 100:
            continue
        lo = g["_score"].quantile(candidate.quantile)
        hi = g["_score"].quantile(1.0 - candidate.quantile)
        long = g[g["_score"].ge(hi)]
        short = g[g["_score"].le(lo)]
        if len(long) < 10 or len(short) < 10:
            continue
        target = _candidate_weight_map(long, short, max_name_weight)
        weights_map = _blend_to_turnover_cap(prev_weights, target, turnover_cap)
        selected, weights, weights_map = _tradable_weights(g, weights_map)
        if selected.empty:
            continue
        turnover = _turnover_between(prev_weights, weights_map)
        prev_weights = weights_map
        rows.append(
            {
                "date": date,
                "candidate": _candidate_id(candidate),
                "score": candidate.score,
                "quantile": candidate.quantile,
                "neutralization": candidate.neutralization,
                "long_short_ret": float((weights * selected["next_ret"]).sum()),
                "long_short_excess": float((weights * selected["next_excess_ret"]).sum()),
                "long_n": int(len(long)),
                "short_n": int(len(short)),
                "net_exposure": float(weights.sum()),
                "gross_exposure": float(weights.abs().sum()),
                "max_abs_weight": float(weights.abs().max()),
                "turnover": float(turnover),
                "capacity_musd": _capacity_proxy_musd(selected, weights, adv_participation),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["long_short_net_25bps"] = out["long_short_ret"] - out["turnover"] * 0.0025
    out["long_short_net_50bps"] = out["long_short_ret"] - out["turnover"] * 0.0050
    return out


def _neutralized_score(g: pd.DataFrame, score_col: str, mode: str) -> pd.Series:
    y = pd.to_numeric(g[score_col], errors="coerce").astype("float64")
    if mode == "none":
        return y
    parts = []
    if mode == "beta_size_sector":
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
    design = np.column_stack([np.ones(int(ok.sum())), x.loc[ok].astype("float64").to_numpy()])
    try:
        beta = np.linalg.lstsq(design, y.loc[ok].to_numpy(dtype="float64"), rcond=None)[0]
    except np.linalg.LinAlgError:
        return y
    resid = y.copy()
    resid.loc[ok] = y.loc[ok] - design @ beta
    return resid


def _candidate_weight_map(long: pd.DataFrame, short: pd.DataFrame, max_name_weight: float) -> dict[int, float]:
    weights: dict[int, float] = {}
    for side, sign in [(long, 1.0), (short, -1.0)]:
        n = len(side)
        if n == 0:
            continue
        w = min(1.0 / n, max_name_weight)
        for permno in side["permno"].astype(int):
            weights[int(permno)] = sign * w
    return weights


def _tradable_weights(g: pd.DataFrame, weight_map: dict[int, float]) -> tuple[pd.DataFrame, pd.Series, dict[int, float]]:
    selected = g[g["permno"].astype(int).isin(weight_map)].copy()
    if selected.empty:
        return selected, pd.Series(dtype="float64"), {}
    weights = pd.Series(selected["permno"].astype(int).map(weight_map).to_numpy(dtype="float64"), index=selected.index)
    weights = weights[weights.abs().gt(1e-12)]
    selected = selected.loc[weights.index]
    return selected, weights, dict(zip(selected["permno"].astype(int), weights.astype(float)))


def _blend_to_turnover_cap(prev: dict[int, float], target: dict[int, float], cap: float) -> dict[int, float]:
    turnover = _turnover_between(prev, target)
    if turnover <= cap or turnover <= 1e-12:
        return target
    blend = max(cap - 1e-3, 0.0) / turnover
    out = {}
    for key in set(prev) | set(target):
        weight = prev.get(key, 0.0) + blend * (target.get(key, 0.0) - prev.get(key, 0.0))
        if abs(weight) > 1e-12:
            out[key] = float(weight)
    return out


def _turnover_between(prev: dict[int, float], current: dict[int, float]) -> float:
    names = set(prev) | set(current)
    return float(0.5 * sum(abs(current.get(k, 0.0) - prev.get(k, 0.0)) for k in names))


def _capacity_proxy_musd(selected: pd.DataFrame, weights: pd.Series, adv_participation: float) -> float:
    if "dollar_vol_lag1" not in selected:
        return np.nan
    adv = pd.to_numeric(selected["dollar_vol_lag1"], errors="coerce")
    ok = adv.gt(0) & weights.abs().gt(0)
    if not ok.any():
        return np.nan
    return float((adv.loc[ok] * adv_participation / weights.loc[ok].abs()).min())


def _summarize_window(rets: pd.DataFrame, window: str, start: str, end: str | None, factors: pd.DataFrame) -> dict[str, Any]:
    start_ts = pd.Period(start, "M").to_timestamp("M")
    mask = rets["date"].ge(start_ts)
    if end is not None:
        mask &= rets["date"].le(pd.Period(end, "M").to_timestamp("M"))
    g = rets.loc[mask].sort_values("date").copy()
    s = pd.to_numeric(g["long_short_ret"], errors="coerce").dropna()
    net25 = pd.to_numeric(g["long_short_net_25bps"], errors="coerce").dropna()
    net50 = pd.to_numeric(g["long_short_net_50bps"], errors="coerce").dropna()
    out: dict[str, Any] = {
        "window": window,
        "months": int(len(s)),
        "annualized_mean": _ann_mean(s),
        "sharpe": _sharpe(s),
        "t_mean": _t_mean(s),
        "net25_annualized_mean": _ann_mean(net25),
        "net25_sharpe": _sharpe(net25),
        "net50_annualized_mean": _ann_mean(net50),
        "net50_sharpe": _sharpe(net50),
        "hit_rate": float(s.gt(0).mean()) if len(s) else np.nan,
        "max_drawdown": _max_drawdown(s),
        "avg_turnover": float(g["turnover"].mean()) if len(g) else np.nan,
        "median_capacity_musd": float(g["capacity_musd"].median()) if len(g) else np.nan,
        "max_abs_weight": float(g["max_abs_weight"].max()) if len(g) else np.nan,
        "avg_gross_exposure": float(g["gross_exposure"].mean()) if len(g) else np.nan,
    }
    out.update(_factor_alpha(g, factors))
    return out


def _ann_mean(s: pd.Series) -> float:
    return float(s.mean() * 12) if len(s) else np.nan


def _sharpe(s: pd.Series) -> float:
    sd = s.std(ddof=1)
    return float((s.mean() / sd) * np.sqrt(12)) if len(s) > 1 and sd else np.nan


def _t_mean(s: pd.Series) -> float:
    sd = s.std(ddof=1)
    return float(s.mean() / (sd / np.sqrt(len(s)))) if len(s) > 1 and sd else np.nan


def _max_drawdown(s: pd.Series) -> float:
    if s.empty:
        return np.nan
    wealth = (1.0 + s.fillna(0.0)).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def _load_factors(root: Path) -> pd.DataFrame:
    path = root / "data" / "raw" / "factors" / "fivefactors_monthly.parquet"
    if not path.exists():
        return pd.DataFrame()
    ff = pd.read_parquet(path)
    ff["date"] = pd.to_datetime(ff["date"]).dt.to_period("M").dt.to_timestamp("M")
    for col in ["mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"]:
        if col in ff:
            ff[col] = pd.to_numeric(ff[col], errors="coerce")
            if ff[col].abs().median(skipna=True) > 1:
                ff[col] = ff[col] / 100.0
    return ff


def _factor_alpha(rets: pd.DataFrame, factors: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {"ff5_umd_alpha_monthly": np.nan, "ff5_umd_alpha_t": np.nan}
    if rets.empty or factors.empty:
        return out
    merged = rets[["date", "long_short_ret"]].merge(factors, on="date", how="left")
    factors_cols = [c for c in ["mktrf", "smb", "hml", "rmw", "cma", "umd"] if c in merged.columns]
    if len(factors_cols) < 3:
        return out
    x = merged[factors_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(merged["long_short_ret"], errors="coerce")
    ok = y.notna() & x.notna().all(axis=1)
    if ok.sum() < 24:
        return out
    fit = sm.OLS(y.loc[ok].astype("float64"), sm.add_constant(x.loc[ok].astype("float64"), has_constant="add")).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    out["ff5_umd_alpha_monthly"] = float(fit.params["const"])
    out["ff5_umd_alpha_t"] = float(fit.tvalues["const"])
    return out


def _select_candidate(sweep: pd.DataFrame, min_capacity: float, max_turnover: float) -> pd.DataFrame:
    if sweep.empty:
        return pd.DataFrame()
    val = sweep[sweep["window"].eq("validation")].copy()
    val = val[
        val["months"].ge(36)
        & val["median_capacity_musd"].ge(min_capacity)
        & val["avg_turnover"].le(max_turnover + 1e-8)
        & val["net25_annualized_mean"].gt(0)
    ]
    if val.empty:
        return pd.DataFrame()
    chosen = val.sort_values(["net25_sharpe", "net25_annualized_mean"], ascending=False).head(1)
    cid = str(chosen.iloc[0]["candidate"])
    return sweep[sweep["candidate"].eq(cid)].sort_values("window").reset_index(drop=True)


def _plot_selected(rets: pd.DataFrame, cid: str, root: Path, sample: str, test_start: str) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test_start_ts = pd.Period(test_start, "M").to_timestamp("M")
    g = rets[rets["date"].ge(test_start_ts)].sort_values("date").copy()
    if g.empty:
        return {}
    wealth = (1.0 + pd.to_numeric(g["long_short_ret"], errors="coerce").fillna(0.0)).cumprod()
    net25 = (1.0 + pd.to_numeric(g["long_short_net_25bps"], errors="coerce").fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(g["date"], wealth, label="Gross", color="#315c72", linewidth=1.8)
    ax.plot(g["date"], net25, label="Net, 25 bps turnover cost", color="#8f4f4f", linewidth=1.8)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_title("Validation-Selected Implementation Strategy: Locked Test")
    ax.set_ylabel("Cumulative wealth")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig_dir = root / "reports" / "figures"
    png = fig_dir / f"strategy_candidate_locked_test_{sample}.png"
    svg = fig_dir / f"strategy_candidate_locked_test_{sample}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png.relative_to(root)), "svg": str(svg.relative_to(root)), "candidate": cid}


def _candidate_id(candidate: Candidate) -> str:
    q = int(round(candidate.quantile * 100))
    return f"{candidate.score}_q{q}_{candidate.neutralization}"


if __name__ == "__main__":
    raise SystemExit(main())
