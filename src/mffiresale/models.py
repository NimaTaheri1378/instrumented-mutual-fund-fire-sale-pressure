from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ProjectPaths
from .extract import sample_window
from .io import manifest_base, write_json, write_parquet_atomic


FEATURES = [
    "score_pressure_reversal",
    "pressure_pred_z",
    "pressure_real_z",
    "ownership_hhi",
    "owner_count",
    "log_mktcap",
    "book_to_market",
    "profitability",
    "asset_growth",
    "leverage",
    "cash_assets",
    "momentum_12_2",
    "reversal_1m",
    "volatility_12m",
    "beta_36m",
    "turnover_lag1",
    "pressure_scaled_dvol",
    "pressure_x_illiquidity",
]


def run_all_models(cfg: dict[str, Any], paths: ProjectPaths, sample: str = "smoke") -> dict[str, Any]:
    paths.ensure()
    panel_path = paths.processed / f"model_panel_{sample}.parquet"
    panel = pd.read_parquet(panel_path)
    manifest = manifest_base(paths.root, "models", sample=sample)
    if panel.empty:
        manifest["status"] = "empty_panel"
        write_json(paths.manifests / f"models_{sample}.json", manifest)
        return manifest
    panel = _clean_panel(panel)
    manifest["fama_macbeth"] = _run_fama_macbeth(panel, cfg, paths, sample)
    manifest["iv_2sls"] = _run_iv(panel, cfg, paths, sample)
    manifest["dml_iv"] = _run_dml_iv(panel, cfg, paths, sample)
    manifest["elastic_net"] = _run_elastic_net(panel, cfg, paths, sample)
    manifest["lightgbm"] = _run_lightgbm(panel, cfg, paths, sample)
    manifest["xgboost"] = _run_xgboost(panel, cfg, paths, sample)
    manifest["torch_mlp"] = _run_torch_mlp(panel, cfg, paths, sample)
    manifest["conditional_deep"] = _run_conditional_deep(panel, cfg, paths, sample)
    write_json(paths.manifests / f"models_{sample}.json", manifest)
    return manifest


def _clean_panel(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["next_excess_ret"] = pd.to_numeric(panel["next_excess_ret"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    panel = panel[panel["next_excess_ret"].notna()]
    available = [c for c in FEATURES if c in panel.columns]
    for col in available:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return panel


def _run_fama_macbeth(panel: pd.DataFrame, cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> dict[str, Any]:
    cols = [c for c in FEATURES if c in panel.columns]
    reg_cols = ["score_pressure_reversal", "log_mktcap", "book_to_market", "momentum_12_2", "reversal_1m", "volatility_12m"]
    reg_cols = [c for c in reg_cols if c in cols]
    rows = []
    for date, g in panel.groupby("date"):
        x = g[reg_cols].replace([np.inf, -np.inf], np.nan)
        y = g["next_excess_ret"]
        ok = y.notna() & x.notna().sum(axis=1).ge(max(2, len(reg_cols) - 2))
        if ok.sum() < max(30, len(reg_cols) + 5):
            continue
        x = x.loc[ok].apply(lambda s: s.fillna(s.median()), axis=0).astype("float64")
        x = sm.add_constant(x, has_constant="add")
        try:
            fit = sm.OLS(y.loc[ok].astype("float64"), x).fit()
            row = {"date": date}
            row.update(fit.params.to_dict())
            rows.append(row)
        except Exception:
            continue
    coefs = pd.DataFrame(rows)
    if coefs.empty:
        out = paths.tables / f"fama_macbeth_monthly_coefficients_{sample}.csv"
        summary_out = paths.tables / f"fama_macbeth_summary_{sample}.csv"
        coefs.to_csv(out, index=False)
        pd.DataFrame([{"status": "too_few_monthly_cross_sections"}]).to_csv(summary_out, index=False)
        return {
            "status": "too_few_monthly_cross_sections",
            "monthly_coefficients": str(out.relative_to(paths.root)),
            "summary": str(summary_out.relative_to(paths.root)),
            "months": 0,
        }
    coefs = coefs.sort_values("date")
    out = paths.tables / f"fama_macbeth_monthly_coefficients_{sample}.csv"
    coefs.to_csv(out, index=False)
    summary_rows = []
    lags = int(cfg["modeling"]["newey_west_lags"])
    for col in [c for c in coefs.columns if c != "date"]:
        s = coefs[col].dropna()
        if len(s) < 12:
            continue
        x = np.ones((len(s), 1))
        fit = sm.OLS(s.values, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        summary_rows.append({"term": col, "coef": fit.params[0], "se": fit.bse[0], "t": fit.tvalues[0], "months": len(s)})
    summary = pd.DataFrame(summary_rows)
    summary_out = paths.tables / f"fama_macbeth_summary_{sample}.csv"
    summary.to_csv(summary_out, index=False)
    return {"monthly_coefficients": str(out.relative_to(paths.root)), "summary": str(summary_out.relative_to(paths.root)), "months": int(len(coefs))}


def _run_iv(panel: pd.DataFrame, cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> dict[str, Any]:
    controls = ["log_mktcap", "book_to_market", "momentum_12_2", "reversal_1m", "volatility_12m", "ownership_hhi"]
    controls = [c for c in controls if c in panel.columns]
    cols = ["next_excess_ret", "pressure_real_z", "pressure_pred_z", *controls, "date"]
    df = panel[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=["next_excess_ret", "pressure_real_z", "pressure_pred_z"]).copy()
    if len(df) < 500:
        return {"status": "too_few_rows", "rows": int(len(df))}
    for col in ["next_excess_ret", "pressure_real_z", "pressure_pred_z", *controls]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["next_excess_ret", "pressure_real_z", "pressure_pred_z"])
    for col in controls:
        df[col] = df[col].fillna(df[col].median())
    df = _demean_by_date(df, ["next_excess_ret", "pressure_real_z", "pressure_pred_z", *controls])
    dm_cols = ["next_excess_ret_dm", "pressure_real_z_dm", "pressure_pred_z_dm", *[c + "_dm" for c in controls]]
    for col in dm_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=dm_cols)
    if len(df) < 500:
        return {"status": "too_few_rows_after_demean", "rows": int(len(df))}
    x1 = sm.add_constant(df[["pressure_pred_z_dm", *[c + "_dm" for c in controls]]].astype("float64"), has_constant="add")
    fs = sm.OLS(df["pressure_real_z_dm"].astype("float64"), x1).fit()
    df["pressure_hat"] = fs.fittedvalues
    x2 = sm.add_constant(df[["pressure_hat", *[c + "_dm" for c in controls]]].astype("float64"), has_constant="add")
    ss = sm.OLS(df["next_excess_ret_dm"].astype("float64"), x2).fit(cov_type="HAC", cov_kwds={"maxlags": int(cfg["modeling"]["newey_west_lags"])})
    xr = sm.add_constant(df[["pressure_pred_z_dm", *[c + "_dm" for c in controls]]].astype("float64"), has_constant="add")
    rf = sm.OLS(df["next_excess_ret_dm"].astype("float64"), xr).fit(cov_type="HAC", cov_kwds={"maxlags": int(cfg["modeling"]["newey_west_lags"])})
    summary = pd.DataFrame(
        [
            {"stage": "first_stage", "term": "pressure_pred_z", "coef": fs.params.get("pressure_pred_z_dm", np.nan), "t": fs.tvalues.get("pressure_pred_z_dm", np.nan), "r2": fs.rsquared, "rows": len(df)},
            {"stage": "reduced_form", "term": "pressure_pred_z", "coef": rf.params.get("pressure_pred_z_dm", np.nan), "t": rf.tvalues.get("pressure_pred_z_dm", np.nan), "r2": rf.rsquared, "rows": len(df)},
            {"stage": "second_stage", "term": "pressure_hat", "coef": ss.params.get("pressure_hat", np.nan), "t": ss.tvalues.get("pressure_hat", np.nan), "r2": ss.rsquared, "rows": len(df)},
        ]
    )
    out = paths.tables / f"iv_2sls_summary_{sample}.csv"
    summary.to_csv(out, index=False)
    return {"summary": str(out.relative_to(paths.root)), "rows": int(len(df))}


def _demean_by_date(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col + "_dm"] = out[col] - out.groupby("date")[col].transform("mean")
    return out


def _run_dml_iv(panel: pd.DataFrame, cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> dict[str, Any]:
    controls = [c for c in FEATURES if c not in {"pressure_real_z", "pressure_pred_z"} and c in panel.columns]
    cols = ["date", "next_excess_ret", "pressure_real_z", "pressure_pred_z", *controls]
    df = panel[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=["next_excess_ret", "pressure_real_z", "pressure_pred_z"]).copy()
    if len(df) < 1000:
        return {"status": "too_few_rows", "rows": int(len(df))}
    x = df[controls]
    y = df["next_excess_ret"].values
    d = df["pressure_real_z"].values
    z = df["pressure_pred_z"].values
    imp = SimpleImputer(strategy="median")
    x_imp = imp.fit_transform(x)
    folds = KFold(n_splits=5, shuffle=True, random_state=int(cfg["project"]["random_seed"]))
    y_res = np.empty_like(y, dtype=float)
    d_res = np.empty_like(d, dtype=float)
    z_res = np.empty_like(z, dtype=float)
    for train, test in folds.split(x_imp):
        model_y = RandomForestRegressor(n_estimators=80, min_samples_leaf=50, n_jobs=-1, random_state=1)
        model_d = RandomForestRegressor(n_estimators=80, min_samples_leaf=50, n_jobs=-1, random_state=2)
        model_z = RandomForestRegressor(n_estimators=80, min_samples_leaf=50, n_jobs=-1, random_state=3)
        model_y.fit(x_imp[train], y[train])
        model_d.fit(x_imp[train], d[train])
        model_z.fit(x_imp[train], z[train])
        y_res[test] = y[test] - model_y.predict(x_imp[test])
        d_res[test] = d[test] - model_d.predict(x_imp[test])
        z_res[test] = z[test] - model_z.predict(x_imp[test])
    beta = np.sum(z_res * y_res) / np.sum(z_res * d_res)
    psi = z_res * (y_res - beta * d_res)
    se = np.sqrt(np.mean(psi**2) / (np.mean(z_res * d_res) ** 2) / len(df))
    out = paths.tables / f"dml_iv_summary_{sample}.csv"
    pd.DataFrame([{"coef": beta, "se": se, "t": beta / se, "rows": len(df)}]).to_csv(out, index=False)
    return {"summary": str(out.relative_to(paths.root)), "rows": int(len(df))}


def _time_split_masks(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.Series, pd.Series, pd.Series]:
    dev_end = pd.Period(cfg["modeling"]["development_end"], "M").to_timestamp("M")
    val_start = pd.Period(cfg["modeling"]["validation_start"], "M").to_timestamp("M")
    val_end = pd.Period(cfg["modeling"]["validation_end"], "M").to_timestamp("M")
    test_start = pd.Period(cfg["modeling"]["test_start"], "M").to_timestamp("M")
    train = df["date"].le(dev_end)
    val = df["date"].between(val_start, val_end)
    test = df["date"].ge(test_start)
    if test.sum() == 0:
        cut = df["date"].quantile(0.70)
        train = df["date"].le(cut)
        val = df["date"].gt(cut)
        test = val
    return train, val, test


def _run_elastic_net(panel: pd.DataFrame, cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> dict[str, Any]:
    cols = [c for c in FEATURES if c in panel.columns]
    df = panel.dropna(subset=["next_excess_ret"]).copy()
    train, val, test = _time_split_masks(df, cfg)
    if train.sum() < 500 or test.sum() < 100:
        return {"status": "too_few_rows", "train_rows": int(train.sum()), "test_rows": int(test.sum())}
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-5, 1, 20), cv=5, max_iter=5000, n_jobs=-1),
    )
    model.fit(df.loc[train, cols], df.loc[train, "next_excess_ret"])
    pred = model.predict(df[cols])
    out = panel[["permno", "date", "next_excess_ret"]].copy()
    out["pred_elasticnet"] = pred
    path = paths.processed / f"predictions_elasticnet_{sample}.parquet"
    write_parquet_atomic(out, path)
    metrics = _prediction_metrics(df, pred, train, val, test)
    metrics["predictions"] = str(path.relative_to(paths.root))
    return metrics


def _run_lightgbm(panel: pd.DataFrame, cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> dict[str, Any]:
    cols = [c for c in FEATURES if c in panel.columns]
    df = panel.dropna(subset=["next_excess_ret"]).copy()
    train, val, test = _time_split_masks(df, cfg)
    if train.sum() < 500 or test.sum() < 100:
        return {"status": "too_few_rows", "train_rows": int(train.sum()), "test_rows": int(test.sum())}
    x_train = df.loc[train, cols].replace([np.inf, -np.inf], np.nan)
    x_val = df.loc[val, cols].replace([np.inf, -np.inf], np.nan)
    x_all = df[cols].replace([np.inf, -np.inf], np.nan)
    med = x_train.median().fillna(0.0)
    x_train = x_train.fillna(med).astype("float64")
    x_val = x_val.fillna(med).astype("float64")
    x_all = x_all.fillna(med).astype("float64")
    y_train = pd.to_numeric(df.loc[train, "next_excess_ret"], errors="coerce").astype("float64")
    y_val = pd.to_numeric(df.loc[val, "next_excess_ret"], errors="coerce").astype("float64") if val.sum() else y_train
    gpu_used = False
    fallback = None
    try:
        import lightgbm as lgb

        params = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": int(cfg["project"]["random_seed"]),
            "device_type": "gpu",
        }
        tuning = _tune_lightgbm_params(lgb, x_train, y_train, x_val, y_val, cfg, paths, sample, params)
        params.update(tuning.get("best_params", {}))
        train_set = lgb.Dataset(x_train, label=y_train)
        val_set = lgb.Dataset(x_val, label=y_val, reference=train_set)
        try:
            model = lgb.train(
                params,
                train_set,
                num_boost_round=int(cfg["modeling"]["lightgbm_num_boost_round"]),
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(int(cfg["modeling"]["lightgbm_early_stopping_rounds"]), verbose=False)],
            )
            gpu_used = True
            pred = model.predict(x_all, num_iteration=model.best_iteration)
        except Exception as exc:
            fallback = f"lightgbm_gpu_failed: {type(exc).__name__}: {str(exc)[:160]}"
            params["device_type"] = "cpu"
            model = lgb.train(params, train_set, num_boost_round=300, valid_sets=[val_set], callbacks=[lgb.log_evaluation(0)])
            pred = model.predict(x_all)
        model_path = paths.reports / "model_artifacts" / f"lightgbm_{sample}.txt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))
        imp = pd.DataFrame({"feature": cols, "importance": model.feature_importance(importance_type="gain")})
        shap_paths = _write_lightgbm_shap_sample(model, x_all, cols, paths, sample)
    except Exception as exc:
        fallback = f"lightgbm_import_or_train_failed: {type(exc).__name__}: {str(exc)[:160]}"
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, l2_regularization=0.1, random_state=0)
        model.fit(x_train, y_train)
        pred = model.predict(x_all)
        imp = pd.DataFrame({"feature": cols, "importance": np.nan})
        model_path = None
        shap_paths = {}
    out = df[["permno", "date", "next_excess_ret"]].copy()
    out["pred_lightgbm"] = pred
    pred_path = paths.processed / f"predictions_lightgbm_{sample}.parquet"
    write_parquet_atomic(out, pred_path)
    imp_path = paths.tables / f"lightgbm_feature_importance_{sample}.csv"
    imp.sort_values("importance", ascending=False).to_csv(imp_path, index=False)
    metrics = _prediction_metrics(df, pred, train, val, test)
    metrics.update(
        {
            "predictions": str(pred_path.relative_to(paths.root)),
            "feature_importance": str(imp_path.relative_to(paths.root)),
            "model_artifact": str(model_path.relative_to(paths.root)) if model_path is not None else None,
            "shap_summary": shap_paths.get("summary"),
            "shap_sample": shap_paths.get("sample"),
            "optuna_tuning": tuning,
            "gpu_used": gpu_used,
            "fallback": fallback,
        }
    )
    return metrics


def _tune_lightgbm_params(
    lgb: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: dict[str, Any],
    paths: ProjectPaths,
    sample: str,
    base_params: dict[str, Any],
) -> dict[str, Any]:
    trials = int(cfg["modeling"].get("lightgbm_optuna_trials", 0))
    if sample == "smoke":
        trials = min(trials, 2)
    if trials <= 0 or x_val.empty:
        return {"status": "skipped", "trials": 0, "best_params": {}}
    try:
        import optuna
    except Exception as exc:
        return {"status": "optuna_missing", "trials": 0, "best_params": {}, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}

    max_rows = int(cfg["modeling"].get("lightgbm_optuna_max_rows", 250000))
    seed = int(cfg["project"]["random_seed"])
    x_tr, y_tr = _sample_rows(x_train, y_train, max_rows, seed)
    x_va, y_va = _sample_rows(x_val, y_val, max_rows // 2, seed + 1)
    if len(x_tr) < 500 or len(x_va) < 100:
        return {"status": "too_few_rows", "trials": 0, "best_params": {}}

    def objective(trial: Any) -> float:
        params = dict(base_params)
        params.update(
            {
                "device_type": "cpu",
                "verbosity": -1,
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 127),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.65, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.65, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-6, 1.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-6, 5.0, log=True),
            }
        )
        train_set = lgb.Dataset(x_tr, label=y_tr, free_raw_data=False)
        val_set = lgb.Dataset(x_va, label=y_va, reference=train_set, free_raw_data=False)
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=300,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        pred = booster.predict(x_va, num_iteration=booster.best_iteration)
        err = np.asarray(pred) - y_va.to_numpy(dtype="float64")
        return float(np.mean(err**2))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    status = "ok"
    try:
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
    except Exception as exc:
        status = f"failed: {type(exc).__name__}: {str(exc)[:160]}"
    rows = []
    for trial in study.trials:
        row = {"number": trial.number, "value": trial.value, "state": str(trial.state)}
        row.update(trial.params)
        rows.append(row)
    out = paths.tables / f"lightgbm_optuna_trials_{sample}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    try:
        best_trial = study.best_trial
        best_params = dict(best_trial.params) if best_trial.value is not None else {}
        best_value = float(best_trial.value) if best_trial.value is not None else np.nan
    except Exception:
        best_params = {}
        best_value = np.nan
    return {
        "status": status,
        "trials": len(study.trials),
        "best_value": best_value,
        "best_params": best_params,
        "trials_table": str(out.relative_to(paths.root)),
        "train_rows": int(len(x_tr)),
        "validation_rows": int(len(x_va)),
    }


def _sample_rows(x: pd.DataFrame, y: pd.Series, max_rows: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    if len(x) <= max_rows:
        return x, y
    rng = np.random.default_rng(seed)
    locs = np.sort(rng.choice(np.arange(len(x)), size=max_rows, replace=False))
    return x.iloc[locs], y.iloc[locs]


def _write_lightgbm_shap_sample(model: Any, x_all: pd.DataFrame, cols: list[str], paths: ProjectPaths, sample: str) -> dict[str, str]:
    if x_all.empty:
        return {}
    n = min(5000, len(x_all))
    x_sample = x_all.sample(n=n, random_state=20260601) if len(x_all) > n else x_all.copy()
    try:
        contrib = model.predict(x_sample, pred_contrib=True)
    except Exception:
        return {}
    contrib = np.asarray(contrib)
    if contrib.ndim != 2 or contrib.shape[1] < len(cols):
        return {}
    shap = pd.DataFrame(contrib[:, : len(cols)], columns=cols, index=x_sample.index)
    summary = pd.DataFrame(
        {
            "feature": cols,
            "mean_abs_shap": shap.abs().mean().values,
            "mean_shap": shap.mean().values,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    summary_path = paths.tables / f"lightgbm_shap_summary_{sample}.csv"
    sample_path = paths.tables / f"lightgbm_shap_sample_{sample}.parquet"
    top = summary["feature"].head(12).tolist()
    long = []
    for feature in top:
        x = pd.to_numeric(x_sample[feature], errors="coerce")
        sd = x.std(ddof=0)
        z = (x - x.mean()) / sd if np.isfinite(sd) and sd > 0 else x * 0
        long.append(
            pd.DataFrame(
                {
                    "feature": feature,
                    "shap_value": shap[feature].to_numpy(),
                    "feature_value_z": z.clip(-3, 3).to_numpy(),
                }
            )
        )
    shap_long = pd.concat(long, ignore_index=True) if long else pd.DataFrame()
    summary.to_csv(summary_path, index=False)
    write_parquet_atomic(shap_long, sample_path)
    return {
        "summary": str(summary_path.relative_to(paths.root)),
        "sample": str(sample_path.relative_to(paths.root)),
    }


def _run_xgboost(panel: pd.DataFrame, cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> dict[str, Any]:
    cols = [c for c in FEATURES if c in panel.columns]
    df = panel.dropna(subset=["next_excess_ret"]).copy()
    train, val, test = _time_split_masks(df, cfg)
    if train.sum() < 500 or test.sum() < 100:
        return {"status": "too_few_rows", "train_rows": int(train.sum()), "test_rows": int(test.sum())}
    x_train = df.loc[train, cols].replace([np.inf, -np.inf], np.nan)
    x_all = df[cols].replace([np.inf, -np.inf], np.nan)
    med = x_train.median().fillna(0.0)
    x_train = x_train.fillna(med).astype("float64")
    x_all = x_all.fillna(med).astype("float64")
    y_train = pd.to_numeric(df.loc[train, "next_excess_ret"], errors="coerce").astype("float64")
    gpu_used = False
    fallback = None
    try:
        import xgboost as xgb
        from xgboost import XGBRegressor

        build_info = getattr(xgb, "build_info", lambda: {})()
        cuda_compiled = bool(build_info.get("USE_CUDA", False) or build_info.get("CUDA_VERSION"))

        try:
            if not cuda_compiled:
                raise RuntimeError("xgboost_build_has_no_cuda")
            model = XGBRegressor(
                n_estimators=600,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                tree_method="hist",
                device="cuda",
                random_state=int(cfg["project"]["random_seed"]),
                n_jobs=8,
            )
            model.fit(x_train, y_train, verbose=False)
            pred = model.predict(x_all)
            gpu_used = True
        except Exception as exc:
            fallback = f"xgboost_gpu_failed: {type(exc).__name__}: {str(exc)[:160]}"
            model = XGBRegressor(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                tree_method="hist",
                device="cpu",
                random_state=int(cfg["project"]["random_seed"]),
                n_jobs=8,
            )
            model.fit(x_train, y_train, verbose=False)
            pred = model.predict(x_all)
        imp = pd.DataFrame({"feature": cols, "importance": model.feature_importances_})
    except Exception as exc:
        return {"status": "xgboost_missing_or_failed", "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    pred_path = paths.processed / f"predictions_xgboost_{sample}.parquet"
    out = df[["permno", "date", "next_excess_ret"]].copy()
    out["pred_xgboost"] = pred
    write_parquet_atomic(out, pred_path)
    imp_path = paths.tables / f"xgboost_feature_importance_{sample}.csv"
    imp.sort_values("importance", ascending=False).to_csv(imp_path, index=False)
    metrics = _prediction_metrics(df, pred, train, val, test)
    metrics.update(
        {
            "predictions": str(pred_path.relative_to(paths.root)),
            "feature_importance": str(imp_path.relative_to(paths.root)),
            "gpu_used": gpu_used,
            "fallback": fallback,
        }
    )
    return metrics


def _run_torch_mlp(panel: pd.DataFrame, cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> dict[str, Any]:
    cols = [c for c in FEATURES if c in panel.columns]
    df = panel.dropna(subset=["next_excess_ret"]).copy()
    train, val, test = _time_split_masks(df, cfg)
    if train.sum() < 500 or test.sum() < 100:
        return {"status": "too_few_rows", "train_rows": int(train.sum()), "test_rows": int(test.sum())}
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        return {"status": "torch_missing_or_failed", "error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    x_train = df.loc[train, cols].replace([np.inf, -np.inf], np.nan)
    x_all = df[cols].replace([np.inf, -np.inf], np.nan)
    med = x_train.median().fillna(0.0)
    x_train = x_train.fillna(med).astype("float32")
    x_all = x_all.fillna(med).astype("float32")
    mean = x_train.mean().fillna(0.0)
    std = x_train.std().replace(0, 1).fillna(1.0)
    x_train = ((x_train - mean) / std).replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    x_all = ((x_all - mean) / std).replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    x_train_np = np.nan_to_num(x_train.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x_all_np = np.nan_to_num(x_all.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_train = pd.to_numeric(df.loc[train, "next_excess_ret"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if not np.isfinite(y_std) or y_std <= 1e-12:
        y_std = 1.0
    y_train_scaled = ((y_train - y_mean) / y_std).clip(-10, 10).astype("float32")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg["project"]["random_seed"]))
    model = nn.Sequential(
        nn.Linear(len(cols), 128),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(x_train_np), torch.from_numpy(y_train_scaled.to_numpy()).reshape(-1, 1))
    loader = DataLoader(ds, batch_size=65536, shuffle=True, num_workers=0)
    epochs = 5 if sample == "smoke" else 20
    model.train()
    losses = []
    for _ in range(epochs):
        epoch_loss = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                return {"status": "nonfinite_loss", "device": str(device), "gpu_used": device.type == "cuda", "epochs": epochs}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(xb)
            seen += len(xb)
        losses.append(epoch_loss / max(seen, 1))

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_all_np), 262144):
            xb = torch.from_numpy(x_all_np[start : start + 262144]).to(device)
            preds.append(model(xb).detach().cpu().numpy().reshape(-1))
    pred = np.concatenate(preds) * y_std + y_mean if preds else np.array([])
    pred = np.nan_to_num(pred, nan=y_mean, posinf=y_mean, neginf=y_mean)
    pred_path = paths.processed / f"predictions_torch_mlp_{sample}.parquet"
    out = df[["permno", "date", "next_excess_ret"]].copy()
    out["pred_torch_mlp"] = pred
    write_parquet_atomic(out, pred_path)
    metrics = _prediction_metrics(df, pred, train, val, test)
    metrics.update(
        {
            "predictions": str(pred_path.relative_to(paths.root)),
            "device": str(device),
            "gpu_used": device.type == "cuda",
            "epochs": epochs,
            "final_train_loss": float(losses[-1]) if losses else np.nan,
        }
    )
    return metrics


def _run_conditional_deep(panel: pd.DataFrame, cfg: dict[str, Any], paths: ProjectPaths, sample: str) -> dict[str, Any]:
    cols = [c for c in FEATURES if c in panel.columns]
    df = panel.dropna(subset=["next_excess_ret"]).copy()
    train, val, test = _time_split_masks(df, cfg)
    if train.sum() < 500 or test.sum() < 100:
        return {"status": "too_few_rows", "train_rows": int(train.sum()), "test_rows": int(test.sum())}
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        return {"status": "torch_missing_or_failed", "error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    x_train = df.loc[train, cols].replace([np.inf, -np.inf], np.nan)
    x_all = df[cols].replace([np.inf, -np.inf], np.nan)
    med = x_train.median().fillna(0.0)
    x_train = x_train.fillna(med).astype("float32")
    x_all = x_all.fillna(med).astype("float32")
    mean = x_train.mean().fillna(0.0)
    std = x_train.std().replace(0, 1).fillna(1.0)
    x_train = ((x_train - mean) / std).replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    x_all = ((x_all - mean) / std).replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    x_train_np = np.nan_to_num(x_train.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x_all_np = np.nan_to_num(x_all.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_train = pd.to_numeric(df.loc[train, "next_excess_ret"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if not np.isfinite(y_std) or y_std <= 1e-12:
        y_std = 1.0
    y_scaled = ((y_train - y_mean) / y_std).clip(-10, 10).astype("float32")
    train_dates = pd.to_datetime(df.loc[train, "date"]).dt.to_period("M").astype(str)
    date_codes, date_index = pd.factorize(train_dates, sort=True)
    if len(date_index) < 12:
        return {"status": "too_few_training_months", "training_months": int(len(date_index))}

    class ConditionalDeepModel(nn.Module):
        def __init__(self, input_dim: int, months: int, latent_factors: int) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.alpha = nn.Linear(64, 1)
            self.beta = nn.Linear(64, latent_factors)
            self.factor = nn.Embedding(months, latent_factors)
            nn.init.normal_(self.factor.weight, mean=0.0, std=0.05)

        def forward(self, x: Any, month_idx: Any | None = None, factor_mean: Any | None = None) -> tuple[Any, Any, Any]:
            h = self.trunk(x)
            alpha = self.alpha(h)
            beta = self.beta(h)
            if month_idx is not None:
                lam = self.factor(month_idx)
            elif factor_mean is not None:
                lam = factor_mean.expand_as(beta)
            else:
                return alpha, alpha, beta
            pred = alpha + (beta * lam).sum(dim=1, keepdim=True)
            return pred, alpha, beta

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg["project"]["random_seed"]))
    k = int(cfg["modeling"].get("conditional_deep_latent_factors", 4))
    model = ConditionalDeepModel(len(cols), len(date_index), k).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(
        torch.from_numpy(x_train_np),
        torch.from_numpy(date_codes.astype("int64")),
        torch.from_numpy(y_scaled.to_numpy(dtype=np.float32)).reshape(-1, 1),
    )
    loader = DataLoader(ds, batch_size=65536, shuffle=True, num_workers=0)
    epochs = int(cfg["modeling"].get("conditional_deep_epochs_smoke" if sample == "smoke" else "conditional_deep_epochs_full", 12))
    model.train()
    losses = []
    for _ in range(epochs):
        total = 0.0
        seen = 0
        for xb, month_idx, yb in loader:
            xb = xb.to(device)
            month_idx = month_idx.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, alpha, beta = model(xb, month_idx)
            pricing_penalty = alpha.mean().pow(2) + 1e-4 * beta.pow(2).mean()
            loss = loss_fn(pred, yb) + 0.01 * pricing_penalty
            if not torch.isfinite(loss):
                return {"status": "nonfinite_loss", "device": str(device), "gpu_used": device.type == "cuda", "epochs": epochs}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
            seen += len(xb)
        losses.append(total / max(seen, 1))

    model.eval()
    factor_mean = model.factor.weight.detach().mean(dim=0, keepdim=True)
    preds = []
    alphas = []
    with torch.no_grad():
        for start in range(0, len(x_all_np), 262144):
            xb = torch.from_numpy(x_all_np[start : start + 262144]).to(device)
            pred_scaled, alpha_scaled, _ = model(xb, None, factor_mean)
            preds.append(pred_scaled.detach().cpu().numpy().reshape(-1))
            alphas.append(alpha_scaled.detach().cpu().numpy().reshape(-1))
    pred_scaled = np.concatenate(preds) if preds else np.array([])
    alpha_scaled = np.concatenate(alphas) if alphas else np.array([])
    pred = np.nan_to_num(pred_scaled * y_std + y_mean, nan=y_mean, posinf=y_mean, neginf=y_mean)
    alpha_pred = np.nan_to_num(alpha_scaled * y_std + y_mean, nan=y_mean, posinf=y_mean, neginf=y_mean)
    pred_path = paths.processed / f"predictions_conditional_deep_{sample}.parquet"
    out = df[["permno", "date", "next_excess_ret"]].copy()
    out["pred_conditional_deep"] = pred
    out["pred_conditional_alpha"] = alpha_pred
    write_parquet_atomic(out, pred_path)

    pe_path = paths.tables / f"conditional_deep_pricing_errors_{sample}.csv"
    _write_conditional_deep_pricing_errors(df, pred, train, val, test, pe_path)
    artifact_path = paths.reports / "model_artifacts" / f"conditional_deep_{sample}.pt"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "features": cols,
            "train_feature_mean": mean.to_dict(),
            "train_feature_std": std.to_dict(),
            "train_feature_median": med.to_dict(),
            "target_mean": y_mean,
            "target_std": y_std,
            "train_months": list(date_index),
            "latent_factors": k,
        },
        artifact_path,
    )
    metrics = _prediction_metrics(df, pred, train, val, test)
    metrics.update(
        {
            "predictions": str(pred_path.relative_to(paths.root)),
            "pricing_errors": str(pe_path.relative_to(paths.root)),
            "model_artifact": str(artifact_path.relative_to(paths.root)),
            "device": str(device),
            "gpu_used": device.type == "cuda",
            "epochs": epochs,
            "latent_factors": k,
            "final_train_loss": float(losses[-1]) if losses else np.nan,
        }
    )
    return metrics


def _write_conditional_deep_pricing_errors(
    df: pd.DataFrame,
    pred: np.ndarray,
    train: pd.Series,
    val: pd.Series,
    test: pd.Series,
    path: Path,
) -> None:
    rows = []
    work = df[["date", "next_excess_ret"]].copy()
    work["_pred"] = pred
    work["_error"] = pd.to_numeric(work["next_excess_ret"], errors="coerce") - work["_pred"]
    for name, mask in [("train", train), ("validation", val), ("test", test)]:
        g = work.loc[mask].dropna(subset=["_error"]).copy()
        if g.empty:
            rows.append({"window": name, "months": 0, "mean_pricing_error": np.nan, "mean_abs_pricing_error": np.nan, "rmse": np.nan})
            continue
        monthly = g.groupby("date")["_error"].mean()
        rows.append(
            {
                "window": name,
                "months": int(monthly.count()),
                "mean_pricing_error": float(monthly.mean()),
                "mean_abs_pricing_error": float(monthly.abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(g["_error"])))),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _prediction_metrics(df: pd.DataFrame, pred: np.ndarray, train: pd.Series, val: pd.Series, test: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, mask in [("train", train), ("validation", val), ("test", test)]:
        if mask.sum() == 0:
            continue
        metric_df = df.loc[mask, ["date", "next_excess_ret"]].copy()
        metric_df["_pred"] = pred[mask.values]
        metric_df = metric_df[np.isfinite(metric_df["next_excess_ret"]) & np.isfinite(metric_df["_pred"])]
        if len(metric_df) < 2:
            out[f"{name}_rows"] = int(len(metric_df))
            out[f"{name}_r2"] = np.nan
            out[f"{name}_rank_ic"] = np.nan
            out[f"{name}_rank_icir"] = np.nan
            continue
        out[f"{name}_rows"] = int(len(metric_df))
        out[f"{name}_r2"] = float(r2_score(metric_df["next_excess_ret"], metric_df["_pred"]))
        rank_ic = metric_df.groupby("date").apply(lambda g: g["_pred"].corr(g["next_excess_ret"], method="spearman"))
        out[f"{name}_rank_ic"] = float(rank_ic.mean(skipna=True))
        out[f"{name}_rank_icir"] = float(rank_ic.mean(skipna=True) / rank_ic.std(skipna=True)) if rank_ic.std(skipna=True) else np.nan
    return out
