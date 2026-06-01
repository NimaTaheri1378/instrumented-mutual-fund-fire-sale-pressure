from __future__ import annotations

import numpy as np
import pandas as pd

from mffiresale.backtest import _long_short_returns


def test_market_neutral_weights_and_cost_columns():
    dates = pd.to_datetime(["2020-01-31"] * 120 + ["2020-02-29"] * 120)
    score = np.tile(np.linspace(-1, 1, 120), 2)
    df = pd.DataFrame(
        {
            "date": dates,
            "permno": list(range(120)) + list(range(60, 180)),
            "score_pressure_reversal": score,
            "next_ret": score * 0.01,
            "next_excess_ret": score * 0.01,
            "implementation_universe": True,
            "beta_36m": np.tile(np.linspace(0.8, 1.2, 120), 2),
            "log_mktcap": np.tile(np.linspace(10, 14, 120), 2),
            "siccd": np.tile([1000, 2000, 3000, 4000], 60),
            "dollar_vol_lag1": 100.0,
        }
    )
    cfg = {
        "modeling": {"top_bottom_quantile": 0.10, "transaction_cost_bps": [10]},
        "universe": {"max_adv_participation": 0.01},
    }
    out = _long_short_returns(df, "score_pressure_reversal", cfg, implementation_only=True)
    assert not out.empty
    assert out["net_exposure"].abs().max() < 1e-12
    assert np.allclose(out["gross_exposure"], 2.0)
    assert out["max_abs_weight"].max() <= 0.1
    assert "long_short_net_10bps" in out.columns
    assert out["turnover"].ge(0).all()


def test_backtest_handles_missing_neutralization_controls():
    dates = pd.to_datetime(["2020-01-31"] * 120)
    df = pd.DataFrame(
        {
            "date": dates,
            "permno": range(120),
            "score_pressure_reversal": np.linspace(-1, 1, 120),
            "next_ret": np.linspace(-1, 1, 120) * 0.01,
            "next_excess_ret": np.linspace(-1, 1, 120) * 0.01,
            "implementation_universe": True,
            "beta_36m": pd.NA,
            "log_mktcap": pd.NA,
            "siccd": pd.NA,
            "dollar_vol_lag1": 100.0,
        }
    )
    cfg = {
        "modeling": {"top_bottom_quantile": 0.10, "transaction_cost_bps": [10]},
        "universe": {"max_adv_participation": 0.01},
    }
    out = _long_short_returns(df, "score_pressure_reversal", cfg, implementation_only=True)
    assert not out.empty
    assert out["net_exposure"].abs().max() < 1e-12


def test_turnover_cap_blends_monthly_weights():
    dates = pd.to_datetime(["2020-01-31"] * 120 + ["2020-02-29"] * 120)
    score = np.r_[np.linspace(-1, 1, 120), np.linspace(1, -1, 120)]
    df = pd.DataFrame(
        {
            "date": dates,
            "permno": list(range(120)) * 2,
            "score_pressure_reversal": score,
            "next_ret": 0.01,
            "next_excess_ret": 0.01,
            "implementation_universe": True,
            "dollar_vol_lag1": 100.0,
        }
    )
    cfg = {
        "modeling": {"top_bottom_quantile": 0.10, "transaction_cost_bps": [10]},
        "universe": {"max_adv_participation": 0.01, "turnover_cap": 0.50},
    }
    out = _long_short_returns(df, "score_pressure_reversal", cfg, implementation_only=True)
    assert not out.empty
    assert out["turnover"].max() <= 0.5000001
    assert out["turnover_cap"].eq(0.50).all()
