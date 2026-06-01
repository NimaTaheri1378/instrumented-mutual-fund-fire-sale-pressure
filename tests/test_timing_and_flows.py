from __future__ import annotations

import pandas as pd


def test_flow_formula_fixture():
    mtna_lag = 100.0
    ret = 0.10
    mtna = 99.0
    flow = (mtna - mtna_lag * (1 + ret)) / (mtna_lag * (1 + ret))
    assert round(flow, 4) == -0.1


def test_holdings_lag_activation_rule():
    report_dt = pd.Timestamp("2020-03-31")
    formation_month = pd.Timestamp("2020-05-31")
    assert report_dt + pd.Timedelta(days=60) <= formation_month
    too_early = pd.Timestamp("2020-04-30")
    assert report_dt + pd.Timedelta(days=60) > too_early
