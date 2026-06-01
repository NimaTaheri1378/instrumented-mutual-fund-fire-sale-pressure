from __future__ import annotations

import pandas as pd

from mffiresale.features import _active_interval_join


def test_active_interval_join_keeps_only_valid_link_dates():
    left = pd.DataFrame({"permno": [1, 1, 2], "date": pd.to_datetime(["2020-01-31", "2021-01-31", "2020-06-30"])})
    intervals = pd.DataFrame(
        {
            "permno": [1, 2],
            "begdt": pd.to_datetime(["2020-06-01", "2020-01-01"]),
            "enddt": pd.to_datetime(["2021-12-31", "2020-12-31"]),
            "gvkey": ["001", "002"],
        }
    )
    out = _active_interval_join(left, intervals, "permno", "date")
    assert set(out["gvkey"]) == {"001", "002"}
    assert len(out) == 2


def test_delisting_return_compounds_with_monthly_return():
    ret = 0.10
    dlret = -0.50
    adjusted = (1 + ret) * (1 + dlret) - 1
    assert round(adjusted, 4) == -0.45
