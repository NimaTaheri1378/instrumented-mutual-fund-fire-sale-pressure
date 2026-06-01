from __future__ import annotations

import pandas as pd

from mffiresale.models import _time_split_masks


def test_model_windows_do_not_overlap_labels():
    dates = pd.date_range("2004-01-31", "2025-12-31", freq="ME")
    df = pd.DataFrame({"date": dates})
    cfg = {
        "modeling": {
            "development_end": "2014-12",
            "validation_start": "2015-01",
            "validation_end": "2018-12",
            "test_start": "2019-01",
        }
    }
    train, val, test = _time_split_masks(df, cfg)
    assert not (train & val).any()
    assert not (train & test).any()
    assert not (val & test).any()
    assert df.loc[train, "date"].max() < df.loc[val, "date"].min()
    assert df.loc[val, "date"].max() < df.loc[test, "date"].min()
