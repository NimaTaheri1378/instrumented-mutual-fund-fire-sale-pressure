from __future__ import annotations

import numpy as np
import pandas as pd

from mffiresale.models import _sample_rows, _write_conditional_deep_pricing_errors


def test_sample_rows_is_reproducible_and_bounded():
    x = pd.DataFrame({"a": np.arange(20)})
    y = pd.Series(np.arange(20, dtype=float))
    x1, y1 = _sample_rows(x, y, 5, seed=7)
    x2, y2 = _sample_rows(x, y, 5, seed=7)
    assert len(x1) == 5
    assert x1["a"].tolist() == x2["a"].tolist()
    assert y1.tolist() == y2.tolist()


def test_conditional_deep_pricing_errors_windows(tmp_path):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"]),
            "next_excess_ret": [0.01, 0.03, -0.01, 0.02],
        }
    )
    pred = np.array([0.00, 0.02, -0.02, 0.01])
    train = pd.Series([True, True, False, False])
    val = pd.Series([False, False, True, False])
    test = pd.Series([False, False, False, True])
    out = tmp_path / "pricing_errors.csv"
    _write_conditional_deep_pricing_errors(df, pred, train, val, test, out)
    rows = pd.read_csv(out)
    assert set(rows["window"]) == {"train", "validation", "test"}
    assert rows.loc[rows["window"].eq("train"), "months"].item() == 1
