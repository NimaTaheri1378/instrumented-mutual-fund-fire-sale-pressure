from __future__ import annotations

import numpy as np
import pandas as pd

from mffiresale.deliverables import _future_compounded_returns


def test_future_compounded_returns_require_consecutive_months():
    panel = pd.DataFrame(
        {
            "permno": [1, 1, 1, 2, 2],
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-01-31", "2020-03-31"]),
            "next_excess_ret": [0.10, 0.20, 0.30, 0.50, 0.60],
        }
    )
    out = _future_compounded_returns(panel, 2)
    assert np.isclose(out.iloc[0], (1.10 * 1.20) - 1.0)
    assert np.isclose(out.iloc[1], (1.20 * 1.30) - 1.0)
    assert np.isnan(out.iloc[2])
    assert np.isnan(out.iloc[3])
