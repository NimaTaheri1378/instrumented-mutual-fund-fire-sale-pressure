from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        return out
    except Exception:
        return None


def manifest_base(root: Path, stage: str, **extra: Any) -> dict[str, Any]:
    return {
        "stage": stage,
        "created_at_utc": utc_now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "git_commit": git_commit(root),
        **extra,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_parquet_atomic(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{int(time.time())}")
    df.to_parquet(tmp, index=False, **kwargs)
    os.replace(tmp, path)


def read_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def month_range(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Period(start, freq="M").to_timestamp("M"), pd.Period(end, freq="M").to_timestamp("M"), freq="ME")


def year_range_from_months(start: str, end: str) -> list[int]:
    return list(range(pd.Period(start, freq="M").year, pd.Period(end, freq="M").year + 1))
