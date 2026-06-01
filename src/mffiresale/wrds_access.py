from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import wrds


def _wrds_module() -> Any:
    import wrds

    return wrds


def connect() -> Any:
    wrds = _wrds_module()
    return wrds.Connection()


def require_pgpass() -> dict[str, Any]:
    path = Path.home() / ".pgpass"
    exists = path.exists()
    mode = oct(path.stat().st_mode)[-3:] if exists else "missing"
    if not exists or mode != "600":
        raise RuntimeError(f"WRDS ~/.pgpass check failed: exists={exists}, mode={mode}")
    return {"pgpass_exists": exists, "pgpass_permissions": mode}


def table_columns(db: "wrds.Connection", library: str, table: str) -> list[str]:
    desc = db.describe_table(library, table)
    return desc["name"].astype(str).tolist()


def safe_raw_sql(db: "wrds.Connection", sql: str, date_cols: list[str] | None = None) -> pd.DataFrame:
    return db.raw_sql(sql, date_cols=date_cols or [])
