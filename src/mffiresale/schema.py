from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import ProjectPaths
from .io import manifest_base, write_json
from .wrds_access import connect, require_pgpass, table_columns


def run_schema_audit(cfg: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
    paths.ensure()
    pgpass = require_pgpass()
    libs_cfg = cfg["wrds"]["libraries"]
    tables_cfg = cfg["wrds"]["tables"]
    audit: dict[str, Any] = manifest_base(paths.root, "schema_audit", pgpass=pgpass)

    with connect() as db:
        libs = db.list_libraries()
        audit["library_count"] = len(libs)
        audit["libraries_present"] = {alias: lib in libs for alias, lib in libs_cfg.items()}
        audit["libraries"] = {alias: lib for alias, lib in libs_cfg.items() if lib in libs}
        audit["tables"] = {}
        for alias, table in tables_cfg.items():
            lib_alias = _library_alias_for_table(alias)
            lib = libs_cfg[lib_alias]
            if lib not in libs:
                audit["tables"][alias] = {"library": lib, "table": table, "present": False}
                continue
            table_list = db.list_tables(lib)
            present = table in table_list
            item: dict[str, Any] = {"library": lib, "table": table, "present": present}
            if present:
                cols = table_columns(db, lib, table)
                item["column_count"] = len(cols)
                item["columns"] = cols
            audit["tables"][alias] = item

    write_json(paths.manifests / "schema_audit.json", audit)
    schema_yml = paths.root / "configs" / "schema_map.yml"
    with schema_yml.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(audit, fh, sort_keys=False)
    return audit


def _library_alias_for_table(table_alias: str) -> str:
    if table_alias.startswith("mf_"):
        return "mf"
    if table_alias.startswith("crsp_"):
        return "stock"
    if table_alias.startswith("ccm_"):
        return "ccm"
    if table_alias.startswith("comp_"):
        return "comp"
    if table_alias.startswith("ff_"):
        return "ff"
    return "stock"
