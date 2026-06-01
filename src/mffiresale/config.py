from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for path in [cur, *cur.parents]:
        if (path / "configs" / "project.yml").exists():
            return path
    raise FileNotFoundError("Could not find configs/project.yml from current path")


def load_config(root: Path | None = None) -> dict[str, Any]:
    project_root = root or find_project_root()
    with (project_root / "configs" / "project.yml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw: Path
    interim: Path
    processed: Path
    reports: Path
    figures: Path
    tables: Path
    manifests: Path
    logs: Path

    @classmethod
    def from_config(cls, cfg: dict[str, Any], root: Path | None = None) -> "ProjectPaths":
        base = root or find_project_root()
        path_cfg = cfg["paths"]
        return cls(
            root=base,
            raw=base / path_cfg["raw"],
            interim=base / path_cfg["interim"],
            processed=base / path_cfg["processed"],
            reports=base / path_cfg["reports"],
            figures=base / path_cfg["figures"],
            tables=base / path_cfg["tables"],
            manifests=base / path_cfg["manifests"],
            logs=base / path_cfg["logs"],
        )

    def ensure(self) -> None:
        for path in [
            self.raw,
            self.interim,
            self.processed,
            self.reports,
            self.figures,
            self.tables,
            self.manifests,
            self.logs,
        ]:
            path.mkdir(parents=True, exist_ok=True)
