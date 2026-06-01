from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "data",
    "logs",
    "manifests",
    "reports",
    "site",
}
SKIP_PREFIXES: tuple[Path, ...] = ()
SKIP_SUFFIXES = {
    ".parquet",
    ".pyc",
    ".pyo",
    ".png",
    ".svg",
    ".tgz",
    ".zip",
}

CHECKS = {
    "private_api_key_assignment": re.compile(r"(?i)\b(api[_ -]?key|secret|token)\b\s*[:=]\s*[A-Za-z0-9_./+=-]{16,}"),
    "password_assignment": re.compile(r"(?i)\bpassword\b\s*[:=]\s*\S{6,}"),
    "postgres_pgpass_entry": re.compile(r"(?i)(wrds|postgres).*:.*:.*:.*:.+"),
    "hardcoded_slurm_jobid": re.compile(r"srun\s+--jobid=\d+", re.IGNORECASE),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}

ALLOWED_LINES = {
    "Never commit WRDS credentials, API keys, passwords, `.pgpass`, raw WRDS extracts, or logs with sensitive content.",
    "The core project is WRDS-only. External API keys are intentionally not used.",
    "If WRDS authentication fails or MFA is not accepted, stop WRDS work and wait.",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check repository files for public-release hazards.")
    parser.add_argument("--root", default=".", help="repository root")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    findings: list[tuple[str, str, int, str]] = []

    for path in _iter_files(root):
        rel = path.relative_to(root)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped in ALLOWED_LINES:
                continue
            for name, pattern in CHECKS.items():
                if pattern.search(line):
                    findings.append((name, str(rel), lineno, stripped[:180]))

    if findings:
        for name, rel, lineno, line in findings:
            print(f"{name}: {rel}:{lineno}: {line}", file=sys.stderr)
        return 1
    print("public_safety_check: pass")
    return 0


def _iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel == Path("scripts") / "public_safety_check.py":
            continue
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if any(_is_relative_to(rel, prefix) for prefix in SKIP_PREFIXES):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        yield path


def _is_relative_to(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
