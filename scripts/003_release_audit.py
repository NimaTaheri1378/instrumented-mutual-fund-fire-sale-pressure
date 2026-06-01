from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local release-readiness checks.")
    parser.add_argument("--artifact-root", default="reports/remote_proposal_grade")
    args = parser.parse_args()
    root = Path(".").resolve()
    commands = [
        [sys.executable, "scripts/public_safety_check.py"],
        [sys.executable, "-m", "pytest", "tests", "-q"],
    ]
    if (root / args.artifact_root).exists():
        commands.append([sys.executable, "scripts/proposal_completion_audit.py", "--artifact-root", args.artifact_root])
    for command in commands:
        print("+", " ".join(command))
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode:
            return int(result.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
