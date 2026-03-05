from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zeroqkernel.reports.export import export_artifacts


def main() -> None:
    export_artifacts("results/runs/latest", "results/runs/latest/export")


if __name__ == "__main__":
    main()
