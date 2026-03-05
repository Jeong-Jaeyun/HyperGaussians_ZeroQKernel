from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zeroqkernel.core.utils import load_yaml
from zeroqkernel.datasets.splits import build_split_strategy


def main() -> None:
    cfg = load_yaml("configs/experiment/zeroday_attack_holdout.yaml")
    strategy = build_split_strategy(cfg["split"])
    print(f"Prepared split strategy: {strategy.__class__.__name__}")


if __name__ == "__main__":
    main()
