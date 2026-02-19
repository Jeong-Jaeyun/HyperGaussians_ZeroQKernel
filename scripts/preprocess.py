from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zeroqkernel.core.utils import ensure_dir, load_yaml, save_json
from zeroqkernel.datasets.loaders import load_dataset
from zeroqkernel.datasets.preprocess import preprocess_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess CICIDS-like CSV data")
    parser.add_argument(
        "--config",
        default="configs/dataset/cicids2017.yaml",
        help="Dataset config YAML path",
    )
    parser.add_argument(
        "--source-path",
        default=None,
        help="Optional override for source.path in config",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional debug cap on loaded rows across all files",
    )
    parser.add_argument(
        "--output",
        default="data/processed/cicids2017_preprocessed.npz",
        help="Output NPZ path",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional summary JSON path (default: <output>.summary.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    source_cfg = cfg.setdefault("source", {})
    if args.source_path:
        source_cfg["path"] = args.source_path
    if args.max_rows is not None:
        source_cfg["max_rows"] = int(args.max_rows)

    raw_data = load_dataset(cfg)
    processed = preprocess_dataset(raw_data, cfg)

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    np.savez_compressed(
        output_path,
        x=processed.x,
        y=processed.y,
        raw_labels=processed.raw_labels.astype("U"),
        feature_names=np.asarray(processed.feature_names, dtype="U"),
    )

    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    save_json(summary_path, processed.stats)

    print(f"Saved preprocessed dataset: {output_path}")
    print(f"Saved preprocessing summary: {summary_path}")
    print(
        "Shapes:"
        f" x={processed.x.shape}, y={processed.y.shape}, features={len(processed.feature_names)}"
    )
    print(
        "Label counts:"
        f" normal={processed.stats['normal_count']}, anomaly={processed.stats['anomaly_count']}"
    )


if __name__ == "__main__":
    main()
