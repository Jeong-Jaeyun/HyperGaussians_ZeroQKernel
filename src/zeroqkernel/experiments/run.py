"""Single-run experiment entrypoint scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from zeroqkernel.core.logging import ExperimentLogger
from zeroqkernel.core.seed import set_seed
from zeroqkernel.core.utils import ensure_dir, load_yaml, make_run_id


def run_experiment(config_path: str) -> Path:
    config = load_yaml(config_path)
    run_id = make_run_id(prefix=config.get("name", "run"))
    run_dir = ensure_dir(Path("results/runs") / run_id)

    set_seed(int(config.get("seed", 42)))
    logger = ExperimentLogger(run_dir)
    logger.log_config(config)

    # Pipeline placeholder: load data -> preprocess -> split -> fit -> score -> metrics.
    logger.log_note(
        "Scaffold run executed. Implement dataset/model wiring in src/zeroqkernel/experiments/run.py"
    )
    logger.log_metrics({"status": "scaffold_only"})
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one zeroqkernel experiment")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_experiment(args.config)
    print(f"Run scaffold completed at: {run_dir}")


if __name__ == "__main__":
    main()
