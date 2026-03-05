"""단일 실험 실행 진입점."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from zeroqkernel.core.logging import ExperimentLogger
from zeroqkernel.core.seed import set_seed
from zeroqkernel.core.utils import ensure_dir, load_yaml, make_run_id, save_json
from zeroqkernel.datasets.splits import (
    AttackFamilyHoldoutSplit,
    ConditionHoldoutSplit,
    GeneratedHoldoutSplit,
    SplitStrategy,
    build_split_strategy,
)
from zeroqkernel.metrics.detection import compute_detection_metrics
from zeroqkernel.models.classical.ocsvm import OCSVMBaseline
from zeroqkernel.models.classical.rbf import rbf_kernel
from zeroqkernel.scoring.thresholds import apply_threshold, calibrate_quantile


@dataclass(slots=True)
class PreparedDataset:
    x: np.ndarray
    y: np.ndarray
    raw_labels: np.ndarray
    feature_names: list[str]


@dataclass(slots=True)
class SplitBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    stats: dict[str, Any]


def _normalize_token(value: str) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def _load_preprocessed_dataset(
    experiment_config: dict[str, Any],
) -> tuple[PreparedDataset, dict[str, Any]]:
    dataset_ref = experiment_config.get("dataset", {})
    dataset_config_path = str(dataset_ref.get("config_path", "configs/dataset/cicids2017.yaml"))
    dataset_config = load_yaml(dataset_config_path)

    default_npz = Path("data/processed") / f"{dataset_config.get('name', 'dataset')}_preprocessed.npz"
    preprocessed_path = Path(dataset_ref.get("preprocessed_path", default_npz))
    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found: {preprocessed_path}. "
            "Run scripts/preprocess.py first."
        )

    with np.load(preprocessed_path, allow_pickle=False) as npz:
        required = {"x", "y", "raw_labels", "feature_names"}
        missing = required.difference(npz.files)
        if missing:
            raise KeyError(f"Missing arrays in {preprocessed_path}: {sorted(missing)}")

        dataset = PreparedDataset(
            x=np.asarray(npz["x"]),
            y=np.asarray(npz["y"], dtype=np.int8).reshape(-1),
            raw_labels=np.asarray(npz["raw_labels"]).astype("U").reshape(-1),
            feature_names=[
                str(name) for name in np.asarray(npz["feature_names"]).astype("U").reshape(-1).tolist()
            ],
        )

    if dataset.x.shape[0] != dataset.y.shape[0] or dataset.y.shape[0] != dataset.raw_labels.shape[0]:
        raise ValueError("Preprocessed arrays have inconsistent row counts")

    resolved = {
        "name": str(dataset_config.get("name", "dataset")),
        "config_path": dataset_config_path,
        "preprocessed_path": str(preprocessed_path),
        "rows": int(dataset.x.shape[0]),
        "features": int(dataset.x.shape[1]),
    }
    return dataset, resolved


def _load_model_config(experiment_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model_ref = experiment_config.get("model", {})

    if "type" in model_ref:
        model_config = dict(model_ref)
        resolved_path = None
    else:
        resolved_path = str(model_ref.get("config_path", "configs/model/rbf_ocsvm.yaml"))
        model_config = load_yaml(resolved_path)

    resolved = {
        "config_path": resolved_path,
        "type": str(model_config.get("type", "unknown")),
    }
    return model_config, resolved


def _match_holdout_tokens(raw_labels: np.ndarray, tokens: list[str]) -> np.ndarray:
    token_keys: list[str] = []
    for token in tokens:
        normalized = _normalize_token(token)
        if normalized:
            token_keys.append(normalized)
    if not token_keys:
        raise ValueError("Holdout labels must not be empty")

    return np.fromiter(
        (
            any(
                token == label_key or token in label_key or label_key in token
                for token in token_keys
            )
            for label_key in (_normalize_token(label) for label in raw_labels)
        ),
        dtype=bool,
        count=int(raw_labels.shape[0]),
    )


def _build_holdout_mask(
    strategy: SplitStrategy,
    y: np.ndarray,
    raw_labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    anomaly_mask = np.asarray(y, dtype=np.int8) == 1

    if isinstance(strategy, AttackFamilyHoldoutSplit):
        holdout_mask = anomaly_mask & _match_holdout_tokens(raw_labels, strategy.holdout_families)
        if not np.any(holdout_mask):
            raise ValueError(
                f"No anomalies matched holdout_families={strategy.holdout_families!r}. "
                "Check the raw label names in the preprocessed dataset."
            )
        return holdout_mask, {"holdout_families": list(strategy.holdout_families)}

    if isinstance(strategy, ConditionHoldoutSplit):
        holdout_mask = anomaly_mask & _match_holdout_tokens(raw_labels, strategy.holdout_values)
        if not np.any(holdout_mask):
            raise ValueError(
                f"No anomalies matched holdout_values={strategy.holdout_values!r} "
                f"for condition_key={strategy.condition_key!r}."
            )
        return holdout_mask, {
            "condition_key": strategy.condition_key,
            "holdout_values": list(strategy.holdout_values),
            "match_mode": "raw_label_substring",
        }

    if isinstance(strategy, GeneratedHoldoutSplit):
        anomaly_idx = np.flatnonzero(anomaly_mask)
        if anomaly_idx.size == 0:
            raise ValueError("No anomaly rows available for generated holdout")

        generator_params = dict(strategy.generator_params)
        holdout_fraction = float(generator_params.get("holdout_fraction", 0.25))
        if not 0.0 < holdout_fraction <= 1.0:
            raise ValueError("generator_params.holdout_fraction must be in (0, 1]")

        rng = np.random.default_rng(seed)
        holdout_size = max(1, int(round(anomaly_idx.size * holdout_fraction)))
        selected = rng.choice(anomaly_idx, size=min(holdout_size, anomaly_idx.size), replace=False)
        holdout_mask = np.zeros_like(anomaly_mask, dtype=bool)
        holdout_mask[selected] = True
        return holdout_mask, {
            "generator_name": strategy.generator_name,
            "generator_params": generator_params,
        }

    raise ValueError(f"Unsupported split strategy type: {type(strategy).__name__}")


def _read_positive_int(config: dict[str, Any], key: str, default: int) -> int:
    value = int(config.get(key, default))
    if value <= 0:
        raise ValueError(f"split.{key} must be positive")
    return value


def _build_split_bundle(
    dataset: PreparedDataset,
    strategy: SplitStrategy,
    split_config: dict[str, Any],
    seed: int,
) -> SplitBundle:
    train_size = _read_positive_int(split_config, "train_size", 2048)
    val_size = _read_positive_int(split_config, "val_size", 1024)
    test_normal_size = _read_positive_int(split_config, "test_normal_size", 2048)
    test_anomaly_size = _read_positive_int(split_config, "test_anomaly_size", 2048)

    holdout_mask, holdout_info = _build_holdout_mask(strategy, dataset.y, dataset.raw_labels, seed=seed)
    normal_idx = np.flatnonzero(dataset.y == 0)
    if normal_idx.size == 0:
        raise ValueError("No normal rows available for one-class training")

    rng = np.random.default_rng(seed)
    normal_perm = normal_idx.copy()
    rng.shuffle(normal_perm)

    train_n = min(train_size, normal_perm.size)
    val_n = min(val_size, max(0, normal_perm.size - train_n))
    test_normal_n = min(test_normal_size, max(0, normal_perm.size - train_n - val_n))
    if train_n == 0 or val_n == 0 or test_normal_n == 0:
        raise ValueError("Not enough normal rows to populate train/val/test splits")

    train_idx = normal_perm[:train_n]
    val_idx = normal_perm[train_n : train_n + val_n]
    test_normal_idx = normal_perm[train_n + val_n : train_n + val_n + test_normal_n]

    holdout_anomaly_idx = np.flatnonzero(holdout_mask)
    test_anomaly_n = min(test_anomaly_size, holdout_anomaly_idx.size)
    if test_anomaly_n == 0:
        raise ValueError("No held-out anomaly rows available for testing")
    test_anomaly_idx = rng.choice(holdout_anomaly_idx, size=test_anomaly_n, replace=False)

    test_idx = np.concatenate([test_normal_idx, test_anomaly_idx])
    rng.shuffle(test_idx)

    stats = {
        "strategy": strategy.__class__.__name__,
        "train_size": int(train_idx.size),
        "val_size": int(val_idx.size),
        "test_normal_size": int(test_normal_idx.size),
        "test_anomaly_size": int(test_anomaly_idx.size),
        "available_normals": int(normal_idx.size),
        "available_holdout_anomalies": int(holdout_anomaly_idx.size),
        "available_seen_anomalies": int(np.sum((dataset.y == 1) & ~holdout_mask)),
    }
    stats.update(holdout_info)

    return SplitBundle(
        x_train=dataset.x[train_idx],
        y_train=dataset.y[train_idx],
        x_val=dataset.x[val_idx],
        y_val=dataset.y[val_idx],
        x_test=dataset.x[test_idx],
        y_test=dataset.y[test_idx],
        stats=stats,
    )


def _fit_one_class_rbf_ocsvm(
    split: SplitBundle,
    model_config: dict[str, Any],
    threshold_fpr: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    rbf_config = model_config.get("rbf", {})
    ocsvm_config = model_config.get("ocsvm", {})
    gamma = float(rbf_config.get("gamma", 0.25))
    nu = float(ocsvm_config.get("nu", 0.05))

    model = OCSVMBaseline(nu=nu)

    fit_start = time.perf_counter()
    k_train = rbf_kernel(split.x_train, split.x_train, gamma=gamma)
    model.fit(k_train)
    k_val = rbf_kernel(split.x_val, split.x_train, gamma=gamma)
    val_scores = model.score_samples(k_val)
    threshold = calibrate_quantile(val_scores, false_positive_rate=threshold_fpr)
    fit_seconds = time.perf_counter() - fit_start

    inference_start = time.perf_counter()
    k_test = rbf_kernel(split.x_test, split.x_train, gamma=gamma)
    test_scores = model.score_samples(k_test)
    inference_seconds = time.perf_counter() - inference_start

    details = {
        "model_type": str(model_config.get("type", "classical_rbf_ocsvm")),
        "gamma": gamma,
        "nu": nu,
        "fit_seconds": float(fit_seconds),
        "inference_seconds": float(inference_seconds),
    }
    return test_scores, float(threshold), details


def _append_summary_row(summary_path: Path, row: dict[str, Any]) -> None:
    ensure_dir(summary_path.parent)
    fieldnames = [
        "run_id",
        "experiment",
        "model",
        "auroc",
        "auprc",
        "tpr_at_fpr_1",
        "eer",
        "latency_sec",
    ]
    write_header = not summary_path.exists() or summary_path.stat().st_size == 0

    with summary_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_experiment(config_path: str) -> Path:
    config = load_yaml(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    training_config = config.get("training", {})
    training_mode = str(training_config.get("mode", "one_class")).lower()
    if training_mode != "one_class":
        raise NotImplementedError(f"Unsupported training.mode: {training_mode}")

    evaluation_config = config.get("evaluation", {})
    threshold_fpr = float(evaluation_config.get("threshold_fpr", 0.01))
    if not 0.0 < threshold_fpr < 1.0:
        raise ValueError("evaluation.threshold_fpr must be in (0, 1)")

    run_id = make_run_id(prefix=str(config.get("name", "run")))
    run_dir = ensure_dir(Path("results/runs") / run_id)
    logger = ExperimentLogger(run_dir)

    dataset, dataset_resolved = _load_preprocessed_dataset(config)
    model_config, model_resolved = _load_model_config(config)
    split_strategy = build_split_strategy(config.get("split", {}))
    split = _build_split_bundle(dataset, split_strategy, config.get("split", {}), seed=seed)

    model_type = str(model_config.get("type", ""))
    if model_type != "classical_rbf_ocsvm":
        raise NotImplementedError(
            "Only model.type=classical_rbf_ocsvm is wired into the runnable pipeline. "
            f"Received {model_type!r}."
        )

    test_scores, threshold, model_details = _fit_one_class_rbf_ocsvm(
        split,
        model_config,
        threshold_fpr=threshold_fpr,
    )

    predictions = apply_threshold(test_scores, threshold)
    detection = compute_detection_metrics(split.y_test, test_scores)

    tp = int(np.sum((predictions == 1) & (split.y_test == 1)))
    tn = int(np.sum((predictions == 0) & (split.y_test == 0)))
    fp = int(np.sum((predictions == 1) & (split.y_test == 0)))
    fn = int(np.sum((predictions == 0) & (split.y_test == 1)))

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    false_positive_rate = float(fp / (fp + tn)) if (fp + tn) else 0.0

    resolved_config = dict(config)
    resolved_config["dataset_resolved"] = dataset_resolved
    resolved_config["model_resolved"] = model_resolved

    metrics = {
        "status": "completed",
        "dataset": dataset_resolved["name"],
        "model": model_details["model_type"],
        "threshold_fpr_target": threshold_fpr,
        "threshold": float(threshold),
        "train_size": int(split.x_train.shape[0]),
        "val_size": int(split.x_val.shape[0]),
        "test_size": int(split.x_test.shape[0]),
        "test_normal_count": int(np.sum(split.y_test == 0)),
        "test_anomaly_count": int(np.sum(split.y_test == 1)),
        "predicted_anomalies": int(np.sum(predictions)),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "auroc": detection.auroc,
        "auprc": detection.auprc,
        "tpr_at_fpr_1": detection.tpr_at_fpr_1,
        "eer": detection.eer,
        "gamma": model_details["gamma"],
        "nu": model_details["nu"],
        "fit_seconds": model_details["fit_seconds"],
        "inference_seconds": model_details["inference_seconds"],
    }

    logger.log_config(resolved_config)
    logger.log_metrics(metrics)
    logger.log_note(
        "Completed: preprocessed NPZ load -> zero-day split -> classical RBF OCSVM -> metrics."
    )

    save_json(run_dir / "split.json", split.stats)
    np.savez_compressed(
        run_dir / "scores.npz",
        y_true=split.y_test.astype(np.int8, copy=False),
        scores=np.asarray(test_scores, dtype=np.float32),
        predictions=predictions.astype(np.int8, copy=False),
        threshold=np.asarray([threshold], dtype=np.float32),
    )

    _append_summary_row(
        Path("results/summary.csv"),
        {
            "run_id": run_id,
            "experiment": str(config.get("name", "run")),
            "model": model_details["model_type"],
            "auroc": detection.auroc,
            "auprc": detection.auprc,
            "tpr_at_fpr_1": detection.tpr_at_fpr_1,
            "eer": detection.eer,
            "latency_sec": model_details["inference_seconds"],
        },
    )

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one zeroqkernel experiment")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_experiment(args.config)
    print(f"Run completed at: {run_dir}")


if __name__ == "__main__":
    main()
