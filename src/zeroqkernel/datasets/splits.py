"""Zero-day split strategies.

This module defines the core experimental assumptions for zero-day settings.
"""

from __future__ import annotations

from typing import Any


class SplitStrategy:
    """Base split strategy interface."""

    def split(self, data, labels):
        raise NotImplementedError


class AttackFamilyHoldoutSplit(SplitStrategy):
    """Hold out full attack families as unseen zero-day events."""

    def __init__(self, holdout_families: list[str]) -> None:
        self.holdout_families = holdout_families

    def split(self, data, labels):
        _ = data
        _ = labels
        raise NotImplementedError("Implement attack-family holdout split")


class ConditionHoldoutSplit(SplitStrategy):
    """Hold out conditions like protocol/service/time for realistic zero-day."""

    def __init__(self, condition_key: str, holdout_values: list[str]) -> None:
        self.condition_key = condition_key
        self.holdout_values = holdout_values

    def split(self, data, labels):
        _ = data
        _ = labels
        raise NotImplementedError("Implement condition holdout split")


class GeneratedHoldoutSplit(SplitStrategy):
    """Use synthetic traffic mutation rules to produce unknown transformations."""

    def __init__(self, generator_name: str, generator_params: dict[str, Any] | None = None) -> None:
        self.generator_name = generator_name
        self.generator_params = generator_params or {}

    def split(self, data, labels):
        _ = data
        _ = labels
        raise NotImplementedError("Implement generated holdout split")


def build_split_strategy(config: dict[str, Any]) -> SplitStrategy:
    strategy = config.get("strategy")
    if strategy == "attack_family_holdout":
        return AttackFamilyHoldoutSplit(config.get("holdout_families", []))
    if strategy == "condition_holdout":
        return ConditionHoldoutSplit(
            condition_key=config.get("condition_key", "service"),
            holdout_values=config.get("holdout_values", []),
        )
    if strategy == "generated_holdout":
        return GeneratedHoldoutSplit(
            generator_name=config.get("generator_name", "traffic_mutation"),
            generator_params=config.get("generator_params", {}),
        )
    raise ValueError(f"Unknown split strategy: {strategy}")
