"""제로데이 설정을 위한 데이터 분할 전략 모음."""

from __future__ import annotations

from typing import Any


class SplitStrategy:
    """분할 전략 기본 인터페이스."""

    def split(self, data, labels):
        raise NotImplementedError


class AttackFamilyHoldoutSplit(SplitStrategy):
    """공격 패밀리 전체를 미관측 제로데이 이벤트로 홀드아웃한다."""

    def __init__(self, holdout_families: list[str]) -> None:
        self.holdout_families = holdout_families

    def split(self, data, labels):
        _ = data
        _ = labels
        raise NotImplementedError("Implement attack-family holdout split")


class ConditionHoldoutSplit(SplitStrategy):
    """프로토콜/서비스/시간 같은 조건 기반 제로데이 홀드아웃을 정의한다."""

    def __init__(self, condition_key: str, holdout_values: list[str]) -> None:
        self.condition_key = condition_key
        self.holdout_values = holdout_values

    def split(self, data, labels):
        _ = data
        _ = labels
        raise NotImplementedError("Implement condition holdout split")


class GeneratedHoldoutSplit(SplitStrategy):
    """합성 트래픽 변형 규칙으로 미지 변환 홀드아웃을 생성한다."""

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
