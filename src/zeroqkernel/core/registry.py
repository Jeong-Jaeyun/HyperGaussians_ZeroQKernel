"""모델, 데이터셋, 지표용 간단한 레지스트리 헬퍼."""

from __future__ import annotations

from collections.abc import Callable


class Registry:
    """가벼운 플러그인식 연결을 위한 이름-호출가능 객체 레지스트리."""

    def __init__(self) -> None:
        self._items: dict[str, Callable] = {}

    def register(self, name: str, obj: Callable) -> None:
        if name in self._items:
            raise KeyError(f"{name!r} is already registered")
        self._items[name] = obj

    def get(self, name: str) -> Callable:
        if name not in self._items:
            available = ", ".join(sorted(self._items.keys()))
            raise KeyError(f"Unknown registry key {name!r}. Available: {available}")
        return self._items[name]

    def build(self, name: str, **kwargs):
        return self.get(name)(**kwargs)
