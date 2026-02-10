"""Simple registry helpers for models, datasets, and metrics."""

from __future__ import annotations

from collections.abc import Callable


class Registry:
    """Name to callable registry for lightweight plugin-style wiring."""

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
