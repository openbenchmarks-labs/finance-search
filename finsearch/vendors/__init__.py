"""Search-API registry.

To benchmark an API that isn't here, subclass `Adapter` (see base.py and adapters.py for examples) and either
add it to `board_adapters()` or put it in your own module and pass `--vendor-module your.module`: any module
that calls `register(...)` on import works.
"""

from __future__ import annotations

import importlib

from .adapters import board_adapters
from .base import Adapter, HttpRequest, VendorCall, hit, dedupe  # noqa: F401  (public interface)

REGISTRY: dict[str, Adapter] = {}


def register(adapter: Adapter) -> Adapter:
    if not adapter.key:
        raise ValueError("adapter needs a key")
    REGISTRY[adapter.key] = adapter
    return adapter


for _a in board_adapters():
    register(_a)

BOARD_VENDORS = tuple(REGISTRY)


def load_module(path: str) -> None:
    """Import a module that registers extra adapters."""
    importlib.import_module(path)


def get(key: str) -> Adapter:
    if key not in REGISTRY:
        raise KeyError(f"unknown vendor {key!r}; known: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[key]


def search(key: str, query: str, *, max_results: int = 10) -> VendorCall:
    return get(key).search(query, max_results=max_results)


def fetch(key: str, url: str, *, objective: str, max_chars: int) -> VendorCall:
    return get(key).fetch(url, objective=objective, max_chars=max_chars)


def required_env(keys: list[str]) -> list[str]:
    return sorted({k for v in keys for k in get(v).env_keys})
