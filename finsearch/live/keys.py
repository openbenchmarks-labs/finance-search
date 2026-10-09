"""Live lookup answer keys. The grader reads them from a source; it never asks a search API.

Stocks, one row per (date, ticker): the official close (the primary listing exchange's closing auction) and
the prior close.
    {"date": "2026-10-07", "ticker": "EQT", "close": 52.31, "prior_close": 52.47}

Coins, one row per (ticker, minute): the low-to-high traded range in that UTC minute (minute = its start).
An optional "buffer" (a fraction, e.g. 0.001) widens the range for that coin; without it the grader's default
for the coin's size band applies.
    {"ticker": "XRP", "minute": "2026-10-07T17:29:00Z", "low": 1.3812, "high": 1.3851}

Sources that ship here:
    file    a JSONL file, a folder holding live-closes.jsonl / live-coins.jsonl, or hf:<owner>/<name>[@rev]
    mock    fixed prices, for tests and dry runs ("mock" or "mock:<close>,<prior>" / "mock:<price>")
Anything else plugs in with --close-source / --coin-source module:ClassName; the class takes the
--closes / --coin-key string and implements the matching protocol below.
"""

from __future__ import annotations

import importlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Protocol

CLOSES_FILE = "live-closes.jsonl"
COINS_FILE = "live-coins.jsonl"


class KeyNotAvailable(RuntimeError):
    """The source has nothing for that day or minute yet."""


class CloseSource(Protocol):
    def closes(self, tickers: list[str], day: date) -> dict[str, dict]:
        """{ticker: {"close": float | None, "prior_close": float | None}}; raise KeyNotAvailable if none."""
        ...


class CoinSource(Protocol):
    def minutes(self, tickers: list[str], start: float, end: float) -> dict[str, dict[int, tuple[float, float, float | None]]]:
        """{ticker: {minute start (epoch s): (low, high, buffer or None)}} for minutes in [start, end)."""
        ...


def _rows(spec: str, filename: str) -> list[dict]:
    if spec.startswith("hf:"):
        from huggingface_hub import hf_hub_download

        repo, _, rev = spec[3:].partition("@")
        path = Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", revision=rev or None))
    else:
        path = Path(spec)
        if path.is_dir():
            path = path / filename
    if not path.exists():
        raise FileNotFoundError(f"answer-key file not found: {path}")
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _epoch(minute: str) -> int:
    t = datetime.fromisoformat(minute.replace("Z", "+00:00"))
    return int((t if t.tzinfo else t.replace(tzinfo=timezone.utc)).timestamp())


class FileCloseSource:
    def __init__(self, spec: str):
        self.rows = {(r["date"], r["ticker"]): r for r in _rows(spec, CLOSES_FILE)}

    def closes(self, tickers, day):
        d = day.isoformat()
        out = {t: {"close": self.rows.get((d, t), {}).get("close"),
                   "prior_close": self.rows.get((d, t), {}).get("prior_close")} for t in tickers}
        if all(v["close"] is None for v in out.values()):
            raise KeyNotAvailable(f"no closes for {d} in the answer key yet")
        return out


class FileCoinSource:
    def __init__(self, spec: str):
        self.by_ticker: dict[str, dict[int, tuple[float, float, float | None]]] = {}
        for r in _rows(spec, COINS_FILE):
            self.by_ticker.setdefault(r["ticker"], {})[_epoch(r["minute"])] = (
                float(r["low"]), float(r["high"]), float(r["buffer"]) if r.get("buffer") is not None else None)

    def minutes(self, tickers, start, end):
        return {t: {m: v for m, v in self.by_ticker.get(t, {}).items() if start <= m < end} for t in tickers}


class MockCloseSource:
    """Every ticker closes at <close> (default 100) with prior close <prior> (default 99)."""

    def __init__(self, spec: str = ""):
        parts = [float(x) for x in spec.split(",") if x.strip()] if spec else []
        self.close, self.prior = (parts + [100.0, 99.0][len(parts):])[:2]

    def closes(self, tickers, day):
        return {t: {"close": self.close, "prior_close": self.prior} for t in tickers}


class MockCoinSource:
    """Every coin trades at exactly <price> (default 100) in every minute."""

    def __init__(self, spec: str = ""):
        self.price = float(spec) if spec else 100.0

    def minutes(self, tickers, start, end):
        first = int(start) - int(start) % 60
        return {t: {m: (self.price, self.price, None) for m in range(first, int(end), 60)} for t in tickers}


def _load(spec: str, class_path: str, file_cls, mock_cls, flag: str):
    if class_path:
        module, _, name = class_path.partition(":")
        return getattr(importlib.import_module(module), name)(spec)
    if spec == "mock" or spec.startswith("mock:"):
        return mock_cls(spec[5:])
    if not spec:
        return None
    return file_cls(spec)


def close_source(spec: str, class_path: str = "") -> CloseSource | None:
    return _load(spec, class_path, FileCloseSource, MockCloseSource, "--closes")


def coin_source(spec: str, class_path: str = "") -> CoinSource | None:
    return _load(spec, class_path, FileCoinSource, MockCoinSource, "--coin-key")
