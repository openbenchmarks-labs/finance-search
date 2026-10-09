"""Prompt text, kept in files so a reviewer can read exactly what the models see."""

from __future__ import annotations

from functools import cache
from pathlib import Path

_DIR = Path(__file__).resolve().parent


@cache
def load(name: str) -> str:
    """The prompt template in prompts/<name>.txt, without the trailing newline."""
    return (_DIR / f"{name}.txt").read_text().rstrip("\n")
