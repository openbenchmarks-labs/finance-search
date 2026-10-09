"""Loading questions. The dataset is not in this repo: it is read from Hugging Face, or from local files.

--data accepts:
  hf:<owner>/<name>[@<revision>]   a Hugging Face dataset repo with historical-lookup.jsonl,
                                    multi-hop-lookup.jsonl and live-lookup.jsonl at its root (the default)
  <directory>                       a local folder with the same files
  <file>.jsonl                      one file; each row needs id, question, answer, tolerance and either
                                    `search` (single | multi) or `task` (historical-lookup | multi-hop-lookup)
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config

TASK_FILES = {task: f"{task}.jsonl" for task in config.CONFIGS}
LIVE_FILE = "live-lookup.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _hf_file(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    repo, _, revision = repo.partition("@")
    return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", revision=revision or None))


def _source(data: str) -> tuple[str, str]:
    data = data or f"hf:{config.DATASET_REPO}"
    if data.startswith("hf:"):
        return "hf", data[3:]
    return "local", data


def _normalise(row: dict, task: str | None, source: str) -> dict:
    q = dict(row)
    q.setdefault("question", q.get("query"))
    if task:
        q["search"] = config.CONFIGS[task]
    elif q.get("task") in config.CONFIGS:
        q["search"] = config.CONFIGS[q["task"]]
    if q.get("search") not in ("single", "multi"):
        raise ValueError(f"{source}: row {q.get('id')} has no search class (set `search` or `task`)")
    q["task"] = "historical-lookup" if q["search"] == "single" else "multi-hop-lookup"
    for k in ("id", "question", "answer"):
        if not q.get(k):
            raise ValueError(f"{source}: row {q.get('id')} is missing {k}")
    q.setdefault("tolerance", "")
    q.setdefault("area", "")
    return q


def load_questions(data: str = "", tasks: tuple[str, ...] = tuple(config.CONFIGS)) -> list[dict]:
    kind, where = _source(data)
    out: list[dict] = []
    if kind == "local" and Path(where).is_file():
        out = [_normalise(r, None, where) for r in _read_jsonl(Path(where))]
        out = [q for q in out if q["task"] in tasks]
    else:
        for task in tasks:
            path = _hf_file(where, TASK_FILES[task]) if kind == "hf" else Path(where) / TASK_FILES[task]
            out += [_normalise(r, task, str(path)) for r in _read_jsonl(path)]
    ids = [q["id"] for q in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids in the dataset")
    return out


def load_live_items(data: str = "") -> list[dict]:
    """Live lookup items: id, asset (equity | crypto), name, ticker, query_template."""
    kind, where = _source(data)
    if kind == "local" and Path(where).is_file():
        path = Path(where)
    else:
        path = _hf_file(where, LIVE_FILE) if kind == "hf" else Path(where) / LIVE_FILE
    rows = _read_jsonl(path)
    for r in rows:
        if r.get("asset") not in ("equity", "crypto"):
            raise ValueError(f"{path}: live item {r.get('id')} has asset {r.get('asset')!r}; expected equity or crypto")
    return rows
