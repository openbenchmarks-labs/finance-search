"""The grader: Claude Opus 5.5 compares the agent's answer with the reference answer and tolerance."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from . import config
from .prompts import load


def judge(client: Any, q: dict[str, Any], candidate: str) -> dict[str, Any]:
    prompt = load("judge").format(
        question=q["question"], answer=q["answer"], tolerance=q.get("tolerance", ""),
        trap=q.get("trap") or "(none)", candidate=candidate or "(no answer)",
    )
    t0 = time.perf_counter()
    response = client.messages.create(
        model=config.JUDGE_MODEL,
        max_tokens=8000,
        output_config={"effort": config.JUDGE_EFFORT},
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    base = {"judge_model": response.model, "judge_ms": latency_ms,
            "judge_input_tokens": response.usage.input_tokens, "judge_output_tokens": response.usage.output_tokens}
    if response.stop_reason == "refusal":
        return base | {"correct": False, "reason": "judge refused", "judge_error": "refusal"}
    text = "".join(b.text for b in response.content if b.type == "text")
    match = re.search(r"\{.*\}", text, re.S)
    try:
        verdict = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        verdict = {}
    if "correct" not in verdict:
        return base | {"correct": False, "reason": f"unparseable judge output: {text[:200]}", "judge_error": "parse"}
    return base | {"correct": bool(verdict["correct"]), "reason": str(verdict.get("reason", ""))}
