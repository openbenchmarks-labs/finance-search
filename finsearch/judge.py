"""The grader: Claude Opus 5.5 compares the agent's answer with the reference answer and tolerance."""

from __future__ import annotations

import json
import time
from typing import Any

from . import config
from .prompts import load


def parse_judge_payload(text: str) -> dict[str, Any]:
    """The first JSON object that carries a correct/incorrect verdict.

    A reply can mention other braces before that object. Taking the first `{`
    through the last `}` makes a real verdict unreadable.
    """
    decoder = json.JSONDecoder()
    body = text or ""
    index = 0
    while True:
        start = body.find("{", index)
        if start < 0:
            return {}
        try:
            payload, end = decoder.raw_decode(body, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(payload, dict) and "correct" in payload:
            return payload
        index = end


def judge(client: Any, q: dict[str, Any], candidate: str) -> dict[str, Any]:
    prompt = load("judge").format(
        question=q["question"], answer=q["answer"], tolerance=q.get("tolerance", ""),
        candidate=candidate or "(no answer)",
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
    verdict = parse_judge_payload(text)
    if "correct" not in verdict:
        return base | {"correct": False, "reason": f"unparseable judge output: {text[:200]}", "judge_error": "parse"}
    return base | {"correct": bool(verdict["correct"]), "reason": str(verdict.get("reason", ""))}
