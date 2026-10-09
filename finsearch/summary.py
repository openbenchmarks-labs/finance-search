"""Board numbers from a run's results.jsonl.

Per (task, vendor): accuracy is the mean over repeats of each repeat's accuracy, sd the sample standard
deviation across repeats, ci the 95% normal interval over all graded runs, flip the share of questions whose
verdict differs between repeats. Rows whose judge call failed are left out (re-grade them with `rejudge`),
not counted wrong.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def compute_summary(rows: list[dict]) -> dict:
    acc: dict = defaultdict(lambda: defaultdict(list))
    per_q: dict = defaultdict(list)
    graded = [r for r in rows if not r.get("judge_error")]
    for r in graded:
        for task in (r["task"], "all"):
            acc[(task, r["vendor"])][r.get("repeat", 0)].append(r["correct"])
        per_q[(r["task"], r["vendor"], r["question_id"])].append(r["correct"])
    cells = []
    for (task, vendor), reps in sorted(acc.items()):
        per_rep = [sum(v) / len(v) for v in reps.values()]
        m, sd = _mean_sd(per_rep)
        n_runs = sum(len(v) for v in reps.values())
        ci = 1.96 * math.sqrt(m * (1 - m) / n_runs) if n_runs else 0.0
        qs = [v for k, v in per_q.items() if k[1] == vendor and (task == "all" or k[0] == task)]
        multi = [v for v in qs if len(v) > 1]
        flip = sum(1 for v in multi if len(set(v)) > 1) / max(1, len(multi))
        cells.append({"task": task, "vendor": vendor, "questions": max(len(v) for v in reps.values()),
                      "repeats": len(reps), "accuracy": m, "sd": sd, "ci": ci, "flip": flip, "n_runs": n_runs})
    cost: dict = defaultdict(float)
    for r in rows:
        cost[(r["vendor"], "llm")] += r.get("llm_cost_usd", 0)
        cost[(r["vendor"], "judge")] += r.get("judge_cost_usd", 0)
        cost[(r["vendor"], "vendor_api")] += r.get("vendor_cost_usd", 0) or 0
    return {"cells": cells, "cost": dict(cost), "n_rows": len(rows), "n_graded": len(graded),
            "judge_errors": sum(1 for r in rows if r.get("judge_error"))}


def read_rows(out_dir: Path) -> list[dict]:
    path = out_dir / "results.jsonl"
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_summary(out_dir: Path) -> Path:
    s = compute_summary(read_rows(out_dir))
    lines = ["# Finance search run", "", f"Run directory: `{out_dir}`", "",
             "Accuracy is the mean over repeats; SD is across repeats; the 95% CI uses all graded runs. "
             "Flip rate is the share of questions with different verdicts across repeats.", "",
             "| Task | Vendor | Questions | Repeats | Accuracy | SD | 95% CI | Flip rate |",
             "|---|---|---|---|---|---|---|---|"]
    for c in s["cells"]:
        lines.append(f"| {c['task']} | {c['vendor']} | {c['questions']} | {c['repeats']} | {c['accuracy']:.1%} | "
                     f"{c['sd'] * 100:.1f} pts | ±{c['ci'] * 100:.1f} pts | {c['flip']:.0%} |")
    lines += ["", "## Cost (USD)", "", "| Vendor | LLM | Judge | Search API | Total |", "|---|---|---|---|---|"]
    for v in sorted({k[0] for k in s["cost"]}):
        parts = [s["cost"].get((v, k), 0.0) for k in ("llm", "judge", "vendor_api")]
        lines.append(f"| {v} | ${parts[0]:.2f} | ${parts[1]:.2f} | ${parts[2]:.2f} | ${sum(parts):.2f} |")
    lines += ["", f"Runs: {s['n_rows']}; graded: {s['n_graded']}; judge errors (left out of accuracy, "
                  f"re-grade with `finsearch rejudge`): {s['judge_errors']}"]
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(
        {**s, "cost": [{"vendor": k[0], "kind": k[1], "usd": v} for k, v in s["cost"].items()]}, indent=1))
    return path
