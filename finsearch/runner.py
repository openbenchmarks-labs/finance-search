"""Running the benchmark: every (question, vendor, repeat) once, graded as it finishes.

Each row of runs/<stamp>/results.jsonl holds the question, every search call and its results, any fetched
pages, the answer, the verdict, tokens and costs. A run can be resumed: finished rows are skipped.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import agent as agent_mod
from . import config, vendors
from .agent import run_agent
from .judge import judge
from .llm import agent_client, judge_client
from .summary import read_rows, write_summary


def check_env(vendor_keys: list[str]) -> None:
    missing = [k for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", *vendors.required_env(vendor_keys)]
               if not os.environ.get(k, "").strip()]
    if missing:
        raise SystemExit(f"missing environment variables: {', '.join(missing)} (see .env.example)")


def run(questions: list[dict], vendor_keys: list[str], *, out_dir: Path, repeats: int, mode: str, today: str,
        data_label: str, single_concurrency: int = 64, multi_concurrency: int = 32) -> Path:
    setup = config.class_setup(mode)
    manifest_path = out_dir / "manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = {(r["question_id"], r["vendor"], r.get("repeat", 0)) for r in read_rows(out_dir)}

    # In search_fetch mode a vendor without a native fetch can't do multi-hop on equal terms: single only.
    no_fetch = {v for v in vendor_keys if not vendors.get(v).native_fetch} if mode == "search_fetch" else set()
    jobs = [(q, v, rep) for rep in range(repeats) for q in questions for v in vendor_keys
            if (q["id"], v, rep) not in done and not (q["search"] == "multi" and v in no_fetch)]
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps({
            "data": data_label, "vendors": vendor_keys, "repeats": repeats, "today_in_prompt": today, "mode": mode,
            "class_setup": setup, "judge": config.JUDGE_MODEL, "judge_effort": config.JUDGE_EFFORT,
            "prices_usd_per_m": config.PRICES, "question_ids": [q["id"] for q in questions],
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2))
    if no_fetch:
        print(f"historical lookup only (no native fetch): {', '.join(sorted(no_fetch))}")
    print(f"mode={mode}: " + "; ".join(f"{c} {s['searches']} search(es) + {s['fetches']} fetch(es) on {s['model']} "
                                         f"({s['effort']})" for c, s in setup.items()))
    print(f"{len(questions)} questions x {len(vendor_keys)} vendors x {repeats} repeats -> {len(jobs)} runs to do "
          f"({len(done)} done) -> {out_dir}", flush=True)

    model_client, grader = agent_client(), judge_client()
    lock = threading.Lock()

    def work(job):
        q, vendor, rep = job
        s = setup[q["search"]]
        result = run_agent(q["id"], q["question"], client=model_client, vendor=vendor, config=s["config"],
                           max_searches=s["searches"], max_fetches=s["fetches"], model=s["model"],
                           today=today, effort=s["effort"])
        row = result.to_dict()
        row.update({"repeat": rep, "reasoning_effort": s["effort"], "search_class": q["search"], "task": q["task"],
                    "area": q.get("area", ""), "question": q["question"], "reference_answer": q["answer"],
                    "tolerance": q.get("tolerance", "")})
        try:
            row.update(judge(grader, q, result.answer))
        except Exception as exc:  # noqa: BLE001 - keep the paid agent run; re-grade later with `rejudge`
            row.update({"correct": False, "reason": "", "judge_error": f"{type(exc).__name__}: {exc}"[:500]})
        row["llm_cost_usd"] = round(config.llm_cost(s["price_model"], row["input_tokens"],
                                                    row.get("cached_input_tokens", 0), row["output_tokens"]), 6)
        row["judge_cost_usd"] = round(config.llm_cost(config.JUDGE_MODEL, row.get("judge_input_tokens") or 0, 0,
                                                      row.get("judge_output_tokens") or 0), 6)
        return row

    errors = judge_fails = 0
    spent = 0.0
    # One pool per class, so slow multi-hop jobs never hold threads the single-search jobs could use.
    pools = {"single": ThreadPoolExecutor(max_workers=single_concurrency),
             "multi": ThreadPoolExecutor(max_workers=multi_concurrency)}
    with pools["single"], pools["multi"]:
        futures = {pools[job[0]["search"]].submit(work, job): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            q, vendor, rep = futures[fut]
            try:
                row = fut.result()
            except Exception:  # noqa: BLE001 - the job is retried on resume
                errors += 1
                print(f"  [{i}/{len(jobs)}] {q['id']} {vendor} r{rep} ERROR\n{traceback.format_exc(limit=3)}",
                      file=sys.stderr)
                continue
            with lock, open(results_path, "a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            spent += row["llm_cost_usd"] + row["judge_cost_usd"] + (row.get("vendor_cost_usd") or 0)
            judge_fails += bool(row.get("judge_error"))
            verdict = "JUDGE FAILED" if row.get("judge_error") else ("correct" if row["correct"] else "wrong")
            print(f"  [{i}/{len(jobs)}] {q['id']} {vendor} r{rep}: {verdict} ({row['n_searches']}s/{row['n_fetches']}f, "
                  f"{row['wall_ms'] / 1000:.0f}s) | errors {errors}, judge fails {judge_fails}, "
                  f"model retries {sum(agent_mod.RETRY_COUNTS.values())}, spent ${spent:.2f}", flush=True)

    path = write_summary(out_dir)
    print(f"summary: {path}")
    if errors:
        print(f"{errors} runs failed; run again with --resume {out_dir} to retry them", file=sys.stderr)
    if judge_fails:
        print(f"{judge_fails} judge calls failed; re-grade with `finsearch rejudge {out_dir}`", file=sys.stderr)
    return out_dir


def rejudge(out_dir: Path, questions: list[dict], ids: set[str] | None = None) -> None:
    """Re-grade rows whose judge call failed, or, with ids, every row of those questions against the current
    answer keys. Rows are rewritten in place after a backup."""
    qs = {q["id"]: q for q in questions}
    rows = read_rows(out_dir)
    backup = out_dir / f"results.before_rejudge_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    backup.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows))
    grader = judge_client()
    todo = [r for r in rows if r.get("judge_error") or (ids and r["question_id"] in ids)]
    missing = sorted({r["question_id"] for r in todo} - set(qs))
    if missing:
        raise SystemExit(f"questions not in the loaded dataset: {missing}")

    def regrade(r):
        q = qs[r["question_id"]]
        r.pop("judge_error", None)
        try:
            r.update(judge(grader, q, r["answer"]))
        except Exception as exc:  # noqa: BLE001 - stays marked; run rejudge again
            r.update({"correct": False, "reason": "", "judge_error": f"{type(exc).__name__}: {exc}"[:500]})
        r["reference_answer"], r["tolerance"] = q["answer"], q.get("tolerance", "")
        r["judge_cost_usd"] = round(config.llm_cost(config.JUDGE_MODEL, r.get("judge_input_tokens") or 0, 0,
                                                    r.get("judge_output_tokens") or 0), 6)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(regrade, todo))
    (out_dir / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows))
    print(f"re-graded {len(todo)} rows (backup: {backup.name})")
    print(f"summary: {write_summary(out_dir)}")
