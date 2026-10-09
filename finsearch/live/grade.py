"""Live lookup, step 2: read each vendor's answer out of its results, then check it against market data.

Extraction: GPT-5.6 Sol (medium effort) reads only that call's results and replies in a strict JSON schema
with the price it would report, the time the results attach to it, and the source. The value may be null,
so the model is never forced to invent a price.

Classes, checked in this order:
  stocks   close (within max(0.02%, $0.01) of the official close), wrong_currency, prior_close, other, none
  coins    target (inside the traded range of the minute before and the minute of the target, widened by a
           buffer by coin size), current (inside the range of the minute it was sent in instead),
           stale (inside an earlier minute's range, up to 4 hours before sending), other, none
Answer keys come from the sources in keys.py, never from a search API.
Accuracy is the share of answers in class close / target.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .. import config
from ..llm import agent_client
from ..prompts import load
from .keys import CloseSource, CoinSource, KeyNotAvailable

UTC = timezone.utc
BUFFER = {"large": 0.0010, "mid": 0.0025, "small": 0.0050}   # coin price buffer by market-cap band
DEFAULT_SIZE = "mid"
HISTORY = timedelta(hours=4)
EXTRACT_MODEL = config.CLASS_MODELS["multi"][0]


def _nullable(kind: str, description: str) -> dict:
    return {"type": [kind, "null"], "description": description}


def _schema(types: list[str]) -> dict:
    props = {
        "price": _nullable("number", "the price in the currency below"),
        "stated_time": _nullable("string", "the date/time the results attach to that value, verbatim"),
        "source_url": _nullable("string", "url of the result the value came from"),
        "price_type": {"type": "string", "enum": types},
        "note": _nullable("string", "one short sentence: why this value, or why none"),
        "currency": {"type": ["string", "null"], "description": "e.g. USD, USDT, EUR"},
    }
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


SCHEMAS = {
    "equity": _schema(["close", "last_trade", "after_hours", "live", "previous_close", "other", "unclear"]),
    "crypto": _schema(["at_target_time", "current", "daily_close", "other", "unclear"]),
}


def results_text(hits: list[dict], limit: int = 60000) -> str:
    text = "\n\n".join(f"[{i + 1}] {h.get('title', '')}\nURL: {h.get('url', '')}\n{h.get('snippet', '')}"
                       for i, h in enumerate(hits)) or "(no results)"
    return text[:limit]


def extract(llm, r: dict, cfg: dict, tries: int = 3) -> dict:
    if r["asset"] == "equity":
        prompt = load("live_extract_equity").format(query=r["query"], name=r["name"], ticker=r["ticker"],
                                                    when=cfg["when_day"], results=results_text(r["hits"]))
    else:
        prompt = load("live_extract_crypto").format(query=r["query"], name=r["name"], when=cfg["when_minute"],
                                                    results=results_text(r["hits"]))
    for _ in range(tries):
        try:
            resp = llm.responses.create(
                model=EXTRACT_MODEL, reasoning={"effort": "medium"}, store=False, max_output_tokens=4000,
                text={"format": {"type": "json_schema", "name": f"{r['asset']}_price", "schema": SCHEMAS[r["asset"]],
                                 "strict": True}},
                input=[{"role": "user", "content": prompt}])
            out = json.loads(resp.output_text)
            u = getattr(resp, "usage", None)
            out["_usage"] = {"input": getattr(u, "input_tokens", None), "output": getattr(u, "output_tokens", None),
                             "cached_input": getattr(getattr(u, "input_tokens_details", None), "cached_tokens", None)}
            return out
        except Exception:  # noqa: BLE001 - network errors, truncated output; counted as no answer after 3 tries
            time.sleep(3)
    return {}


def _price(ex: dict) -> float | None:
    try:
        return float(ex["price"]) if ex.get("price") is not None else None
    except (TypeError, ValueError):
        return None


def near(p: float | None, x: float | None) -> bool:
    return p is not None and x is not None and abs(p - x) <= max(x * 0.0002, 0.01)


def grade_equity(rows: list[dict], cfg: dict, source: CloseSource) -> list[dict]:
    day = date.fromisoformat(cfg["trade_date"])
    keys = source.closes(sorted({r["ticker"] for r in rows}), day)
    out = []
    for r in rows:
        g, ex = keys[r["ticker"]], r.get("extracted") or {}
        p = _price(ex)
        cur = (ex.get("currency") or "USD").upper()
        if p is None:
            cls = "none"
        elif near(p, g["close"]):
            cls = "close"
        elif cur not in {"USD", "US$", "$"}:
            cls = "wrong_currency"
        elif near(p, g["prior_close"]):
            cls = "prior_close"
        else:
            cls = "other"
        out.append({"price": p, "currency": cur, "close": g["close"], "prior_close": g["prior_close"],
                    "class": cls, "correct": cls == "close", "key_available": g["close"] is not None})
    return out


def grade_crypto(rows: list[dict], cfg: dict, source: CoinSource) -> list[dict]:
    target = int(datetime.fromisoformat(cfg["target"]).timestamp())
    last = max(r["t_sent"] for r in rows)
    bars = source.minutes(sorted({r["ticker"] for r in rows}), target - HISTORY.total_seconds(), last + 120)
    if not any(bars.values()):
        raise KeyNotAvailable("no coin minutes in the answer key for this run yet")
    out = []
    for r in rows:
        c, ex = r["ticker"], r.get("extracted") or {}
        b = bars.get(c) or {}
        size = r.get("size") if r.get("size") in BUFFER else DEFAULT_SIZE
        key_buf = next((v[2] for v in b.values() if v[2] is not None), None)
        buf = key_buf if key_buf is not None else BUFFER[size]
        p = _price(ex)
        tw = [b[m] for m in (target - 60, target) if m in b]
        lo, hi = (min(x[0] for x in tw), max(x[1] for x in tw)) if tw else (None, None)
        sent_minute = int(r["t_sent"]) - int(r["t_sent"]) % 60
        cur = b.get(sent_minute) or b.get(sent_minute - 60)
        cur = (cur[0], cur[1]) if cur else None

        def within(x, a, z):
            return x is not None and a is not None and a * (1 - buf) <= x <= z * (1 + buf)

        age = None
        if p is not None:
            for m in sorted((m for m in b if m <= r["t_sent"]), reverse=True):
                if within(p, b[m][0], b[m][1]):
                    age = max(0.0, r["t_sent"] - (m + 60))
                    break
        if p is None:
            cls = "none"
        elif within(p, lo, hi):
            cls = "target"
        elif cur and within(p, *cur):
            cls = "current"
        elif age is not None:
            cls = "stale"
        else:
            cls = "other"
        out.append({"price": p, "currency": (ex.get("currency") or "USD").upper(), "size": size, "buffer": buf,
                    "target_low": lo, "target_high": hi, "current": cur, "implied_age_s": age, "class": cls,
                    "correct": cls == "target", "key_available": lo is not None})
    return out


def grade(run_dir: Path, assets: tuple[str, ...] = ("equity", "crypto"), close_source: CloseSource | None = None,
          coin_source: CoinSource | None = None) -> Path:
    cfg = json.loads((run_dir / "config.json").read_text())
    rows = [json.loads(line) for line in open(run_dir / "raw.jsonl") if line.strip()]
    rows = [r for r in rows if r["asset"] in assets]
    if not rows:
        raise SystemExit(f"no {'/'.join(assets)} rows in {run_dir}")
    llm = agent_client()
    todo = [r for r in rows if "extracted" not in r]
    print(f"extracting {len(todo)} answers with {EXTRACT_MODEL} ...", flush=True)
    with ThreadPoolExecutor(32) as pool:
        for r, ex in zip(todo, pool.map(lambda r: extract(llm, r, cfg) if r["status"] == "ok" else {}, todo)):
            r["extracted"] = ex
    _save_extractions(run_dir, rows)

    graded: list[dict] = []
    for asset in assets:
        part = [r for r in rows if r["asset"] == asset]
        if not part:
            continue
        print(f"answer key for {len(part)} {asset} answers ...", flush=True)
        try:
            if asset == "equity":
                if close_source is None:
                    raise KeyNotAvailable("no answer key given (--closes)")
                res = grade_equity(part, cfg, close_source)
            else:
                if coin_source is None:
                    raise KeyNotAvailable("no answer key given (--coin-key)")
                res = grade_crypto(part, cfg, coin_source)
        except KeyNotAvailable as exc:
            print(f"{'stocks' if asset == 'equity' else 'coins'} not graded: {exc}", flush=True)
            continue
        for r, g in zip(part, res):
            ex = r.get("extracted") or {}
            graded.append({k: r.get(k) for k in ("id", "asset", "ticker", "name", "offset_s", "vendor", "query",
                                                 "t_sent", "latency_ms", "status", "error_class")}
                          | {"sent_after_target_s": round(r["t_sent"] - datetime.fromisoformat(
                              cfg["close"] if asset == "equity" else cfg["target"]).timestamp(), 2),
                             "stated_time": ex.get("stated_time"), "price_type": ex.get("price_type"),
                             "source_url": ex.get("source_url")} | g)
    if not graded:
        raise SystemExit("nothing graded yet")
    (run_dir / "graded.jsonl").write_text("".join(json.dumps(g, default=str) + "\n" for g in graded))
    path = write_live_summary(run_dir, graded)
    print(path.read_text())
    return path


def _save_extractions(run_dir: Path, rows: list[dict]) -> None:
    """Keep extractions next to the raw hits, so re-grading doesn't pay for them again."""
    by_key = {(r["id"], r["vendor"], r["offset_s"]): r for r in rows}
    all_rows = [json.loads(line) for line in open(run_dir / "raw.jsonl") if line.strip()]
    merged = [by_key.get((r["id"], r["vendor"], r["offset_s"]), r) for r in all_rows]
    tmp = run_dir / "raw.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in merged))
    tmp.replace(run_dir / "raw.jsonl")


def write_live_summary(run_dir: Path, graded: list[dict]) -> Path:
    lines = [f"# Live lookup run {run_dir.name}", ""]
    for asset in ("equity", "crypto"):
        g = [x for x in graded if x["asset"] == asset and x["key_available"]]
        if not g:
            continue
        hit_class = "close" if asset == "equity" else "target"
        offsets = sorted({x["offset_s"] for x in g})
        vendors = sorted({x["vendor"] for x in g})
        cell = defaultdict(list)
        for x in g:
            cell[(x["vendor"], x["offset_s"])].append(x["correct"])
            cell[(x["vendor"], "all")].append(x["correct"])
        lines += [f"## {'Stocks: official close' if asset == 'equity' else 'Coins: price at the target minute'}", "",
                  f"Share of answers in class `{hit_class}`, by delay after the "
                  f"{'close' if asset == 'equity' else 'target'}.", "",
                  "| Vendor | " + " | ".join(f"+{o}s" for o in offsets) + " | All |",
                  "|---|" + "---|" * (len(offsets) + 1)]
        for v in vendors:
            vals = [cell[(v, o)] for o in offsets] + [cell[(v, "all")]]
            lines.append(f"| {v} | " + " | ".join(f"{sum(c) / len(c):.0%}" if c else "-" for c in vals) + " |")
        classes = Counter((x["vendor"], x["class"]) for x in g)
        names = sorted({c for _, c in classes})
        lines += ["", "Answer classes (all delays)", "", "| Vendor | " + " | ".join(names) + " |",
                  "|---|" + "---|" * len(names)]
        for v in vendors:
            lines.append(f"| {v} | " + " | ".join(str(classes.get((v, c), 0)) for c in names) + " |")
        lat = {v: statistics.median([x["latency_ms"] for x in g if x["vendor"] == v]) for v in vendors}
        lines += ["", "Median latency (ms): " + ", ".join(f"{v} {lat[v]:.0f}" for v in vendors), ""]
    missing = sorted({x["ticker"] for x in graded if not x["key_available"]})
    if missing:
        lines += [f"No answer key for {', '.join(missing)} (left out above).", ""]
    path = run_dir / "summary.md"
    path.write_text("\n".join(lines))
    return path
