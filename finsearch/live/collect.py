"""Live lookup, step 1: ask every search API for each item's price at set delays after the target moment.

Stocks are asked for the official close of --date (16:00 New York time, 13:00 on half days). Coins are asked
for the price at the --target minute (UTC). At target + each delay every item is sent once to every vendor;
each row keeps its actual send time, so a late batch is graded by when it was really sent.

Calls are paced per API key, not per config: configs that share a key (the Exa types, the Parallel modes)
share one budget, which is how the vendors enforce their limits.

Output: runs/live/<stamp>/ with config.json and raw.jsonl. Grade it with `finsearch live-grade <run dir>`
once the answer keys cover the run (keys.py).
"""

from __future__ import annotations

import collections
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .. import vendors

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
DEFAULT_OFFSETS = (5, 30, 60, 900, 3600, 10800)

# NYSE / Nasdaq calendar. Other years: pass --close-time.
HOLIDAYS = {date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
            date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25)}
HALF_DAYS = {date(2026, 11, 27), date(2026, 12, 24)}          # close at 13:00 New York time
CALENDAR_YEARS = {2026}

# Optional rotating wordings (--rotate-wordings): one per delay, so no query string is sent twice in a run and a
# vendor's cache for an earlier delay can't answer a later one. The default is each item's query_template.
WORDINGS = {
    "equity": ["{ticker} closing price {date}", "{name} ({ticker}) stock close {date}",
               "{ticker} stock price at close {date}", "what did {ticker} close at on {date}",
               "{name} share price close {date}", "{ticker} close {date}", "{name} stock closing price {date}"],
    "crypto": ["{name} price {date} {time}", "{name} ({ticker}) price at {date} {time}",
               "{ticker} USD price {date} {time}", "{name} to USD {date} {time}",
               "what was the {name} price at {date} {time}", "{name} price in US dollars {date} {time}",
               "{ticker}/USD {date} {time}"],
}

# Pacing per API key, set from the limits the vendors enforced in our runs. Others: 10 calls in flight.
LIMITS = {
    "exa": dict(rps=8, concurrent=16),
    "parallel": dict(rps=6, per_minute=240, concurrent=12),
    "tako": dict(concurrent=4),
    "serp": dict(rps=5, concurrent=5),
    "perplexity": dict(rps=5, concurrent=8),
    "tinyfish": dict(per_minute=30, concurrent=2),
}


class Limit:
    """Requests per second, a rolling 60 s budget and calls in flight, for one API key."""

    def __init__(self, rps: float | None = None, per_minute: int | None = None, concurrent: int = 10):
        self.sem = threading.BoundedSemaphore(concurrent)
        self.gap = 1 / rps if rps else 0.0
        self.per_minute = per_minute
        self.lock = threading.Lock()
        self.next_ok = 0.0
        self.sent: collections.deque[float] = collections.deque()

    def acquire(self) -> None:
        self.sem.acquire()
        while True:
            with self.lock:
                now = time.time()
                while self.sent and now - self.sent[0] >= 60:
                    self.sent.popleft()
                wait = self.next_ok - now
                if self.per_minute and len(self.sent) >= self.per_minute:
                    wait = max(wait, 60 - (now - self.sent[0]))
                if wait <= 0:
                    self.next_ok = now + self.gap
                    self.sent.append(now)
                    return
            time.sleep(min(wait, 0.25))

    def release(self) -> None:
        self.sem.release()


_limits: dict[str, Limit] = {}
_limits_lock = threading.Lock()


def limit_for(vendor: str) -> Limit:
    key = vendor.split("_")[0]
    with _limits_lock:
        if key not in _limits:
            _limits[key] = Limit(**LIMITS.get(key, {"concurrent": 10}))
        return _limits[key]


def error_class(status: str, error: str | None) -> str | None:
    """Why a call failed: local (this host or network), account (credits, spend cap), rate_limit, vendor."""
    if status == "ok":
        return None
    e = str(error or "")
    if any(k in e for k in ("Too many open files", "ConnectionError", "SSLError", "NewConnectionError")):
        return "local"
    if any(k in e for k in ("insufficient_quota", "INSUFFICIENT_FUNDS", "pay-as-you-go", "add credits")) \
            or e.startswith(("HTTP 401", "HTTP 402", "HTTP 433")):
        return "account"
    if e.startswith("HTTP 429"):
        return "rate_limit"
    return "vendor"


def close_time(d: date, override: str = "") -> datetime:
    """The official close of trading day d, in UTC."""
    if override:
        return datetime.combine(d, dtime.fromisoformat(override), ET).astimezone(UTC)
    if d.year not in CALENDAR_YEARS:
        raise SystemExit(f"no exchange calendar for {d.year}; pass --close-time HH:MM (New York time)")
    if d.weekday() >= 5 or d in HOLIDAYS:
        raise SystemExit(f"{d} is not a trading day")
    return datetime.combine(d, dtime(13 if d in HALF_DAYS else 16), ET).astimezone(UTC)


def date_text(t: datetime) -> str:
    return f"{t:%B} {t.day}, {t.year}"


def query_for(item: dict, template: str, when_day: str, when_min: datetime) -> str:
    t = template.replace("{{date}}", "{date}").replace("{{time}}", "{time}")
    if item["asset"] == "equity":
        return t.format(name=item["name"], ticker=item["ticker"], date=when_day, time="")
    return t.format(name=item["name"], ticker=item["ticker"], date=date_text(when_min), time=f"{when_min:%H:%M} UTC")


def _call(vendor: str, query: str, base: dict) -> dict:
    lim = limit_for(vendor)
    lim.acquire()
    try:
        sent = time.time()
        try:
            vc = vendors.search(vendor, query, max_results=10)
            status, error, hits, cost = vc.status, vc.error, vc.hits or [], vc.cost_usd
        except Exception as exc:  # noqa: BLE001
            status, error, hits, cost = "error", str(exc)[:300], [], None
        recv = time.time()
    finally:
        lim.release()
    return base | {"vendor": vendor, "query": query, "t_sent": sent, "t_received": recv,
                   "latency_ms": round((recv - sent) * 1000), "status": status, "error": error,
                   "error_class": error_class(status, error), "vendor_cost_usd": cost, "hits": hits}


def collect(items: list[dict], vendor_keys: list[str], *, out_dir: Path, trade_date: date, target: datetime | None,
            offsets: list[int], close_override: str = "", rotate_wordings: bool = False,
            per_vendor_concurrency: int = 20) -> Path:
    close = close_time(trade_date, close_override)
    target = target or close
    if any(i["asset"] == "equity" for i in items) and target != close:
        raise SystemExit("stocks are asked at the close: run stocks and coins as separate jobs when --target "
                         "is not the close")
    when_day = date_text(close.astimezone(ET))
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"trade_date": trade_date.isoformat(), "close": close.isoformat(), "target": target.isoformat(),
           "offsets": offsets, "vendors": vendor_keys, "when_day": when_day,
           "when_minute": f"{date_text(target)} {target:%H:%M} UTC", "rotate_wordings": rotate_wordings,
           "items": items, "limits": LIMITS}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"{len(items)} items x {len(vendor_keys)} vendors at +{offsets}s after {target.isoformat()} -> {out_dir}",
          flush=True)

    raw = open(out_dir / "raw.jsonl", "a")
    lock = threading.Lock()
    pools = {v: ThreadPoolExecutor(max_workers=per_vendor_concurrency) for v in vendor_keys}
    finishers = []

    def finish(off, fire, jobs):
        rows = [j.result() for j in jobs]       # collected off the main thread so the next send is never late
        bad = collections.Counter(f"{r['vendor']}:{r['error_class']}" for r in rows if r["status"] != "ok")
        lat = sorted(r["latency_ms"] for r in rows)
        with lock:
            for r in rows:
                raw.write(json.dumps(r, default=str) + "\n")
            raw.flush()
            print(f"+{off}s: {len(rows)} calls, last sent {max(r['t_sent'] for r in rows) - fire:+.1f}s, "
                  f"latency median {lat[len(lat) // 2]} ms, not ok {sum(bad.values())} {dict(bad.most_common(8))}",
                  flush=True)

    for i, off in enumerate(offsets):
        fire = target.timestamp() + off
        late = time.time() - fire
        if late > 2:
            print(f"+{off}s: {late:.0f}s late, sending anyway", flush=True)
        while (wait := fire - time.time()) > 0:
            time.sleep(min(wait, 0.05 if wait < 2 else 1))
        jobs = []
        for it in items:
            tpl = WORDINGS[it["asset"]][i % len(WORDINGS[it["asset"]])] if rotate_wordings else it["query_template"]
            q = query_for(it, tpl, when_day, target)
            base = {"offset_s": off, "id": it["id"], "asset": it["asset"], "ticker": it["ticker"], "name": it["name"],
                    "size": it.get("stratum") or it.get("size")}
            jobs += [pools[v].submit(_call, v, q, base) for v in vendor_keys]
        t = threading.Thread(target=finish, args=(off, fire, jobs))
        t.start()
        finishers.append(t)
    for t in finishers:
        t.join()
    for p in pools.values():
        p.shutdown()
    raw.close()
    print("collection done", flush=True)
    return out_dir
