"""finsearch: run the finance search benchmark.

  finsearch vendors                                   list the search APIs and the key each needs
  finsearch run --vendors exa_instant,tavily          historical + multi-hop lookup, 3 repeats, graded
  finsearch run --resume runs/<stamp>                 continue a stopped run
  finsearch rejudge runs/<stamp> [--ids a,b]          re-grade failed judge calls (or given questions)
  finsearch summary runs/<stamp>                      rebuild summary.md from results.jsonl
  finsearch live-collect --date 2026-10-07 --assets equity --vendors exa_instant
  finsearch live-collect --date 2026-10-07 --assets crypto --target 2026-10-07T17:30:00Z
  finsearch live-grade runs/live/<stamp> --closes keys/ --coin-key keys/   grade against the answer keys
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from . import config, vendors


def _vendor_list(raw: str) -> list[str]:
    keys = [v.strip() for v in raw.split(",") if v.strip()]
    for k in keys:
        vendors.get(k)
    return keys


def cmd_vendors(_args) -> None:
    for k, a in vendors.REGISTRY.items():
        have = all(os.environ.get(e) for e in a.env_keys)
        print(f"{k:22} {a.provider:26} fetch={'yes' if a.native_fetch else 'no ':3}  "
              f"${a.search_unit_cost_usd}/search  {', '.join(a.env_keys)}{'' if have else '  (key not set)'}")


def cmd_run(args) -> None:
    from .data import load_questions
    from .runner import check_env, run

    out_dir = Path(args.resume) if args.resume else Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    manifest = out_dir / "manifest.json"
    if args.resume and manifest.exists():          # a resumed run keeps its settings
        m = json.loads(manifest.read_text())
        args.data, args.vendors, args.repeats = m["data"], ",".join(m["vendors"]), m["repeats"]
        args.today, args.mode, args.ids = m["today_in_prompt"], m["mode"], ",".join(m["question_ids"])
    tasks = tuple(t for t in args.tasks.split(",") if t)
    questions = load_questions(args.data, tasks)
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    if args.limit:
        questions = [q for cls in ("single", "multi") for q in [x for x in questions if x["search"] == cls][: args.limit]]
    keys = _vendor_list(args.vendors)
    check_env(keys)
    run(questions, keys, out_dir=out_dir, repeats=args.repeats, mode=args.mode, today=args.today,
        data_label=args.data or f"hf:{config.DATASET_REPO}",
        single_concurrency=args.single_concurrency, multi_concurrency=args.multi_concurrency)


def cmd_rejudge(args) -> None:
    from .data import load_questions
    from .runner import rejudge

    out_dir = Path(args.run_dir)
    m = json.loads((out_dir / "manifest.json").read_text())
    rejudge(out_dir, load_questions(args.data or m["data"]), ids=set(args.ids.split(",")) if args.ids else None)


def cmd_summary(args) -> None:
    from .summary import write_summary

    print(write_summary(Path(args.run_dir)).read_text())


def cmd_live_collect(args) -> None:
    from .data import load_live_items
    from .live.collect import DEFAULT_OFFSETS, collect

    target = datetime.fromisoformat(args.target.replace("Z", "+00:00")).astimezone(timezone.utc) if args.target else None
    if target and (target.second or target.microsecond):
        raise SystemExit("--target must be a whole minute (coins are graded on 1-minute candles)")
    assets = tuple(a for a in args.assets.split(",") if a)
    items = [i for i in load_live_items(args.data) if i["asset"] in assets]
    if args.ids:
        wanted = set(args.ids.split(","))
        items = [i for i in items if i["id"] in wanted]
    if not items:
        raise SystemExit("no live items selected")
    keys = _vendor_list(args.vendors)
    missing = [k for k in vendors.required_env(keys) if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"missing environment variables: {', '.join(missing)}")
    offsets = [int(x) for x in args.offsets.split(",") if x] if args.offsets else list(DEFAULT_OFFSETS)
    out_dir = Path(args.out) / f"{'-'.join(assets)}_{time.strftime('%Y%m%d-%H%M%S')}"
    collect(items, keys, out_dir=out_dir, trade_date=date.fromisoformat(args.date), target=target, offsets=offsets,
            close_override=args.close_time, rotate_wordings=args.rotate_wordings)
    print(f"grade with: finsearch live-grade {out_dir}")


def cmd_live_grade(args) -> None:
    from .live.grade import grade
    from .live.keys import close_source, coin_source

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (answers are extracted with the agent model)")
    assets = tuple(a for a in args.assets.split(",") if a)
    grade(Path(args.run_dir), assets, close_source=close_source(args.closes, args.close_source),
          coin_source=coin_source(args.coin_key, args.coin_source))


def main(argv: list[str] | None = None) -> None:
    load_dotenv(Path.cwd() / ".env")
    ap = argparse.ArgumentParser(prog="finsearch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vendor-module", action="append", default=[], help="import a module that registers extra adapters")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("vendors", help="list search APIs").set_defaults(fn=cmd_vendors)

    p = sub.add_parser("run", help="historical and multi-hop lookup")
    p.add_argument("--data", default="", help="hf:<owner>/<name>[@rev], a local folder, or a .jsonl file "
                                              f"(default hf:{config.DATASET_REPO})")
    p.add_argument("--tasks", default="historical-lookup,multi-hop-lookup")
    p.add_argument("--vendors", required=False, default="exa_instant")
    p.add_argument("--repeats", type=int, default=config.DEFAULT_REPEATS)
    p.add_argument("--mode", choices=sorted(config.BUDGETS), default="search",
                   help="search (the board): no page fetching; search_fetch: adds a fetch budget")
    p.add_argument("--today", default=config.DEFAULT_TODAY, help="date the agent is told is today")
    p.add_argument("--ids", default="", help="comma-separated question ids")
    p.add_argument("--limit", type=int, default=0, help="first N questions of each task (smoke tests)")
    p.add_argument("--out", default="runs")
    p.add_argument("--resume", default="", help="run directory to continue")
    p.add_argument("--single-concurrency", type=int, default=64)
    p.add_argument("--multi-concurrency", type=int, default=32)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("rejudge", help="re-grade a run")
    p.add_argument("run_dir")
    p.add_argument("--ids", default="", help="re-grade every row of these questions (after an answer-key change)")
    p.add_argument("--data", default="", help="default: the dataset recorded in the run's manifest")
    p.set_defaults(fn=cmd_rejudge)

    p = sub.add_parser("summary", help="rebuild a run's summary")
    p.add_argument("run_dir")
    p.set_defaults(fn=cmd_summary)

    p = sub.add_parser("live-collect", help="live lookup: ask the vendors at set delays after the target")
    p.add_argument("--date", required=True, help="trading day whose close stocks are asked about (YYYY-MM-DD)")
    p.add_argument("--target", default="", help="UTC moment coins are asked about (default: that day's close)")
    p.add_argument("--assets", default="equity,crypto")
    p.add_argument("--offsets", default="", help="seconds after the target (default 5,30,60,900,3600,10800)")
    p.add_argument("--vendors", default="exa_instant")
    p.add_argument("--ids", default="")
    p.add_argument("--data", default="")
    p.add_argument("--close-time", default="", help="HH:MM New York time, for days outside the built-in calendar")
    p.add_argument("--rotate-wordings", action="store_true", help="a different query wording at each delay")
    p.add_argument("--out", default="runs/live")
    p.set_defaults(fn=cmd_live_collect)

    p = sub.add_parser("live-grade", help="grade a live lookup run")
    p.add_argument("run_dir")
    p.add_argument("--assets", default="equity,crypto")
    p.add_argument("--closes", default="", help="stock answer key: a JSONL file, a folder, hf:<owner>/<name>, or mock")
    p.add_argument("--close-source", default="", help="module:Class implementing CloseSource (gets --closes as its argument)")
    p.add_argument("--coin-key", default="", help="coin answer key: a JSONL file, a folder, hf:<owner>/<name>, or mock")
    p.add_argument("--coin-source", default="", help="module:Class implementing CoinSource (gets --coin-key as its argument)")
    p.set_defaults(fn=cmd_live_grade)

    args = ap.parse_args(argv)
    for m in args.vendor_module:
        vendors.load_module(m)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
