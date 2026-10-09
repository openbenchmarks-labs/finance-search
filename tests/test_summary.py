import math

from finsearch.summary import compute_summary


def row(q, v, rep, ok, task="historical-lookup", **kw):
    return {"question_id": q, "vendor": v, "repeat": rep, "correct": ok, "task": task, **kw}


def test_mean_sd_flip_and_judge_errors():
    rows = [row("a", "x", 0, True), row("b", "x", 0, False),
            row("a", "x", 1, True), row("b", "x", 1, True),
            row("a", "x", 2, True), row("b", "x", 2, True, judge_error="parse")]
    s = compute_summary(rows)
    cell = next(c for c in s["cells"] if c["task"] == "historical-lookup")
    per_rep = [0.5, 1.0, 1.0]
    m = sum(per_rep) / 3
    sd = math.sqrt(sum((x - m) ** 2 for x in per_rep) / 2)
    assert math.isclose(cell["accuracy"], m) and math.isclose(cell["sd"], sd)
    assert cell["flip"] == 0.5                     # b flips (False, True); a doesn't
    assert s["judge_errors"] == 1 and s["n_graded"] == 5


def test_all_row_pools_tasks():
    rows = [row("a", "x", 0, True), row("m", "x", 0, False, task="multi-hop-lookup")]
    cells = {c["task"]: c for c in compute_summary(rows)["cells"]}
    assert cells["all"]["accuracy"] == 0.5 and cells["all"]["questions"] == 2
