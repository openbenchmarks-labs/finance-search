"""The fixed parts of the benchmark: models, search budgets, prices and defaults.

The reference board is produced with exactly these settings. Change them and your numbers are no longer
comparable with the board.
"""

from __future__ import annotations

import os

# Hugging Face dataset repo holding the public sample. Override with --data.
DATASET_REPO = os.environ.get("FINSEARCH_DATASET_REPO", "openbenchmarks/OB-Finance-Search")
CONFIGS = {"historical-lookup": "single", "multi-hop-lookup": "multi"}   # dataset config -> search class

# One budget per search class, each on its own model and reasoning effort.
# mode "search" (the board): searches only. mode "search_fetch": adds a page-fetch budget.
# The env overrides are for deployment names that differ from the model name (e.g. on Azure);
# costs are always priced as the reference model.
REFERENCE_MODELS = {"single": ("gpt-5.6-luna", "high"), "multi": ("gpt-5.6-sol", "medium")}
CLASS_MODELS = {
    "single": (os.environ.get("FINSEARCH_SINGLE_MODEL", "gpt-5.6-luna"), "high"),
    "multi": (os.environ.get("FINSEARCH_MULTI_MODEL", "gpt-5.6-sol"), "medium"),
}
BUDGETS = {
    "search": {"single": ("single_s1", 1, 0), "multi": ("multi_s5", 5, 0)},
    "search_fetch": {"single": ("single_s1_f2", 1, 2), "multi": ("multi_s5_f10", 5, 10)},
}
MAX_RESULTS = 10            # results asked of the search API per call
FETCH_MAX_CHARS = 30_000    # page text handed to the agent per fetch
SAFETY_MAX_TURNS = 24       # turns are not budgeted; this only stops a runaway loop

JUDGE_MODEL = "claude-opus-5-5"
JUDGE_EFFORT = "medium"

# Date the agent is told is "today". The board was run with 2026-10-03; keep it for comparable numbers.
DEFAULT_TODAY = "2026-10-03"
DEFAULT_REPEATS = 3

# USD per million tokens: (uncached input, cached input, output). Used for the cost columns only.
PRICES = {
    "gpt-5.6-sol": (4.0, 0.2, 20.0),
    "gpt-5.6-luna": (0.2, 0.02, 1.2),
    JUDGE_MODEL: (4.0, 4.0, 20.0),
}


def class_setup(mode: str) -> dict[str, dict]:
    return {cls: {"config": cfg, "searches": s, "fetches": f,
                  "model": CLASS_MODELS[cls][0], "effort": CLASS_MODELS[cls][1],
                  "price_model": REFERENCE_MODELS[cls][0]}
            for cls, (cfg, s, f) in BUDGETS[mode].items()}


def llm_cost(model: str, input_tokens: int, cached: int, output_tokens: int) -> float:
    unc, cac, out = PRICES.get(model, (0.0, 0.0, 0.0))
    return ((input_tokens - cached) * unc + cached * cac + output_tokens * out) / 1e6
