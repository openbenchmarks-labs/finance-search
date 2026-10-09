# Finance Search Benchmark

The harness behind the OpenBenchmarks finance search benchmark. It measures how much a search API helps an
agent answer the questions a finance professional actually asks. The agent, its prompt, its search budget and
the grader are fixed. The search API is the only thing that changes between runs.

## What is measured

| Task | Budget | Agent model | What a question looks like |
|---|---|---|---|
| Historical lookup | 1 search | gpt-5.6-luna, reasoning effort high | One published figure for a named entity and period. |
| Multi-hop lookup | up to 5 searches | gpt-5.6-sol, reasoning effort medium | Several published facts combined: a peer ranking, a trend, a result against guidance. |
| Live lookup | 1 query per item and delay | none (gpt-5.6-sol reads the price out of the results) | A stock's official close or a coin's price, asked seconds to hours after the moment. |

For the first two tasks the agent sees only the question, today's date and its budget. It gets the search
API's results (url, title, snippet) as tool output and must answer from them alone. Page fetching is off on the board (`--mode search`).

Each (question, search API) pair runs 3 times. Claude Opus 5.5 grades every answer against the reference
answer and its tolerance, using [the judge prompt](finsearch/prompts/judge.txt). An answer is correct only if every part the question asks for is right. The board reports the mean accuracy over the 3 repeats and the standard deviation across repeats.

Live lookup measures freshness. At 5 s, 30 s, 1 min, 15 min, 1 h and 3 h after the target moment, every item goes to every search API once. A model reads each response and reports the price it would quote. The answer keys come from market data pulled afterwards.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env              # then fill in the keys you need
```

The agent runs on any endpoint that serves the OpenAI Responses API with reasoning items. Set
`OPENAI_BASE_URL` for Azure or another gateway, and `FINSEARCH_SINGLE_MODEL` / `FINSEARCH_MULTI_MODEL` if your
deployment names differ from the model names. Costs are always priced as the reference models.

## Run

```bash
finsearch vendors                                         # the search APIs and the key each needs
finsearch run --vendors exa_instant --limit 2 --repeats 1 # smoke test: 2 questions per task
finsearch run --vendors exa_instant,tavily,parallel_basic  # full public sample, 3 repeats
```

`--data` picks the questions: `hf:<owner>/<name>[@revision]` (default `hf:openbenchmarks/OB-Finance-Search`),
a local folder holding `historical-lookup.jsonl` and `multi-hop-lookup.jsonl`, or a single `.jsonl` file whose
rows carry `search: single|multi`.

Each run writes `runs/<stamp>/`:

- `manifest.json`: settings, models, prices, question ids.
- `results.jsonl`: one row per (question, search API, repeat). It holds every search call with its raw
  results, the answer, the verdict and reason, tokens, latency and cost.
- `summary.md` and `summary.json`: the board numbers.

Live lookup is two steps:

```bash
# stocks: asked after the close of --date; grade the next day
finsearch live-collect --date 2026-10-07 --assets equity --vendors exa_instant,tavily
# coins: asked after a whole UTC minute
finsearch live-collect --date 2026-10-07 --assets crypto --target 2026-10-07T17:30:00Z --vendors exa_instant,tavily
finsearch live-grade runs/live/crypto_20261007-172900 --coin-key keys/
finsearch live-grade runs/live/equity_20261007-155800 --closes keys/
```

Start `live-collect` before the target moment and leave it running until the last delay (3 hours by default).
Calls are paced per API key, so configs that share a key share its rate limit.

### Live answer keys

Live answers are graded against answer keys built from licensed and exchange market data, recorded after the fact. We don't publish these keys. To grade a live run you collected, plug in your own data by implementing `CloseSource` or `CoinSource` in [finsearch/live/keys.py](finsearch/live/keys.py). Use `--closes mock` and `--coin-key mock` to check the pipeline without keys.

The key files are JSONL. Stocks have one row per trading day and ticker:

```json
{"date": "2026-10-07", "ticker": "EQT", "close": 52.31, "prior_close": 52.47}
```

Coins have one row per ticker and UTC minute (the minute's start):

```json
{"ticker": "XRP", "minute": "2026-10-07T17:29:00Z", "low": 1.3812, "high": 1.3851}
```

`--closes` and `--coin-key` each take a file, a folder holding the files, or `hf:<owner>/<name>`. Pass your own
source with `--close-source` / `--coin-source module:ClassName`.

## Adding a search API

Subclass `Adapter` in [finsearch/vendors/base.py](finsearch/vendors/base.py). Build the HTTP request for a query and turn the response into hits. Add the fetch pair too if the API can extract a page. The 17 adapters in [finsearch/vendors/adapters.py](finsearch/vendors/adapters.py) are working examples.

```python
from finsearch.vendors import Adapter, HttpRequest, dedupe, hit, register
from finsearch.vendors.base import env

class Acme(Adapter):
    key, provider, env_keys = "acme", "Acme Search", ("ACME_API_KEY",)
    search_unit_cost_usd = 0.004

    def search_request(self, query, max_results):
        return HttpRequest("POST", "https://api.acme.example/search",
                           {"Authorization": f"Bearer {env('ACME_API_KEY')}"},
                           {"q": query, "n": max_results})

    def parse_hits(self, payload, max_results):
        return dedupe([hit(r["url"], r["title"], r["text"]) for r in payload.get("results", [])], max_results)

register(Acme())
```

```bash
finsearch --vendor-module my_acme run --vendors acme
```

Keep the request at the API's defaults unless a setting is part of the configuration you want ranked. Return the API's own text as the snippet, with nothing summarised or rewritten on your side.

## Tests

```bash
pip install -e '.[dev]' && pytest
```

The tests use a scripted model and a fake search API, so they make no network calls.

## License

MIT
