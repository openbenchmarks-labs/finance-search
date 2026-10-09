"""The agent loop against a scripted model and a fake search API: budgets, fetch guard, answer capture."""
import json
from types import SimpleNamespace as NS

from finsearch import vendors
from finsearch.agent import run_agent
from finsearch.vendors import Adapter, HttpRequest


class FakeVendor(Adapter):
    key, provider, env_keys, native_fetch = "fake", "Fake", (), True
    search_unit_cost_usd, fetch_unit_cost_usd = 0.01, 0.001

    def search(self, query, *, max_results=10):
        return vendors.VendorCall("ok", 5, {"q": query}, {}, None, [], 0.01,
                                  hits=[{"url": "https://seen.example/a", "title": "T", "snippet": f"about {query}"}])

    def fetch(self, url, *, objective, max_chars):
        return vendors.VendorCall("ok", 5, {"url": url}, {}, None, [], 0.001,
                                  page={"final_url": url, "title": "T", "text": "page", "truncated": False})


vendors.register(FakeVendor())


def call(name, args, cid):
    return NS(type="function_call", name=name, arguments=json.dumps(args), call_id=cid, id=cid)


class ScriptedModel:
    def __init__(self, turns):
        self.turns, self.seen = list(turns), []
        self.responses = self

    def create(self, **kw):
        self.seen.append(kw)
        out = self.turns.pop(0)
        usage = NS(input_tokens=100, output_tokens=10, output_tokens_details=NS(reasoning_tokens=4),
                   input_tokens_details=NS(cached_tokens=20))
        if isinstance(out, str):
            return NS(output=[], output_text=out, usage=usage)
        return NS(output=out, output_text="", usage=usage)


def test_budget_and_fetch_guard():
    model = ScriptedModel([
        [call("web_search", {"query": "q1"}, "c1"), call("web_search", {"query": "q2"}, "c2")],
        [call("web_fetch", {"url": "https://unseen.example", "objective": "x"}, "c3"),
         call("web_fetch", {"url": "https://seen.example/a", "objective": "x"}, "c4")],
        "Answer: 42\nSources: https://seen.example/a",
    ])
    r = run_agent("q", "What?", client=model, vendor="fake", config="c", max_searches=1, max_fetches=2,
                  model="m", effort="high", today="2026-10-03")
    assert r.finished and r.answer.startswith("Answer: 42")
    assert len(r.searches) == 1                      # second search refused: budget 1
    assert len(r.fetches) == 1                       # unseen URL refused
    assert any("unseen" in n for n in r.notes)
    first = model.seen[0]
    assert first["reasoning"] == {"effort": "high"} and first["store"] is False
    assert "Today's date is 2026-10-03." in first["input"][0]["content"]
    assert r.to_dict()["vendor_cost_usd"] == 0.011
    assert r.input_tokens == 300 and r.cached_input_tokens == 60


def test_tools_withdrawn_when_budget_spent():
    model = ScriptedModel([[call("web_search", {"query": "q1"}, "c1")], "Answer: x"])
    run_agent("q", "What?", client=model, vendor="fake", config="c", max_searches=1, max_fetches=0,
              model="m", effort="medium")
    assert [t["name"] for t in model.seen[0]["tools"]] == ["web_search"]
    assert model.seen[1]["tools"] == []
