"""Live lookup calls two adapters differently; everything else is identical in both modes."""

import os

os.environ.setdefault("TAKO_API_KEY", "test")

from finsearch import vendors
from finsearch.vendors import adapters, base


def test_exa_live_request_adds_livecrawl_only_in_live_mode():
    exa = vendors.get("exa_instant")
    assert "livecrawl" not in exa.search_request("XRP price", 10).body["contents"]
    live = exa.live_search_request("XRP price", 10)
    assert live.body["contents"] == {"highlights": True, "livecrawl": "always"}
    assert live.body["type"] == "instant" and live.timeout == 120


def test_tako_live_request_adds_data_source_and_cards_come_first():
    tako = vendors.get("tako")
    assert set(tako.search_request("XRP price", 10).body["sources"]) == {"web"}
    assert set(tako.live_search_request("XRP price", 10).body["sources"]) == {"data", "web"}
    payload = {
        "cards": [{"card_id": "c1", "title": "XRP to USD conversion rate",
                   "description": "The current rate is $1.36036 per XRP.", "semantic_description": "Source: CoinMarketCap.",
                   "webpage_url": "https://tako.com/card/c1",
                   "content": {"dataset": {"columns": [{"name": "Timestamp"}, {"name": "XRP (USD)"}],
                                           "rows": [["2026-10-08T18:50:00+00:00", 1.3603], ["2026-10-08T18:55:00+00:00", 1.3610]]}}}],
        "web_results": [{"url": "https://coinmarketcap.com/currencies/xrp/", "title": "XRP price", "snippet": "$1.36"}],
    }
    assert [h["url"] for h in tako.parse_hits(payload, 10)] == ["https://coinmarketcap.com/currencies/xrp/"]
    live = tako.live_parse_hits(payload, 10)
    assert [h["url"] for h in live] == ["https://tako.com/card/c1", "https://coinmarketcap.com/currencies/xrp/"]
    assert live[0]["metadata"] == {"kind": "card", "card_id": "c1", "n_rows": 2}
    assert "1.361" in live[0]["snippet"] and "cannot be fetched" in live[0]["snippet"]


def test_search_uses_live_hooks_only_when_asked(monkeypatch):
    sent = []

    def fake_send(req, **_):
        sent.append(req.body)
        return base.VendorCall("ok", 1, {}, {"results": []}, None)

    monkeypatch.setattr(base, "send", fake_send)
    vendors.search("exa_deep", "ETH price", max_results=5)
    vendors.search("exa_deep", "ETH price", max_results=5, live=True)
    assert "livecrawl" not in sent[0]["contents"] and sent[1]["contents"]["livecrawl"] == "always"


def test_other_adapters_are_unchanged_in_live_mode():
    class Acme(base.Adapter):
        key, provider, env_keys = "acme_test", "Acme", ()

        def search_request(self, query, max_results):
            return base.HttpRequest("GET", "https://acme.example/s", {}, params={"q": query})

        def parse_hits(self, payload, max_results):
            return [base.hit("https://acme.example/1", "one", "1")]

    a = Acme()
    assert a.live_search_request("q", 10) == a.search_request("q", 10)
    assert a.live_parse_hits({}, 10) == a.parse_hits({}, 10)
    assert isinstance(adapters.card_hits({"cards": "not a list"}), list) and adapters.card_hits({}) == []
