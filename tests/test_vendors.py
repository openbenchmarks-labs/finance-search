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


# --------------------------------------------------------------------------- Keenable
def _reply(status, payload):
    return base.VendorCall("ok" if status < 400 else "error", 5, {"url": "x"}, payload,
                           None if status < 400 else f"HTTP {status}: {payload}",
                           [{"attempt": 1, "status_code": status, "latency_ms": 5}])


def _scripted(monkeypatch, replies):
    sent = []

    def fake_send(req, **_):
        sent.append(req)
        return replies.pop(0)

    monkeypatch.setattr(base, "send", fake_send)
    monkeypatch.setattr(base, "assert_public_url", lambda url: None)
    return sent


def test_keenable_arms_are_on_the_board_and_share_one_rate_limit():
    from finsearch.live import collect
    assert {"keenable_pro", "keenable_realtime"} <= set(vendors.BOARD_VENDORS)
    assert vendors.required_env(["keenable_pro", "keenable_realtime"]) == ["KEENABLE_API_KEY"]
    assert collect.limit_for("keenable_pro") is collect.limit_for("keenable_realtime")
    assert collect.LIMITS["keenable"] == {"rps": 8, "concurrent": 8}


def test_keenable_search_request_contract(monkeypatch):
    monkeypatch.setenv("KEENABLE_API_KEY", "k-secret")
    for mode in ("pro", "realtime"):
        req = vendors.get(f"keenable_{mode}").search_request("AAPL close 2026-10-07", 10)
        assert (req.method, req.url) == ("POST", "https://api.keenable.ai/v1/search")
        assert req.headers == {"X-API-Key": "k-secret", "Content-Type": "application/json"}
        assert req.body == {"query": "AAPL close 2026-10-07", "mode": mode, "max_results": 10}
    assert vendors.get("keenable_pro").search_request("q", 80).body["max_results"] == 50
    assert vendors.get("keenable_pro").search_request("q", 0).body["max_results"] == 1


def test_keenable_parse_hits_prefers_snippet_and_dedupes():
    k = vendors.get("keenable_pro")
    payload = {"results": [
        {"title": "A", "url": "https://a.example/", "description": "", "snippet": "long snippet",
         "published_at": "2026-10-07"},
        {"title": "B", "url": "https://b.example/", "description": "desc only", "snippet": ""},
        {"title": "no url", "description": "x", "snippet": "y"},
        {"title": "A again", "url": "https://a.example/", "snippet": "dup"},
        "junk",
    ]}
    hits = k.parse_hits(payload, 10)
    assert [(h["url"], h["snippet"]) for h in hits] == [("https://a.example/", "long snippet"),
                                                       ("https://b.example/", "desc only")]
    assert hits[0]["metadata"]["published_at"] == "2026-10-07"
    assert k.parse_hits({}, 10) == [] and k.parse_hits({"results": None}, 10) == []
    assert k.parse_hits(None, 10) == []


def test_keenable_search_cost_uses_credits_else_list_price(monkeypatch):
    _scripted(monkeypatch, [_reply(200, {"results": [], "usage": {"credits": 2}}), _reply(200, {"results": []}),
                            _reply(500, {"error": "boom"})])
    assert vendors.search("keenable_pro", "q").cost_usd == 0.008
    assert vendors.search("keenable_realtime", "q").cost_usd == 0.004
    assert vendors.search("keenable_pro", "q").cost_usd == 0.0
    k = vendors.get("keenable_pro")
    assert k.reported_cost({"usage": {"credits": True}}) is None
    assert k.reported_cost({"usage": {"credits": "2"}}) is None and k.reported_cost(None) is None


def test_keenable_fetch_indexed_copy_needs_no_live_retry(monkeypatch):
    sent = _scripted(monkeypatch, [_reply(200, {"url": "https://a.example/", "title": "A", "content": "page text",
                                                "published_at": 1791331200})])
    vc = vendors.fetch("keenable_pro", "https://a.example/", objective="o", max_chars=30000)
    assert vc.status == "ok" and vc.page["text"] == "page text" and vc.page["title"] == "A"
    assert len(sent) == 1 and (sent[0].method, sent[0].url) == ("GET", "https://api.keenable.ai/v1/fetch")
    assert sent[0].params == {"url": "https://a.example/", "max_chars": 30000, "live": "false"}
    assert vc.cost_usd == 0.004 and [a["live"] for a in vc.attempts] == [False]


def test_keenable_fetch_retries_live_once_on_404_or_empty(monkeypatch):
    for first in (_reply(404, {"detail": "not indexed"}), _reply(200, {"url": "https://a.example/", "content": ""})):
        sent = _scripted(monkeypatch, [first, _reply(200, {"content": "live text", "usage": {"credits": 1}})])
        vc = vendors.fetch("keenable_realtime", "https://a.example/", objective="o", max_chars=100)
        assert vc.status == "ok" and vc.page["text"] == "live text"
        assert [r.params["live"] for r in sent] == ["false", "true"]
        assert [(a["status_code"], a["live"]) for a in vc.attempts] == [(first.attempts[0]["status_code"], False),
                                                                         (200, True)]
        assert vc.cost_usd == 0.004   # the failed indexed attempt costs nothing


def test_keenable_fetch_both_attempts_fail(monkeypatch):
    sent = _scripted(monkeypatch, [_reply(422, {"detail": "bad"}), _reply(404, {"detail": "gone"})])
    vc = vendors.fetch("keenable_pro", "https://a.example/", objective="o", max_chars=100)
    assert vc.status == "error" and vc.page is None and vc.cost_usd == 0.0 and len(sent) == 2
    assert [a["live"] for a in vc.attempts] == [False, True]


def test_keenable_fetch_no_live_retry_on_other_errors(monkeypatch):
    sent = _scripted(monkeypatch, [_reply(401, {"detail": "bad key"})])
    vc = vendors.fetch("keenable_pro", "https://a.example/", objective="o", max_chars=100)
    assert vc.status == "error" and len(sent) == 1


def test_api_key_headers_are_redacted():
    class Resp:
        ok, status_code = True, 200

        def json(self):
            return {}

    vc = base.send(base.HttpRequest("GET", "https://x.example", {"X-API-Key": "k-secret", "x-api-key": "k2"}),
                   request_fn=lambda **_: Resp())
    assert vc.raw_request["headers"] == {"X-API-Key": base.REDACTED, "x-api-key": base.REDACTED}
    assert "k-secret" not in str(vc.raw_request)
