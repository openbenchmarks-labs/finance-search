"""The search APIs on the reference board, each configured exactly as it was when the board was run.

Each config is one Adapter instance. Requests ask for 10 results; costs are list prices per call unless the
response reports its own cost.
"""

from __future__ import annotations

import math
import os
from typing import Any

from .base import Adapter, HttpRequest, dedupe, env, hit


def _json_headers(**extra: str) -> dict[str, str]:
    return {**extra, "Content-Type": "application/json"}


# --------------------------------------------------------------------------- Exa
class Exa(Adapter):
    env_keys = ("EXA_API_KEY",)
    native_fetch = True
    fetch_unit_cost_usd = 0.001

    def __init__(self, kind: str, price: float):
        self.kind, self.key, self.provider, self.search_unit_cost_usd = kind, f"exa_{kind}", f"Exa {kind}", price

    def search_request(self, query, max_results):
        return HttpRequest("POST", "https://api.exa.ai/search", _json_headers(**{"x-api-key": env("EXA_API_KEY")}),
                           {"query": query, "type": self.kind, "numResults": max_results,
                            "contents": {"highlights": True}},
                           timeout=120 if self.kind == "deep" else 60)

    def parse_hits(self, payload, max_results):
        hits = []
        for item in payload.get("results") or []:
            highlights = item.get("highlights") or []
            snippet = " ".join(x for x in highlights if isinstance(x, str)) or item.get("text")
            hits.append(hit(item.get("url"), item.get("title"), snippet))
        return dedupe(hits, max_results)

    def fetch_request(self, url, objective, max_chars):
        return HttpRequest("POST", "https://api.exa.ai/contents", _json_headers(**{"x-api-key": env("EXA_API_KEY")}),
                           {"urls": [url], "text": {"maxCharacters": max_chars, "includeHtmlTags": False}})

    def parse_page(self, payload, url):
        item = ((payload or {}).get("results") or [{}])[0]
        return item.get("text") or "", item.get("url") or url, item.get("title") or ""


# --------------------------------------------------------------------------- Parallel
class Parallel(Adapter):
    env_keys = ("PARALLEL_API_KEY",)
    native_fetch = True
    fetch_unit_cost_usd = 0.001

    def __init__(self, mode: str, price: float):
        self.mode, self.key, self.provider, self.search_unit_cost_usd = mode, f"parallel_{mode}", f"Parallel {mode}", price

    def search_request(self, query, max_results):
        return HttpRequest("POST", "https://api.parallel.ai/v1/search",
                           _json_headers(**{"x-api-key": env("PARALLEL_API_KEY")}),
                           {"objective": query, "search_queries": [query], "mode": self.mode,
                            "advanced_settings": {"max_results": max_results}},
                           timeout=120 if self.mode == "advanced" else 60)

    def parse_hits(self, payload, max_results):
        hits = [hit(i.get("url"), i.get("title"), "\n".join(x for x in i.get("excerpts") or [] if isinstance(x, str)))
                for i in payload.get("results") or []]
        return dedupe(hits, max_results)

    def fetch_request(self, url, objective, max_chars):
        return HttpRequest("POST", "https://api.parallel.ai/v1/extract",
                           _json_headers(**{"x-api-key": env("PARALLEL_API_KEY")}),
                           {"urls": [url], "objective": objective, "max_chars_total": max_chars,
                            "advanced_settings": {"full_content": True}}, timeout=120)

    def parse_page(self, payload, url):
        item = ((payload or {}).get("results") or [{}])[0]
        return (item.get("full_content") or "\n".join(item.get("excerpts") or []),
                item.get("url") or url, item.get("title") or "")


# --------------------------------------------------------------------------- You.com
KNOWLEDGE_URL = "you://knowledge"   # knowledge blocks have no page to fetch


class You(Adapter):
    """You.com web search with highlights. The `core` config also asks for the knowledge block, which often
    holds the figure itself with an as-of time; each block is added as a hit ahead of the web results."""
    env_keys = ("YOU_API_KEY",)
    native_fetch = True
    search_unit_cost_usd = 0.005
    fetch_unit_cost_usd = 0.001

    def __init__(self, core: bool):
        self.core = core
        self.key = "you_highlights_core" if core else "you_highlights"
        self.provider = "You.com highlights core" if core else "You.com highlights"

    def search_request(self, query, max_results):
        body = {"count": min(max_results, 20), "extraction": {"extraction_mode": "highlights"}, "query": query}
        if self.core:
            body["knowledge"] = "core"
        return HttpRequest("POST", "https://ydc-index.io/v1/search",
                           _json_headers(**{"X-API-Key": env("YOU_API_KEY")}), body)

    def parse_hits(self, payload, max_results):
        results = payload.get("results") or {}
        hits = []
        for item in results.get("web") or []:
            contents = item.get("contents") or {}
            snippets = contents.get("highlights") or item.get("snippets") or []
            snippet = "\n".join(v for v in snippets if isinstance(v, str))
            hits.append(hit(item.get("url"), item.get("title"), snippet or item.get("description"),
                            {"page_age": item.get("page_age")}))
        return knowledge_hits(payload) + dedupe(hits, max_results)

    def fetch_request(self, url, objective, max_chars):
        return HttpRequest("POST", "https://ydc-index.io/v1/contents",
                           _json_headers(**{"X-API-Key": env("YOU_API_KEY")}),
                           {"urls": [url], "formats": ["markdown", "metadata"]})

    def parse_page(self, payload, url):
        item = (payload if isinstance(payload, list) else [{}])[0]
        return item.get("markdown") or item.get("html") or "", item.get("url") or url, item.get("title") or ""


def knowledge_hits(payload: Any) -> list[dict[str, Any]]:
    results = payload.get("results") if isinstance(payload, dict) else None
    hits = []
    for i, item in enumerate((results or {}).get("knowledge") or []):
        if not isinstance(item, dict):
            continue
        sources = ", ".join(a.get("name", "") for a in item.get("attribution") or [] if isinstance(a, dict))
        parts = [str(item.get("description") or "").strip()]
        if item.get("as_of"):
            parts.append(f"(as of {item['as_of']})")
        if sources:
            parts.append(f"Sources: {sources}.")
        parts.append("(Answer box; not a web page, cannot be fetched.)")
        snippet = " ".join(p for p in parts if p)
        if snippet:
            hits.append({"url": f"{KNOWLEDGE_URL}/{i}", "title": str(item.get("title") or "You.com knowledge"),
                         "snippet": snippet[:4000],
                         "metadata": {"kind": "knowledge", "type": item.get("type"), "as_of": item.get("as_of"),
                                      "attribution": item.get("attribution")}})
    return hits


# --------------------------------------------------------------------------- Tavily
class Tavily(Adapter):
    env_keys = ("TAVILY_API_KEY",)
    native_fetch = True
    fetch_unit_cost_usd = 0.0032

    def __init__(self, key: str, depth: str, price: float, extra: dict[str, Any] | None = None):
        self.key, self.depth, self.search_unit_cost_usd = key, depth, price
        self.provider, self.extra = f"Tavily {depth}", extra or {}

    def search_request(self, query, max_results):
        return HttpRequest("POST", "https://api.tavily.com/search",
                           _json_headers(Authorization=f"Bearer {env('TAVILY_API_KEY')}"),
                           {"query": query, "max_results": max_results, "search_depth": self.depth, **self.extra})

    def parse_hits(self, payload, max_results):
        return dedupe([hit(i.get("url"), i.get("title"), i.get("content"), {"score": i.get("score")})
                       for i in payload.get("results") or []], max_results)

    def fetch_request(self, url, objective, max_chars):
        return HttpRequest("POST", "https://api.tavily.com/extract",
                           _json_headers(Authorization=f"Bearer {env('TAVILY_API_KEY')}"),
                           {"urls": [url], "extract_depth": "advanced"})

    def parse_page(self, payload, url):
        item = ((payload or {}).get("results") or [{}])[0]
        return item.get("raw_content") or item.get("content") or "", item.get("url") or url, ""


# --------------------------------------------------------------------------- Nimble
class Nimble(Adapter):
    env_keys = ("NIMBLE_API_KEY",)
    native_fetch = True

    def __init__(self, depth: str, price: float):
        self.depth, self.key, self.provider, self.search_unit_cost_usd = depth, f"nimble_{depth}", f"Nimble {depth}", price

    def fetch_cost(self):
        raw = os.environ.get("NIMBLE_EXTRACT_USD_PER_REQUEST", "").strip()   # no public list price
        if not raw:
            return None
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError("NIMBLE_EXTRACT_USD_PER_REQUEST must be finite and nonnegative")
        return value

    def search_request(self, query, max_results):
        return HttpRequest("POST", "https://sdk.nimbleway.com/v2/search",
                           _json_headers(Authorization=f"Bearer {env('NIMBLE_API_KEY')}"),
                           {"query": query, "search_depth": self.depth, "full_content": False,
                            "focus": "general", "max_results": max_results})

    def parse_hits(self, payload, max_results):
        rows = payload.get("results")
        hits = [hit(r["url"].strip(), r.get("title"), r.get("description"))
                for r in (rows if isinstance(rows, list) else [])
                if isinstance(r, dict) and isinstance(r.get("url"), str)]
        return dedupe(hits, max_results)

    def fetch_request(self, url, objective, max_chars):
        return HttpRequest("POST", "https://sdk.nimbleway.com/v2/extract",
                           _json_headers(Authorization=f"Bearer {env('NIMBLE_API_KEY')}"),
                           {"url": url, "formats": ["markdown"]})

    def parse_page(self, payload, url):
        if not isinstance(payload, dict) or payload.get("status") not in (None, "success"):
            raise RuntimeError("Nimble Extract returned an unsuccessful response")
        status = payload.get("status_code")
        if status is not None and not 200 <= int(status) < 300:
            raise RuntimeError(f"Nimble Extract upstream HTTP {status}")
        data = payload.get("data") or {}
        text = data.get("markdown") if isinstance(data, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Nimble Extract returned no markdown content")
        return text, payload.get("url") or url, ""


# --------------------------------------------------------------------------- Firecrawl
class Firecrawl(Adapter):
    key, provider = "firecrawl", "Firecrawl"
    env_keys = ("FIRECRAWL_API_KEY",)
    native_fetch = True
    search_unit_cost_usd = 0.005
    fetch_unit_cost_usd = 0.0025

    def search_request(self, query, max_results):
        return HttpRequest("POST", "https://api.firecrawl.dev/v2/search",
                           _json_headers(Authorization=f"Bearer {env('FIRECRAWL_API_KEY')}"),
                           {"query": query, "limit": max_results})

    def parse_hits(self, payload, max_results):
        data = payload.get("data") or {}
        rows = (data.get("web") or data.get("results") or []) if isinstance(data, dict) else data
        return dedupe([hit(i.get("url") or i.get("link"), i.get("title"), i.get("description") or i.get("snippet"))
                       for i in rows or []], max_results)

    def fetch_request(self, url, objective, max_chars):
        return HttpRequest("POST", "https://api.firecrawl.dev/v2/scrape",
                           _json_headers(Authorization=f"Bearer {env('FIRECRAWL_API_KEY')}"),
                           {"url": url, "formats": ["markdown"], "onlyMainContent": True})

    def parse_page(self, payload, url):
        item = (payload or {}).get("data") or {}
        meta = item.get("metadata") or {}
        return item.get("markdown") or "", meta.get("sourceURL") or url, meta.get("title") or ""


# --------------------------------------------------------------------------- Tako
class Tako(Adapter):
    key, provider = "tako", "Tako"
    env_keys = ("TAKO_API_KEY",)
    native_fetch = True
    search_unit_cost_usd = 0.007
    fetch_unit_cost_usd = 0.001

    def search_request(self, query, max_results):
        return HttpRequest("POST", "https://tako.com/api/v3/search", _json_headers(**{"X-API-Key": env("TAKO_API_KEY")}),
                           {"query": query, "effort": "fast",
                            "sources": {"web": {"count": min(max_results, 20), "highlights": True}}})

    def parse_hits(self, payload, max_results):
        return dedupe([hit(i.get("url"), i.get("title"), i.get("snippet"),
                           {"source_name": i.get("source_name"), "publish_date": i.get("publish_date")})
                       for i in payload.get("web_results") or []], max_results)

    def fetch_request(self, url, objective, max_chars):
        return HttpRequest("POST", "https://tako.com/api/v1/contents", _json_headers(**{"X-API-Key": env("TAKO_API_KEY")}),
                           {"url": url, "mode": "inline", "max_chars": max_chars}, timeout=120)

    def parse_page(self, payload, url):
        item = ((payload or {}).get("contents") or [{}])[0]
        return item.get("data") or "", item.get("source_url") or url, ""


# --------------------------------------------------------------------------- TinyFish
class TinyFish(Adapter):
    key, provider = "tinyfish", "TinyFish"
    env_keys = ("TINYFISH_API_KEY",)
    native_fetch = True
    search_unit_cost_usd = 0.0
    fetch_unit_cost_usd = 0.0

    def search_request(self, query, max_results):
        return HttpRequest("GET", "https://api.search.tinyfish.ai",
                           {"X-API-Key": env("TINYFISH_API_KEY"), "Accept": "application/json"},
                           params={"query": query}, timeout=45)

    def parse_hits(self, payload, max_results):
        return dedupe([hit(i.get("url"), i.get("title"), i.get("snippet"),
                           {"site_name": i.get("site_name"), "date": i.get("date")})
                       for i in payload.get("results") or []], max_results)

    def fetch_request(self, url, objective, max_chars):
        return HttpRequest("POST", "https://api.fetch.tinyfish.ai", _json_headers(**{"X-API-Key": env("TINYFISH_API_KEY")}),
                           {"urls": [url], "purpose": objective, "format": "markdown"}, timeout=120)

    def parse_page(self, payload, url):
        item = ((payload or {}).get("results") or [{}])[0]
        return item.get("text") or "", item.get("final_url") or item.get("url") or url, item.get("title") or ""


# --------------------------------------------------------------------------- search-only APIs
class Perplexity(Adapter):
    key, provider = "perplexity", "Perplexity"
    env_keys = ("PERPLEXITY_API_KEY",)
    search_unit_cost_usd = 0.005

    def search_request(self, query, max_results):
        return HttpRequest("POST", "https://api.perplexity.ai/search",
                           _json_headers(Authorization=f"Bearer {env('PERPLEXITY_API_KEY')}"),
                           {"query": query, "max_results": min(max_results, 20), "search_context_size": "high"})

    def parse_hits(self, payload, max_results):
        return dedupe([hit(i.get("url"), i.get("title"), i.get("snippet"),
                           {"date": i.get("date"), "last_updated": i.get("last_updated")})
                       for i in payload.get("results") or []], max_results)


class Brave(Adapter):
    key, provider = "brave", "Brave Search"
    env_keys = ("BRAVE_SEARCH_API_KEY",)
    search_unit_cost_usd = 0.005

    def search_request(self, query, max_results):
        return HttpRequest("GET", "https://api.search.brave.com/res/v1/web/search",
                           {"X-Subscription-Token": env("BRAVE_SEARCH_API_KEY"), "Accept": "application/json"},
                           params={"q": query, "count": min(max_results, 20), "result_filter": "web"}, timeout=45)

    def parse_hits(self, payload, max_results):
        return dedupe([hit(i.get("url"), i.get("title"), i.get("description") or i.get("snippet"))
                       for i in (payload.get("web") or {}).get("results") or []], max_results)


class Serp(Adapter):
    """Google results through a RapidAPI SERP endpoint."""
    key, provider = "serp", "SERP (RapidAPI)"
    env_keys = ("RAPIDAPI_KEY",)
    search_unit_cost_usd = 0.003

    def search_request(self, query, max_results):
        return HttpRequest("GET", "https://google-search74.p.rapidapi.com/",
                           {"x-rapidapi-key": env("RAPIDAPI_KEY"), "x-rapidapi-host": "google-search74.p.rapidapi.com"},
                           params={"query": query, "limit": max_results}, timeout=45)

    def parse_hits(self, payload, max_results):
        return dedupe([hit(i.get("url") or i.get("link"), i.get("title"), i.get("description") or i.get("snippet"))
                       for i in payload.get("results") or []], max_results)


def board_adapters() -> list[Adapter]:
    return [
        Exa("instant", 0.007), Exa("deep", 0.012),
        Parallel("advanced", 0.005), Parallel("basic", 0.005), Parallel("fast", 0.001),
        You(core=False), You(core=True),
        Tavily("tavily", "advanced", 0.016, {"chunks_per_source": 3}), Tavily("tavily_basic", "basic", 0.008),
        Nimble("standard", 0.005), Nimble("lite", 0.0011),
        Firecrawl(), Tako(), TinyFish(), Perplexity(), Brave(), Serp(),
    ]
