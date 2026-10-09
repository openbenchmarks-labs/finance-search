"""Model clients: the agent on any OpenAI-compatible Responses API endpoint, the judge on Anthropic."""

from __future__ import annotations

import os
from typing import Any


def agent_client(timeout: int = 240) -> Any:
    """OpenAI client for the agent. Reads OPENAI_API_KEY and, if set, OPENAI_BASE_URL (e.g. an Azure
    OpenAI v1 endpoint). The endpoint must serve the Responses API with reasoning items."""
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (see .env.example)")
    return OpenAI(timeout=timeout, max_retries=1)


def judge_client() -> Any:
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")
    return anthropic.Anthropic(max_retries=4)
