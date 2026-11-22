"""OpenAI-compatible LLM client with optional multi-endpoint failover and caching."""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class LLMClient(Protocol):
    """Minimal protocol required by ``MechSightAuto`` and ``MechSightGuided``."""

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float,
        tag: str = "",
        cache_key: str | None = None,
    ) -> str | None:
        """Return model response text, or ``None`` on unrecoverable failure."""


@dataclass(frozen=True)
class EndpointConfig:
    """Configuration for a single OpenAI-compatible chat-completions endpoint."""

    base_url: str | None
    model: str
    api_key: str
    concurrency: int = 128


class OpenAICompatibleClient:
    """Streaming chat-completions client with retries, failover, and disk caching.

    Reads ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, and ``MECHSIGHT_MODEL`` from
    the environment by default, and optionally a secondary failover gateway via
    ``FALLBACK_API_KEY`` / ``FALLBACK_BASE_URL``. Failed HTTP attempts are never
    cached.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        fallback_api_key: str | None = None,
        fallback_base_url: str | None = None,
        reasoning_effort: str | None = "high",
        max_tokens: int = 65536,
        timeout: float = 900.0,
        max_attempts: int = 8,
        cache_dir: str | Path | None = None,
    ) -> None:
        from openai import OpenAI

        primary_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        primary_model = model or os.environ.get("MECHSIGHT_MODEL", "gemini-3.8-flash")
        primary_url = base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL")

        endpoints: list[dict[str, Any]] = []
        if primary_key:
            endpoints.append(
                {
                    "model": primary_model,
                    "base_url": primary_url,
                    "api_key": primary_key,
                    "client": OpenAI(api_key=primary_key, base_url=primary_url, timeout=timeout, max_retries=0),
                }
            )
        fb_key = fallback_api_key or os.environ.get("FALLBACK_API_KEY", "")
        fb_url = fallback_base_url if fallback_base_url is not None else os.environ.get("FALLBACK_BASE_URL")
        if fb_key:
            endpoints.append(
                {
                    "model": os.environ.get("FALLBACK_MODEL", primary_model),
                    "base_url": fb_url,
                    "api_key": fb_key,
                    "client": OpenAI(api_key=fb_key, base_url=fb_url, timeout=timeout, max_retries=0),
                }
            )
        if not endpoints:
            raise ValueError(
                "No API key configured: pass `api_key=...` or export OPENAI_API_KEY (and optionally FALLBACK_API_KEY)."
            )

        self.endpoints = endpoints
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._lock = threading.Lock()

    def _cache_file(self, tag: str, cache_key: str | None, prompt: str, temperature: float) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = cache_key or hashlib.sha256(f"{tag}|{temperature}|{prompt}".encode()).hexdigest()[:24]
        safe_tag = (tag or "default").replace("/", "_")
        return self.cache_dir / safe_tag / f"{digest}.json"

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float,
        tag: str = "",
        cache_key: str | None = None,
    ) -> str | None:
        cfile = self._cache_file(tag, cache_key, prompt, temperature)
        if cfile is not None and cfile.exists():
            return json.loads(cfile.read_text()).get("answer")

        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            ep = self.endpoints[attempt % len(self.endpoints)]
            kwargs: dict[str, Any] = {
                "model": ep["model"],
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": temperature,
                "stream": True,
            }
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
            try:
                stream = ep["client"].chat.completions.create(**kwargs)
                parts: list[str] = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        parts.append(chunk.choices[0].delta.content)
                answer = "".join(parts)
                if not answer.strip():
                    raise RuntimeError("empty answer payload")
                if cfile is not None:
                    with self._lock:
                        cfile.parent.mkdir(parents=True, exist_ok=True)
                        cfile.write_text(
                            json.dumps(
                                {"tag": tag, "temperature": temperature, "model": ep["model"], "answer": answer},
                                ensure_ascii=False,
                                indent=1,
                            )
                        )
                return answer
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_attempts - 1:
                    time.sleep(min(30.0, 1.5 * (2 ** (attempt // len(self.endpoints)))) + random.random())
        raise RuntimeError(f"All {self.max_attempts} LLM attempts failed for tag={tag!r}: {last_exc}") from last_exc
