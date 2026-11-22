from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class OpenAIChatConfig:
    model: str = "gemma4-31b"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 8192
    enable_thinking: bool = True
    send_thinking_kwargs: bool = True
    timeout: float = 300.0
    max_retries: int = 2
    trace_path: Optional[str] = None


@dataclass
class ChatResult:
    answer: str
    reasoning: Optional[str]
    message: Dict[str, Any]
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value:
                    parts.append(str(value))
        return "\n".join(parts)
    return str(content)


class OpenAIChatModel:
    """Small OpenAI-compatible chat wrapper used by both agents.

    Reasoning is recorded for inspection but never concatenated into the answer
    or subsequent prompts. Protocol parsers consume only ``message.content``.
    """

    def __init__(self, config: OpenAIChatConfig, client: Any = None):
        self.config = config
        self.events: List[Dict[str, Any]] = []
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("openai is required for the OpenAI-compatible runtime") from exc
            key = config.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
            client = OpenAI(
                api_key=key,
                base_url=config.base_url,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
        self.client = client

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        phase: str,
        enable_thinking: Optional[bool] = None,
    ) -> ChatResult:
        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.send_thinking_kwargs:
            thinking = (
                self.config.enable_thinking
                if enable_thinking is None
                else enable_thinking
            )
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": thinking}
            }
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        if hasattr(message, "model_dump"):
            message_dict = message.model_dump()
        elif isinstance(message, dict):
            message_dict = dict(message)
        else:
            message_dict = {
                "content": getattr(message, "content", None),
                "reasoning": getattr(message, "reasoning", None),
                "reasoning_content": getattr(message, "reasoning_content", None),
            }
        reasoning = message_dict.get("reasoning") or message_dict.get("reasoning_content")
        answer = _content_text(message_dict.get("content"))
        if not answer.strip() and not reasoning:
            raise ValueError(f"empty assistant content for phase={phase}")
        finish_reason = getattr(choice, "finish_reason", None)
        usage_obj = getattr(response, "usage", None)
        if hasattr(usage_obj, "model_dump"):
            usage = usage_obj.model_dump()
        elif isinstance(usage_obj, dict):
            usage = dict(usage_obj)
        else:
            usage = None
        event = {
            "phase": phase,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "enable_thinking": (
                self.config.enable_thinking
                if enable_thinking is None
                else enable_thinking
            ),
            "finish_reason": finish_reason,
            "usage": usage,
            "reasoning": reasoning,
            "answer": answer,
        }
        self.events.append(event)
        self._append_trace(event)
        return ChatResult(
            answer=answer,
            reasoning=reasoning,
            message=message_dict,
            finish_reason=finish_reason,
            usage=usage,
        )

    def _append_trace(self, event: Dict[str, Any]) -> None:
        if not self.config.trace_path:
            return
        path = Path(self.config.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
