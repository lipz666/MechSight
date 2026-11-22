from __future__ import annotations

import json
import re
from typing import Any, Dict

from .schemas import ExecutorProposal


def extract_tag(text: str, tag: str, *, required: bool = True) -> str:
    pattern = re.compile(rf"<{re.escape(tag)}\b[^>]*>(.*?)</{re.escape(tag)}>", re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(text or "")
    if not matches:
        if required:
            raise ValueError(f"missing <{tag}>...</{tag}> block")
        return ""
    if len(matches) != 1:
        raise ValueError(f"expected exactly one <{tag}>...</{tag}> block")
    return matches[0].strip()


def _strip_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def parse_json_tag(text: str, tag: str) -> Dict[str, Any]:
    try:
        payload = extract_tag(text, tag)
    except ValueError as exc:
        # Thinking-capable chat templates sometimes return the requested JSON
        # in a markdown fence but omit the outer protocol tag. Accept that
        # single-object form while retaining all schema validation downstream.
        if not str(exc).startswith(f"missing <{tag}>"):
            raise
        payload = text
    payload = _strip_fence(payload)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(payload[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"<{tag}> must contain one JSON object")
    return parsed


def parse_executor_output(
    text: str,
    *,
    proposal_id: str,
    directive_id: str,
    review_round: int,
) -> ExecutorProposal:
    event_summary = extract_tag(text, "event_summary")
    plan_objection = extract_tag(text, "plan_objection")
    action_pgl = extract_tag(text, "action")
    if not action_pgl and not plan_objection:
        raise ValueError("Executor must provide either <action> or <plan_objection>")
    return ExecutorProposal(
        proposal_id=proposal_id,
        directive_id=directive_id,
        review_round=review_round,
        event_summary=event_summary,
        plan_objection=plan_objection,
        action_pgl=action_pgl,
    )
