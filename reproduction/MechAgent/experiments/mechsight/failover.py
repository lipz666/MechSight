"""Two-gateway client (vps proxy + AI Studio gateway), same Gemini model on both.

The gateways have separate concurrency limits that add up (300 each). Each gateway gets a
semaphore; a call goes to the gateway with more free slots and holds the slot while it
streams. A failed attempt releases the slot and the retry goes to the other gateway.
When several processes share the gateways, give each one a slice with FAILOVER_CAP_VPS /
FAILOVER_CAP_117 (default 300 each); FAILOVER_PREFER=117 (or vps) fills that gateway first. Keys come only from the environment: OPENAI_API_KEY
for the vps proxy, FALLBACK_API_KEY for the gateway. Every logged event records the gateway
that answered.
"""
from __future__ import annotations

import os
import random
import threading
import time
from pathlib import Path

import common as C

ENDPOINTS = [
    {'base_url': 'https://LLM-GATEWAY-2.example/v1', 'model': 'gemini-3.8-flash-high', 'key_env': 'OPENAI_API_KEY',
     'cap_env': 'FAILOVER_CAP_VPS'},
    {'base_url': 'http://LLM-GATEWAY-1.example:8080/v1', 'model': 'gemini-3.8-flash', 'key_env': 'FALLBACK_API_KEY',
     'cap_env': 'FAILOVER_CAP_117'},
]
MAX_ATTEMPTS = 14
_LOCK = threading.Lock()
_SLOTS = {}  # base_url -> [semaphore, free count]


def _slots(ep):
    with _LOCK:
        if ep['base_url'] not in _SLOTS:
            cap = int(os.environ.get(ep['cap_env'], 300))
            _SLOTS[ep['base_url']] = [threading.Semaphore(cap), cap]
        return _SLOTS[ep['base_url']]


def _acquire(endpoints, avoid=None):
    """Take a slot on the gateway with most free slots (not `avoid` when there is a choice)."""
    while True:
        with _LOCK:
            options = [e for e in endpoints if e is not avoid] or endpoints
            prefer = os.environ.get('FAILOVER_PREFER', '')  # e.g. '117': fill that gateway first
            options = sorted(options, key=lambda e: (not (prefer and prefer in e['base_url']), -_SLOTS[e['base_url']][1]))
            for ep in options:
                slot = _SLOTS[ep['base_url']]
                if slot[0].acquire(blocking=False):
                    slot[1] -= 1
                    return ep
        time.sleep(0.2 + random.random() * 0.3)


def _release(ep):
    with _LOCK:
        slot = _SLOTS[ep['base_url']]
        slot[0].release()
        slot[1] += 1


class FailoverModel(C.Model):
    def __init__(self, trace, max_tokens=65536, temperature=0.2, timeout=900):
        from openai import OpenAI
        self.trace, self.max_tokens, self.temperature = Path(trace), max_tokens, temperature
        self.model = 'gemini-3.8-flash(-high)'
        self.endpoints = [dict(e, client=OpenAI(api_key=os.environ[e['key_env']], base_url=e['base_url'],
                                                timeout=timeout, max_retries=0))
                          for e in ENDPOINTS if os.environ.get(e['key_env'])
                          and os.environ.get(e['cap_env'], '300') != '0']  # cap 0 = gateway switched off
        for e in self.endpoints:
            _slots(e)
        self.events = []

    def _redact(self, text):
        for e in self.endpoints:
            text = text.replace(os.environ[e['key_env']], '[REDACTED]')
        return text

    def complete(self, messages, phase):
        start, last = time.monotonic(), None
        for attempt in range(MAX_ATTEMPTS):
            ep = _acquire(self.endpoints, avoid=last)
            try:
                stream = ep['client'].chat.completions.create(
                    model=ep['model'], messages=messages, max_tokens=self.max_tokens, temperature=self.temperature,
                    reasoning_effort='high', stream=True, stream_options={'include_usage': True})
                parts, usage, returned = [], None, None
                for chunk in stream:
                    returned = chunk.model
                    if chunk.usage:
                        usage = chunk.usage.model_dump()
                    if chunk.choices and chunk.choices[0].delta.content:
                        parts.append(chunk.choices[0].delta.content)
                answer = ''.join(parts)
                if not answer.strip():
                    raise RuntimeError('empty answer')
                _release(ep)
                break
            except Exception as exc:
                _release(ep)
                last = ep
                self._save({'phase': phase, 'api_error': type(exc).__name__, 'status': getattr(exc, 'status_code', None),
                            'endpoint': ep['base_url'], 'retry': attempt, 'detail': self._redact(str(exc))[:300],
                            'elapsed_seconds': round(time.monotonic() - start, 1)})
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                if attempt % 2 == 1:  # both gateways failed in a row: back off
                    time.sleep(min(60, 3 * 2 ** (attempt // 2)) + random.random() * 3)
        event = {'phase': phase, 'model': ep['model'], 'endpoint': ep['base_url'], 'returned_model': returned,
                 'usage': usage, 'generation_tokens': C.generation_tokens(usage),
                 'elapsed_seconds': round(time.monotonic() - start, 1), 'answer': answer}
        self.events.append(event)
        self._save(event)
        return answer

    @property
    def endpoints_used(self):
        return sorted({e['endpoint'] for e in self.events})
