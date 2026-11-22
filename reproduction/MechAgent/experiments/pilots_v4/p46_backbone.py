"""P46: generic external backbone on T3_confirm (gpt-6.1-sol, claude-sonnet-5-5-high; 2026-11-08), copied from p38.
Key is read from the env var named by --key-env; never written to disk.

Runs the frozen MechSight-Auto pipeline exactly as the second-backbone run (p17_argon_confirm.py): candidate pool
(Direct, samp1..4, enum3) -> atom mapping -> state completion -> revision -> second round if a gap remains.
Only the endpoint and model change. Self-consistency is the MBR vote over the same pool (no extra calls).
Outputs: OUT/luna/<ds>/<tag>/<cid>.json; summary OUT/SUMMARY_LUNA_T3.json. Post hoc, not pre-registered.

python p38_luna_t3.py --phase ping|pool|requests|r1|r2|analyze [--workers 40]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time

from openai import OpenAI

import common4 as K
import p11_mechagent as M
import p17_argon_confirm as A
import p3_selector as P3

MODEL = None
KEY_ENV = 'OPENAI_API_KEY'
BASE = 'https://LLM-GATEWAY-2.example/v1'
SLUG = None
WALL = 900
_SEM = None
EFFORT = None  # set by --effort; None = model default (no reasoning)


def luna_call(tag, cid, prompt, temperature=0.2, system=K.SYSTEM, cap=40):
    global _SEM
    dest = A.ARGON_OUT / tag / f'{cid}.json'
    if dest.exists():
        return json.loads(dest.read_text())['answer']
    trace = A.ARGON_OUT / tag / 'traces' / f'{cid}.jsonl'
    trace.parent.mkdir(parents=True, exist_ok=True)
    if _SEM is None:
        _SEM = threading.Semaphore(cap)
    client = OpenAI(api_key=os.environ[KEY_ENV], base_url=BASE, timeout=WALL, max_retries=0)
    start = time.monotonic()
    for attempt in range(A.MAX_ATTEMPTS):
        with _SEM:
            try:
                t0 = time.monotonic()
                stream = client.chat.completions.create(
                    model=MODEL, messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}],
                    max_tokens=32768, temperature=temperature, stream=True, stream_options={'include_usage': True},
                    **({'reasoning_effort': EFFORT} if EFFORT else {}))
                parts, usage, returned = [], None, None
                for chunk in stream:
                    if time.monotonic() - t0 > WALL:
                        stream.close()
                        raise TimeoutError(f'stream exceeded {WALL}s')
                    returned = chunk.model
                    if chunk.usage:
                        usage = chunk.usage.model_dump()
                    if chunk.choices and chunk.choices[0].delta.content:
                        parts.append(chunk.choices[0].delta.content)
                answer = ''.join(parts)
                if not answer.strip():
                    raise RuntimeError('empty answer')
                dest.write_text(json.dumps({'id': cid, 'tag': tag, 'model': MODEL, 'endpoint': BASE, 'reasoning_effort': EFFORT,
                                            'returned_model': returned, 'temperature': temperature, 'usage': usage,
                                            'prompt': prompt, 'answer': answer}, ensure_ascii=False, indent=1))
                with trace.open('a') as f:
                    f.write(json.dumps({'phase': tag, 'model': MODEL, 'returned_model': returned, 'usage': usage,
                                        'elapsed_seconds': round(time.monotonic() - start, 1)}) + '\n')
                return answer
            except Exception as exc:  # noqa: BLE001
                with trace.open('a') as f:
                    f.write(json.dumps({'phase': tag, 'api_error': type(exc).__name__,
                                        'status': getattr(exc, 'status_code', None), 'retry': attempt,
                                        'detail': str(exc)[:300]}) + '\n')
        time.sleep(min(30, 3 * (attempt + 1)) + random.random() * 3)
    return None


A.argon_call = luna_call


def analyze(all_sets):
    rep = {}
    for ds, cases in all_sets.items():
        rows = {a: [] for a in ('Direct', 'SelfConsistency', 'MechSight_Auto')}
        strata = {}
        for cid, case in cases.items():
            base = A.acached(f'{ds}/base_default', cid)
            if base is None or A.acached(f'{ds}/V5_rev1', cid) is None:
                continue
            A.reload_stepmap_cache()
            p = A.apool(ds, cid)
            pick = P3.consensus(p, list(p)) if len(p) >= 2 else None
            ans = {'Direct': K.parse_mech(base), 'SelfConsistency': p.get(pick) if pick else K.parse_mech(base),
                   'MechSight_Auto': K.parse_mech(A.afinal(ds, cid))}
            for a, m in ans.items():
                s = K.score(case, m)
                rows[a].append((cid, s))
                if 'pgl' in case:
                    h = s['n_hidden_events']
                    strata.setdefault('0' if h == 0 else ('1-2' if h <= 2 else '>=3'), {}).setdefault(a, []).append(
                        s['all_present'])
        if not rows['Direct']:
            continue
        keys = ['all_present', 'in_order_frac', 'heavy_hidden_recall'] if ds == 'T3_confirm' else ['all_present']
        keys = [k for k in keys if k in rows['Direct'][0][1] or k == 'all_present']
        block = {'model': MODEL, 'reasoning_effort': EFFORT, 'n': len(rows['Direct']), **{a: K.summarize(r, keys) for a, r in rows.items()}}
        for a in ('SelfConsistency', 'MechSight_Auto'):
            block[f'{a}_minus_Direct'] = K.paired(rows[a], rows['Direct'], 'all_present')
        block['MechSight_Auto_minus_SelfConsistency'] = K.paired(rows['MechSight_Auto'], rows['SelfConsistency'], 'all_present')
        if strata:
            block['strata_all_present'] = {b: {a: [len(v), round(sum(v) / len(v), 3)] for a, v in d.items()}
                                           for b, d in sorted(strata.items())}
        rep[ds] = block
    (K.OUT / f"SUMMARY_BB_{SLUG}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['ping', 'pool', 'requests', 'r1', 'r2', 'analyze'], required=True)
    ap.add_argument('--sets', default='T3_confirm')
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--effort', default=None)
    ap.add_argument('--model', required=True)
    ap.add_argument('--key-env', required=True)
    cli = ap.parse_args()
    MODEL, KEY_ENV, EFFORT = cli.model, cli.key_env, cli.effort
    SLUG = MODEL.replace('.', '_') + ('_' + EFFORT if EFFORT else '')
    A.ARGON_OUT = K.OUT / f'bb_{SLUG}'
    A.ARGON_REQ = K.OUT / f'stepmap_bb_{SLUG}_requests.jsonl'
    A.ARGON_CACHE = K.OUT / f'stepmap_bb_{SLUG}_cache.json'
    if cli.phase == 'ping':
        c = OpenAI(api_key=os.environ[KEY_ENV], base_url=BASE, timeout=120, max_retries=0)
        r = c.chat.completions.create(model=MODEL, messages=[{'role': 'user', 'content': 'Reply with OK.'}], max_tokens=2000, **({'reasoning_effort': EFFORT} if EFFORT else {}))
        print('ping', r.model, repr(r.choices[0].message.content))
        raise SystemExit
    A.reload_stepmap_cache()
    s = M.sets(tuple(cli.sets.split(',')))
    if cli.phase == 'pool':
        arms = ['base_default'] + [f'samp{i}' for i in range(1, 5)] + ['enum3']
        K.run_pool(A.job_pool, [(ds, a, c, cli.workers) for ds, cs in s.items() for a in arms for c in cs.values()],
                   cli.workers)
    elif cli.phase == 'requests':
        A.requests(s)
    elif cli.phase == 'r1':
        K.run_pool(A.job_r1, [(ds, c, cli.workers) for ds, cs in s.items() for c in cs.values()], cli.workers)
    elif cli.phase == 'r2':
        A.requests(s)
        A.reload_stepmap_cache()
        K.run_pool(A.job_r2, [(ds, c, cli.workers) for ds, cs in s.items() for c in cs.values()], cli.workers)
    elif cli.phase == 'analyze':
        analyze(s)
