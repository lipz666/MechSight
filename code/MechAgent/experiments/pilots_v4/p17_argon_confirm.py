"""P17: Second-backbone confirmation with gemini-4-argon (Fig. 3b).

Runs the exact frozen MechSight v5 pipeline (Direct, samp1..4, enum3, stepmap, V5_comp1/rev1,
V5_comp2/rev2, plus format-only relabelling on oMe-Gold for S_partial) using gemini-4-argon on
http://LLM-GATEWAY-1.example:8080/v1 (key from FALLBACK_API_KEY).
Outputs are stored under OUT/argon/<ds>/<tag>/<cid>.json and summarized in SUMMARY_ARGON_CONFIRM.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from pathlib import Path

from openai import OpenAI

import common4 as K
import p2_enumerate as P2
import p3_selector as P3
import p11_mechagent as M
import p15_relabel as RL
import stepmap as SMAP

ARGON_OUT = K.OUT / 'argon'
ARGON_REQ = K.OUT / 'stepmap_argon_requests.jsonl'
ARGON_CACHE = K.OUT / 'stepmap_argon_cache.json'
ENDPOINTS = [
    {'base_url': 'http://LLM-GATEWAY-1.example:8080/v1', 'model': 'gemini-4-argon', 'key_env': 'FALLBACK_API_KEY'},
]
MAX_ATTEMPTS = 8
_SEM = None


def reload_stepmap_cache():
    c = json.loads(SMAP.CACHE.read_text()) if SMAP.CACHE.exists() else {}
    if ARGON_CACHE.exists():
        c.update(json.loads(ARGON_CACHE.read_text()))
    M._cache = c


def _get_sem(cap):
    global _SEM
    if _SEM is None:
        _SEM = threading.Semaphore(cap)
    return _SEM


def argon_call(tag, cid, prompt, temperature=0.2, system=K.SYSTEM, cap=80):
    dest = ARGON_OUT / tag / f'{cid}.json'
    if dest.exists():
        return json.loads(dest.read_text())['answer']
    trace = ARGON_OUT / tag / 'traces' / f'{cid}.jsonl'
    trace.parent.mkdir(parents=True, exist_ok=True)
    eps = [
        dict(e, client=OpenAI(api_key=os.environ[e['key_env']], base_url=e['base_url'], timeout=180, max_retries=0))
        for e in ENDPOINTS
        if os.environ.get(e['key_env'])
    ]
    sem = _get_sem(cap)
    start = time.monotonic()
    answer, usage, returned, used_ep = None, None, None, None
    for attempt in range(MAX_ATTEMPTS):
        ep = eps[attempt % len(eps)]
        effort = 'high' if attempt == 0 else ('medium' if attempt <= 2 else 'low')
        with sem:
            try:
                att_start = time.monotonic()
                stream = ep['client'].chat.completions.create(
                    model=ep['model'],
                    messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}],
                    max_tokens=32768,
                    temperature=temperature,
                    reasoning_effort=effort,
                    stream=True,
                    stream_options={'include_usage': True},
                )
                parts = []
                for chunk in stream:
                    if time.monotonic() - att_start > 180:
                        stream.close()
                        raise TimeoutError('stream exceeded 180s wall clock')
                    returned = chunk.model
                    if chunk.usage:
                        usage = chunk.usage.model_dump()
                    if chunk.choices and chunk.choices[0].delta.content:
                        parts.append(chunk.choices[0].delta.content)
                answer = ''.join(parts)
                if not answer.strip():
                    raise RuntimeError('empty answer')
                used_ep = ep
                break
            except Exception as exc:
                err_rec = {
                    'phase': tag,
                    'endpoint': ep['base_url'],
                    'api_error': type(exc).__name__,
                    'status': getattr(exc, 'status_code', None),
                    'retry': attempt,
                    'detail': str(exc)[:300],
                    'elapsed_seconds': round(time.monotonic() - start, 1),
                }
                with trace.open('a') as f:
                    f.write(json.dumps(err_rec, ensure_ascii=False) + '\n')
                if attempt == MAX_ATTEMPTS - 1:
                    return None
        time.sleep(min(30, 3 * (attempt + 1)) + random.random() * 3)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                'id': cid,
                'tag': tag,
                'model': used_ep['model'],
                'endpoint': used_ep['base_url'],
                'returned_model': returned,
                'temperature': temperature,
                'usage': usage,
                'prompt': prompt,
                'answer': answer,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return answer


def acached(tag, cid):
    f = ARGON_OUT / tag / f'{cid}.json'
    return json.loads(f.read_text())['answer'] if f.exists() else None


def apool(ds, cid):
    names = ['base_default'] + [f'samp{i}' for i in range(1, 5)]
    out = {}
    for n in names:
        m = K.parse_mech(acached(f'{ds}/{n}', cid))
        if m:
            out[n] = m
    for i, m in enumerate(P2.paths(acached(f'{ds}/enum3', cid)), 1):
        if m:
            out[f'enum{i}'] = m
    return out


def acand_list(ds, cid):
    p = apool(ds, cid)
    out, seen = [], set()
    for n in P3.order(cid, p):
        key = tuple(M.canon(s.get('intermediate_smiles')) for s in p[n])
        if key not in seen:
            seen.add(key)
            out.append(p[n])
    return out


def job_pool(ds, arm, case, cap):
    cid = case['id']
    if arm == 'base_default':
        return 'ok' if argon_call(f'{ds}/base_default', cid, K.official_prompt(case), temperature=0.2, cap=cap) else 'error'
    if arm.startswith('samp'):
        return 'ok' if argon_call(f'{ds}/{arm}', cid, K.official_prompt(case), temperature=0.7, cap=cap) else 'error'
    if arm == 'enum3':
        return 'ok' if argon_call(f'{ds}/enum3', cid, P2.prompt(case), temperature=0.7, cap=cap) else 'error'
    return 'skip'


def requests(all_sets):
    import subprocess
    reqs = set()
    for ds, cases in all_sets.items():
        for cid, case in cases.items():
            mechs = acand_list(ds, cid)
            m = K.parse_mech(acached(f'{ds}/V5_rev1', cid))
            if m:
                mechs.append(m)
            for m in mechs:
                reqs.update(r for r in M.step_rxns(case, m) if r)
    ARGON_REQ.write_text('\n'.join(json.dumps({'rxn': r}) for r in sorted(reqs)) + '\n')
    print(len(reqs), 'argon step reactions requested')
    code = (
        'import json, sys; sys.path.insert(0, "."); import stepmap as S; from rxnmapper import RXNMapper; '
        f'req, cache_p, base_p = {str(ARGON_REQ)!r}, {str(ARGON_CACHE)!r}, {str(SMAP.CACHE)!r}; '
        'import pathlib; '
        'base = json.loads(pathlib.Path(base_p).read_text()) if pathlib.Path(base_p).exists() else {}; '
        'cache = json.loads(pathlib.Path(cache_p).read_text()) if pathlib.Path(cache_p).exists() else {}; '
        'todo = sorted({json.loads(l)["rxn"] for l in pathlib.Path(req).read_text().splitlines() if l} - base.keys() - cache.keys()); '
        'print(len(todo), "argon to map", flush=True); '
        'mapper = RXNMapper() if todo else None\n'
        'for i in range(0, len(todo), 16):\n'
        '    chunk = todo[i:i+16]\n'
        '    try: res = mapper.get_attention_guided_atom_maps(chunk, canonicalize_rxns=False)\n'
        '    except Exception: res = [None]*len(chunk)\n'
        '    for rxn, r in zip(chunk, res):\n'
        '        try: cache[rxn] = S.n_changes(r["mapped_rxn"]) if r else None\n'
        '        except Exception: cache[rxn] = None\n'
        'pathlib.Path(cache_p).write_text(json.dumps(cache)); print("argon map done", len(cache), flush=True)'
    )
    subprocess.run(['../../../external/env-map/bin/python', '-c', code], check=True)


def job_r1(ds, case, cap):
    cid = case['id']
    cands = acand_list(ds, cid)
    if len(cands) < 2:
        return 'skip'
    text = M.render(case, cands)
    comp = argon_call(f'{ds}/V5_comp1', cid, M.COMPLETE.format(cands=text, **M.fields(case)), temperature=0.2, cap=cap)
    if comp is None:
        return 'error'
    st = M.checked_states(case, comp)
    states = '\n'.join(f'- ({loc}) {s}' for loc, s in st) or '- none'
    return (
        'ok'
        if argon_call(f'{ds}/V5_rev1', cid, M.REVISE.format(cands=text, states=states, **M.fields(case)), temperature=0.2, cap=cap)
        else 'error'
    )


def job_r2(ds, case, cap):
    cid = case['id']
    m = K.parse_mech(acached(f'{ds}/V5_rev1', cid))
    if not m or not M.gaps(case, m):
        return 'nogap'
    text = M.render(case, [m])
    comp = argon_call(f'{ds}/V5_comp2', cid, M.COMPLETE.format(cands=text, **M.fields(case)), temperature=0.2, cap=cap)
    if comp is None:
        return 'error'
    st = M.checked_states(case, comp)
    if not st:
        return 'nostates'
    states = '\n'.join(f'- ({loc}) {s}' for loc, s in st)
    return (
        'ok'
        if argon_call(f'{ds}/V5_rev2', cid, M.REVISE.format(cands=text, states=states, **M.fields(case)), temperature=0.2, cap=cap)
        else 'error'
    )


def afinal(ds, cid):
    a2 = acached(f'{ds}/V5_rev2', cid)
    if a2 is not None and K.parse_mech(a2):
        return a2
    return acached(f'{ds}/V5_rev1', cid)


def job_relabel(case, cap):
    cid = case['id']
    orig = K.parse_mech(afinal('gold', cid))
    if not orig:
        return 'noparse'
    steps = '\n'.join(f"  step {i}: {s['intermediate_smiles']}" for i, s in enumerate(orig, 1))
    prompt = RL.PROMPT.format(allowed=json.dumps(RL.ALLOWED, indent=1), steps=steps)
    return 'ok' if argon_call('gold/RL_MechAgent', cid, prompt, temperature=0.0, cap=cap) else 'error'


def arelabelled(cid):
    orig = K.parse_mech(afinal('gold', cid))
    lab = K.parse_mech(acached('gold/RL_MechAgent', cid))
    if not orig or not lab or len(orig) != len(lab):
        return orig
    return [
        {
            'step': i,
            'type': str(l.get('type', '')),
            'subtype': str(l.get('subtype', '')),
            'intermediate_smiles': o['intermediate_smiles'],
        }
        for i, (o, l) in enumerate(zip(orig, lab), 1)
    ]


def analyze(all_sets):
    rep, pooled = {}, {}
    for ds, cases in all_sets.items():
        rows = {a: [] for a in ('Direct', 'SelfConsistency', 'MechAgent')}
        strata = {}
        for cid, case in cases.items():
            base = acached(f'{ds}/base_default', cid)
            if base is None or acached(f'{ds}/V5_rev1', cid) is None:
                continue
            reload_stepmap_cache()
            p = apool(ds, cid)
            sc_pick = P3.consensus(p, list(p)) if len(p) >= 2 else None
            ma = arelabelled(cid) if ds == 'gold' and acached('gold/RL_MechAgent', cid) else K.parse_mech(afinal(ds, cid))
            ans = {
                'Direct': K.parse_mech(base),
                'SelfConsistency': p.get(sc_pick) if sc_pick else K.parse_mech(base),
                'MechAgent': ma,
            }
            for a, m in ans.items():
                s = K.score(case, m)
                rows[a].append((cid, s))
                pooled.setdefault(a, []).append((f'{ds}:{cid}', s))
                if 'pgl' in case:
                    h = s['n_hidden_events']
                    b = '0' if h == 0 else ('1-2' if h <= 2 else '>=3')
                    strata.setdefault(b, {}).setdefault(a, []).append(s['all_present'])
        if not rows['Direct']:
            continue
        keys = ['all_present', 'in_order_frac'] + (['S_partial', 'S_total', 'in_order_full'] if ds == 'gold' else [])
        block = {a: K.summarize(r, keys) for a, r in rows.items() if r}
        for a in ('SelfConsistency', 'MechAgent'):
            block[f'{a}_minus_Direct'] = K.paired(rows[a], rows['Direct'], 'all_present')
        block['MechAgent_minus_SelfConsistency'] = K.paired(rows['MechAgent'], rows['SelfConsistency'], 'all_present')
        if ds == 'gold':
            block['MechAgent_minus_Direct_S_partial'] = K.paired(rows['MechAgent'], rows['Direct'], 'S_partial')
        if strata:
            block['strata_all_present'] = {
                b: {a: [len(v), round(sum(v) / len(v), 3)] for a, v in d.items()} for b, d in sorted(strata.items())
            }
        rep[ds] = block
    if pooled.get('Direct'):
        rep['pooled_primary'] = {
            a: K.summarize([(c, {'all_present': m['all_present']}) for c, m in r], ['all_present'])
            for a, r in pooled.items()
        }
        for a in ('SelfConsistency', 'MechAgent'):
            rep['pooled_primary'][f'{a}_minus_Direct'] = K.paired(pooled[a], pooled['Direct'], 'all_present')
    (K.OUT / 'SUMMARY_ARGON_CONFIRM.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['pool', 'requests', 'r1', 'r2', 'relabel', 'analyze'], required=True)
    ap.add_argument('--sets', default='T3_confirm')
    ap.add_argument('--workers', type=int, default=45)
    cli = ap.parse_args()
    reload_stepmap_cache()
    s = M.sets(tuple(cli.sets.split(',')))
    if cli.phase == 'pool':
        arms = ['base_default'] + [f'samp{i}' for i in range(1, 5)] + ['enum3']
        K.run_pool(job_pool, [(ds, a, c, cli.workers) for ds, cs in s.items() for a in arms for c in cs.values()], cli.workers)
    elif cli.phase == 'requests':
        requests(s)
    elif cli.phase == 'r1':
        K.run_pool(job_r1, [(ds, c, cli.workers) for ds, cs in s.items() for c in cs.values()], cli.workers)
    elif cli.phase == 'r2':
        K.run_pool(job_r2, [(ds, c, cli.workers) for ds, cs in s.items() for c in cs.values()], cli.workers)
    elif cli.phase == 'relabel':
        if 'gold' in s:
            K.run_pool(job_relabel, [(c, cli.workers) for c in s['gold'].values()], cli.workers)
    elif cli.phase == 'analyze':
        analyze(s)
