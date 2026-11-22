"""Shared pieces for the v4 pilots (2026-11-03): datasets, the official oMeBench prompt and
output format, LLM calls through the two gemini-3.8-flash gateways, and scoring.

Every arm writes the official oMeBench JSON mechanism (step, type, subtype,
intermediate_smiles), so one format is scored two ways:
  oMe-style cases (oMe-Gold, oMe-Silver): official utils_eval.oMeS (subtype + SMILES, step
      weights) -> S_total, S_partial, V, L; plus species-level in-order / all-present.
  Fukuyama cases (J3, T3): exact PGL ground truth via smiles_mech.score -> all GT
      intermediates present, in-order fraction, heavy hidden events recalled.
Run with external/venv-llm (openai + rdkit). Keys only from the environment
(OPENAI_API_KEY = vps, FALLBACK_API_KEY = 117 gateway).
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'smiles'))
sys.path.insert(0, str(HERE.parent / 'mechsight'))
sys.path.insert(0, str(HERE.parent / 'agent_v2'))
import smiles_mech as SM  # noqa: E402
from smiles_mech import A, C, P, PL, chem, phase2  # noqa: E402
import failover as F  # noqa: E402
import j4_score  # noqa: E402

ROOT = P.ROOT
OUT = ROOT / 'outputs' / 'pilots_v4_20261103'
OME = ROOT.parent / 'external' / 'oMeBench'
sys.path.insert(0, str(OME / 'scripts'))
from utils_eval import oMeS  # noqa: E402

PROMPTS = {k: (OME / 'prompts' / f'{k}.txt').read_text() for k in ('default', 'cot')}
SYSTEM = 'You are an expert organic chemist who writes rigorous stepwise arrow-pushing mechanisms.'


# ------------------------------------------------------------------ datasets
def _ome_case(d, src):
    return {'id': d['reaction_id'], 'src': src, 'name': d.get('name', ''), 'level': d.get('level'),
            'reactants': d['reactants_smiles'], 'products': d['products_smiles'], 'conditions': d.get('conditions', ''),
            'gold': d['mechanism']}


def load_gold():
    return {c['id']: c for c in (_ome_case(d, 'gold') for d in json.loads((OME / 'data/oMe_Gold.json').read_text()))}


def load_template():
    return [_ome_case(d, 'template') for d in json.loads((OME / 'data/oMe_Template.json').read_text())]


def load_silver():
    return [_ome_case(json.loads(l), 'silver') for l in (OME / 'data/oMe_Silver.jsonl').read_text().splitlines() if l]


def norm_name(x):
    return re.sub(r'[^a-z0-9]+', ' ', str(x or '').lower()).strip()


def silver_dev(n=60, seed=20261103):
    """One held-out Silver reaction for each of n names with >= 5 Silver entries (fixed seed)."""
    split = OUT / 'silver_dev.json'
    silver = load_silver()
    if not split.exists():
        by = {}
        for c in silver:
            by.setdefault(norm_name(c['name']), []).append(c['id'])
        names = sorted(k for k, v in by.items() if len(v) >= 5)
        rng = random.Random(seed)
        pick = sorted(rng.choice(sorted(by[k])) for k in rng.sample(names, n))
        split.parent.mkdir(parents=True, exist_ok=True)
        split.write_text(json.dumps({'ids': pick, 'note': f'seed {seed}; one per name, names with >=5 entries'}, indent=1))
    ids = set(json.loads(split.read_text())['ids'])
    return {c['id']: c for c in silver if c['id'] in ids}


def _fuku_case(case, src):
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    target = chem.parse_pgl_literal(case['target_state_pgl'])
    return {'id': case['case_id'], 'src': src, 'name': '', 'level': None,
            'reactants': P.endpoint_smiles(initial).split('.'), 'products': P.endpoint_smiles(target).split('.'),
            'conditions': A.conditions_text(case), 'pgl': case}


def load_fukuyama(which):
    """which: 'J3' (147 diagnostics), 'T3_dev' (47) or 'T3_confirm' (48)."""
    if which == 'J3':
        import j3_dev
        return {cid: _fuku_case(c, 'J3') for cid, c in j3_dev.load_cases().items()}
    import baselines as B
    t3 = B.load_sets()['T3']
    ids = json.loads((ROOT / 'outputs/benchmarks_20261103/splits.json').read_text())[which]
    return {cid: _fuku_case(t3[cid], which) for cid in ids}


# ------------------------------------------------------------------ prompting / calls
def official_prompt(case, kind='default'):
    t = PROMPTS[kind]
    for token, value in {'{ reactants_smiles }': case['reactants'], '{ products_smiles }': case['products'],
                         '{ conditions }': case['conditions']}.items():
        t = t.replace(token, json.dumps(value))
    return t


def call(tag, cid, prompt, temperature=0.2, system=SYSTEM):
    """One cached gemini-3.8-flash call; returns the answer text (None on failure)."""
    dest = OUT / tag / f'{cid}.json'
    if dest.exists():
        return json.loads(dest.read_text())['answer']
    m = F.FailoverModel(OUT / tag / 'traces' / f'{cid}.jsonl', max_tokens=P.MAX_TOKENS, temperature=temperature)
    try:
        ans = m.complete([{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}], tag)
    except Exception:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({'id': cid, 'tag': tag, 'temperature': temperature, 'endpoints': m.endpoints_used,
                                'prompt': prompt, 'answer': ans}, ensure_ascii=False, indent=1))
    return ans


def cached(tag, cid):
    f = OUT / tag / f'{cid}.json'
    return json.loads(f.read_text())['answer'] if f.exists() else None


def parse_mech(text):
    """Official-format mechanism (list of step dicts) from an answer, or None."""
    if not text:
        return None
    m = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', text, re.S)
    body = m.group(1) if m else text
    body = re.sub(r'```(?:json)?', '', body)
    lo, hi = body.find('['), body.rfind(']')
    if lo < 0 or hi <= lo:
        return None
    try:
        out = json.loads(body[lo:hi + 1])
    except Exception:
        return None
    if isinstance(out, dict):
        out = out.get('mechanism')
    if not isinstance(out, list):
        return None
    return [s for s in out if isinstance(s, dict) and isinstance(s.get('intermediate_smiles'), str)]


def tagged(text, tag):
    m = re.search(rf'<{tag}>(.*?)</{tag}>', text or '', re.S)
    return [ln.strip().strip('`').strip() for ln in m.group(1).splitlines() if ln.strip()] if m else []


def run_pool(fn, jobs, workers):
    """PL.pool plus a progress file OUT/_progress/<pid>_<n>.json for the progress page (progress_server.py)."""
    import functools
    import os
    import threading
    import time
    from collections import Counter
    run_pool.n = getattr(run_pool, 'n', 0) + 1
    path = OUT / '_progress' / f'{os.getpid()}_{run_pool.n}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    label = ' '.join([Path(sys.argv[0]).name] + sys.argv[1:]) or fn.__name__
    st = {'label': label, 'fn': fn.__name__, 'pid': os.getpid(), 'of': len(jobs), 'done': 0, 'workers': workers,
          'statuses': Counter(), 'started': time.time(), 'updated': time.time(), 'finished': None}
    lock = threading.Lock()
    last = [0.0]

    def dump(force=False):
        now = time.time()
        if force or now - last[0] > 2:
            last[0] = now
            st['updated'] = now
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(st))
            tmp.replace(path)

    @functools.wraps(fn)
    def wrapped(*a, **k):
        try:
            r = fn(*a, **k)
        except Exception:
            r = 'error: exception'
            raise
        finally:
            with lock:
                st['done'] += 1
                st['statuses'][str(r).split(':')[0][:20]] += 1
                dump()
        return r

    dump(True)
    try:
        PL.pool(wrapped, jobs, workers)
    finally:
        st['finished'] = time.time()
        dump(True)


# ------------------------------------------------------------------ scoring
def species_keys(smiles):
    return {k for k, _ in P.smiles_keys(smiles)} if smiles else set()


def score(case, steps):
    """Metrics for one predicted mechanism (None = unparseable, scored as zero)."""
    steps = steps or []
    smis = [s.get('intermediate_smiles') or '' for s in steps]
    if case['src'] in ('J3', 'T3_dev', 'T3_confirm'):
        s = SM.score(case['pgl'], smis) if smis else SM.score(case['pgl'], [''])
        ev = list(s['heavy_events'].values())
        return {'parsed': bool(steps), 'all_present': float(s['all_present']) if steps else 0.0,
                'in_order_frac': s['in_order_frac'] if steps else 0.0,
                'heavy_hit': sum(ev) if steps else 0, 'heavy_n': len(ev), 'n_hidden_events': s['n_hidden_events'],
                'hid_hit': s['hid_hit'] if steps else 0, 'hid_n': s['hid_n']}
    if case['src'] == 'uspto31k':
        return score_states(case, smis)
    gold = [(g['subtype'], g['intermediate_smiles'], float(g.get('step_weight') or 0)) for g in case['gold']]
    if gold and not any(w for _, _, w in gold):  # oMe-Silver/Template carry no step weights: uniform 1/N
        gold = [(t, smi, 1.0 / len(gold)) for t, smi, _ in gold]
    pred = [(str(s.get('subtype', '')), s.get('intermediate_smiles') or '') for s in steps]
    try:
        r = oMeS(gold, pred)
        off = {'S_total': r.S_total, 'S_partial': r.S_partial, 'V': r.V, 'L': r.L}
    except Exception:
        off = {'S_total': 0.0, 'S_partial': 0.0, 'V': 0.0, 'L': 0.0}
    gk = [k for g in case['gold'] if (k := j4_score.keys_of_smiles(g['intermediate_smiles']))]
    pk = [j4_score.keys_of_smiles(x) or set() if x else set() for x in smis]
    hit = cur = 0
    for keys in gk:
        f = next((i for i in range(cur, len(pk)) if keys <= pk[i]), None)
        if f is not None:
            hit, cur = hit + 1, f + 1
    allp = set().union(*pk) if pk else set()
    return {'parsed': bool(steps), **off, 'in_order_full': float(bool(gk) and hit == len(gk) and bool(steps)),
            'in_order_frac': hit / len(gk) if gk else 0.0,
            'all_present': float(bool(steps) and all(k <= allp for k in gk))}


def score_states(case, smis):
    """Species-level scoring against a GT state sequence (mech-USPTO-31k): new species of every
    intermediate state, in order, and all present."""
    start = {k for x in case['reactants'] for k in (j4_score.keys_of_smiles(x) or set())}
    final = j4_score.keys_of_smiles(case['states'][-1]) or set()
    gk = [g for st in case['states'][:-1] if (g := (j4_score.keys_of_smiles(st) or set()) - start - final)]
    pk = [j4_score.keys_of_smiles(x) or set() if x else set() for x in smis]
    hit = cur = 0
    for keys in gk:
        f = next((i for i in range(cur, len(pk)) if keys <= pk[i]), None)
        if f is not None:
            hit, cur = hit + 1, f + 1
    allp = set().union(*pk) if pk else set()
    return {'parsed': bool(smis), 'in_order_full': float(bool(smis) and hit == len(gk)),
            'in_order_frac': hit / len(gk) if gk else 1.0, 'all_present': float(bool(smis) and all(k <= allp for k in gk))}


def load_uspto():
    return {c['id']: {**c, 'src': 'uspto31k', 'name': '', 'conditions': '', 'level': None}
            for c in json.loads((OUT / 'spec_bench_uspto.json').read_text())}


def summarize(rows, keys):
    """rows: list of (case_id, metrics); case-clustered bootstrap mean [lo, hi] per key."""
    ids = [c for c, _ in rows]
    out = {'n': len(rows)}
    for k in keys:
        vals = [float(m.get(k) or 0) for _, m in rows]
        out[k] = phase2.ci(vals, ids) if vals else None
    if rows and 'heavy_n' in rows[0][1]:
        ev = [(c, 1.0) for c, m in rows for _ in range(m['heavy_hit'])] + \
             [(c, 0.0) for c, m in rows for _ in range(m['heavy_n'] - m['heavy_hit'])]
        out['heavy_hidden_recall'] = phase2.ci([v for _, v in ev], [c for c, _ in ev]) if ev else None
    return out


def paired(rows_a, rows_b, key):
    """Mean difference a - b over shared cases with a case bootstrap CI."""
    b = dict(rows_b)
    shared = [(c, float(m.get(key) or 0) - float(b[c].get(key) or 0)) for c, m in rows_a if c in b]
    return phase2.ci([d for _, d in shared], [c for c, _ in shared]) if shared else None
