"""P44 (2026-11-08): single-step prediction given the reference previous state, for the LLM and the specialists.

Every reference elementary step S_{k-1} -> S_k becomes one item: the model sees the complete reference state S_{k-1}
(all species, atom-balanced) and predicts the state after one elementary step. A step counts as correct when every
species that is new in S_k is present in the predicted state (strict: sigma-graph + H keys; chemically equivalent:
standard-InChIKey first block). Steps that change only bond orders or charges leave the sigma graph unchanged
and are excluded. The LLM is told all overall products, ChRIMP its usual main-product goal, FlowER nothing. Specialists are scored at top-1 and top-5 (their own ranked outputs); the LLM gives
one answer.

Sets
  fuku       FukuyamaBench (dev + test, 95 reactions): complete PGL states, out of the specialists' template space
  uspto      200 template-expanded mech-USPTO reactions (in template)
  flowertest 800 random steps of FlowER's own test split (seed 20261108; no final product available)
oMe-Gold is not used: it lists key intermediates only, not atom-balanced states, so FlowER cannot be fed fairly.

python p44_stepwise.py build
python p44_stepwise.py llm [--workers 100]          (FALLBACK_API_KEY in env; 117 gateway, gemini-3.8-flash)
python p44_stepwise.py score                        (reads llm cache + specialist preds_<kind>.jsonl)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

import common4 as K

RDLogger.DisableLog('rdApp.*')
RUN = K.OUT / 'stepwise_20261108'
STEPS = RUN / 'steps.json'
TAG = 'stepwise_llm'
FLOWER_TEST = Path(__file__).resolve().parents[3] / 'external' / 'FlowER' / 'data' / 'flower_new_dataset' / 'test.txt'


def unmap(smi):
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return smi
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    try:
        m = Chem.RemoveHs(m)
    except Exception:
        pass
    return Chem.MolToSmiles(m)


def main_product(products, reactants):
    start = {Chem.MolToSmiles(m) for s in reactants if (m := Chem.MolFromSmiles(s))}
    new = [s for s in products if (m := Chem.MolFromSmiles(s)) and Chem.MolToSmiles(m) not in start]
    return max(new or products, key=lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms() if Chem.MolFromSmiles(s) else 0)


def state_species(state):
    return [K.C._molecule_smiles(state, comp) for comp in K.C._components(state)]


def build():
    items = []
    for which in ('T3_dev', 'T3_confirm'):
        for cid, c in sorted(K.load_fukuyama(which).items()):
            pgl = c['pgl']
            states = [K.chem.parse_pgl_literal(pgl['initial_state_pgl'])] + K.C.gt_states(pgl)
            sp = [state_species(s) for s in states]
            prod = main_product(c['products'], c['reactants'])
            for k in range(1, len(sp)):
                items.append({'sid': f'fuku__{cid}__{k}', 'set': 'fuku', 'split': which, 'cid': cid, 'k': k,
                              'prev': sp[k - 1], 'next': sp[k], 'product': prod, 'products': c['products'],
                              'conditions': c.get('conditions', '')})
    for cid, c in sorted(K.load_uspto().items()):
        sp = [list(c['reactants'])] + [s.split('.') for s in c['states']]
        prod = main_product(c['products'], c['reactants'])
        for k in range(1, len(sp)):
            items.append({'sid': f'uspto__{cid}__{k}', 'set': 'uspto', 'cid': cid, 'k': k,
                          'prev': sp[k - 1], 'next': sp[k], 'product': prod, 'products': c['products'],
                          'conditions': c.get('conditions', '')})
    lines = FLOWER_TEST.read_text().splitlines()
    for i in sorted(random.Random(20261108).sample(range(len(lines)), 800)):
        rxn = lines[i].split('|')[0]
        reac, prod = rxn.split('>>')
        items.append({'sid': f'flowertest__{i}', 'set': 'flowertest', 'cid': str(i), 'k': 0,
                      'prev': unmap(reac).split('.'), 'next': unmap(prod).split('.'), 'prev_mapped': reac,
                      'product': '', 'products': [], 'conditions': ''})
    keep = [it for it in items if targets(it)[0]]
    RUN.mkdir(parents=True, exist_ok=True)
    STEPS.write_text(json.dumps(keep, indent=0))
    from collections import Counter
    print({'steps': len(keep), 'dropped_no_new_species': len(items) - len(keep),
           'by_set': Counter(i['set'] for i in keep)})


# ------------------------------------------------------------------ scoring keys
def ekeys(smi):
    out = set()
    for s in (smi or '').split('.'):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            out.add(Chem.MolToInchiKey(m).split('-')[0])
        except Exception:
            pass
    return out


def keyset(species, fn):
    out = set()
    for s in species:
        out |= fn(s)
    return out


def targets(it):
    """(strict new keys, equivalent new keys) of S_k relative to S_{k-1}."""
    ts = keyset(it['next'], K.species_keys) - keyset(it['prev'], K.species_keys)
    te = keyset(it['next'], ekeys) - keyset(it['prev'], ekeys)
    return ts, te


def hit(it, species):
    ts, te = targets(it)
    if not species:
        return 0.0, 0.0
    st = float(ts <= keyset(species, K.species_keys))
    eq = float(bool(te) and te <= keyset(species, ekeys))
    return st, max(st, eq)  # equivalent matching only relaxes strict matching (same rule as Table 1)


# ------------------------------------------------------------------ LLM
PROMPT = """{context}Current reaction mixture (every species present after the preceding elementary steps), as SMILES:
{state}

Predict the single next elementary step from this mixture (one arrow-pushing event: a bond formation, a bond cleavage, a proton transfer, or one concerted step).
Return every species present AFTER this one step, including unchanged spectators and counter-ions, as SMILES, one per line, inside <state></state>. Do not skip ahead to later steps."""


def prompt(it):
    ctx = ''
    if it['conditions']:
        ctx += f"Conditions: {it['conditions']}\n"
    if it['products']:
        ctx += f"Overall the reaction gives: {'.'.join(it['products'])}\n"
    ctx = ctx + '\n' if ctx else ''
    return PROMPT.format(context=ctx, state='\n'.join(it['prev']))


def parse_state(text):
    m = re.findall(r'<state>(.*?)</state>', text or '', re.S)
    if not m:
        return []
    out = []
    for line in m[-1].splitlines():
        line = line.strip().strip('-*` ').strip()
        out += [x for x in line.split('.') if x and Chem.MolFromSmiles(x) is not None]
    return out


def job_llm(it):
    return 'ok' if K.call(TAG, it['sid'], prompt(it)) else 'error'


def llm(workers):
    items = json.loads(STEPS.read_text())
    todo = [it for it in items if K.cached(TAG, it['sid']) is None]
    print(len(items), 'steps,', len(todo), 'to run', flush=True)
    K.run_pool(job_llm, [(it,) for it in todo], workers)


# ------------------------------------------------------------------ summary
def score():
    from collections import defaultdict
    items = json.loads(STEPS.read_text())
    preds = {}
    for kind in ('flower', 'chrimp'):
        f = RUN / f'preds_{kind}.jsonl'
        if f.exists():
            preds[kind] = {r['sid']: r for r in map(json.loads, f.read_text().splitlines())}
    rows = defaultdict(lambda: defaultdict(list))
    for it in items:
        s = it['set']
        ans = K.cached(TAG, it['sid'])
        if ans is not None:
            st, eq = hit(it, parse_state(ans))
            rows[s]['LLM@1 strict'].append(st)
            rows[s]['LLM@1 equiv'].append(eq)
        for kind, P in preds.items():
            r = P.get(it['sid'])
            if r is None or 'OutOfMemory' in r.get('error', '') or 'CUDA' in r.get('error', ''):
                continue  # GPU failure = missing coverage, not a model miss
            cands = [unmap(x).split('.') for x in r.get('preds', [])]
            h = [hit(it, c) for c in cands]
            for k in (1, 5):
                rows[s][f'{kind}@{k} strict'].append(max([x[0] for x in h[:k]], default=0.0))
                rows[s][f'{kind}@{k} equiv'].append(max([x[1] for x in h[:k]], default=0.0))
    out = {s: {m: {'n': len(v), 'acc': round(100 * sum(v) / len(v), 1)} for m, v in sorted(d.items())}
           for s, d in rows.items()}
    (RUN / 'SUMMARY_STEPWISE.json').write_text(json.dumps(out, indent=1))
    for s, d in out.items():
        print('==', s)
        for m, v in d.items():
            print(f'   {m:22s} {v["acc"]:5.1f}  (n={v["n"]})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'llm', 'score'])
    ap.add_argument('--workers', type=int, default=100)
    a = ap.parse_args()
    {'build': build, 'llm': lambda: llm(a.workers), 'score': score}[a.cmd]()
