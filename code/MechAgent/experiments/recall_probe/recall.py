"""Candidate recall for the next mechanistic step (direction two, after the scale probe).

Same 115 GT-prefix steps of the 24 dev cases as experiments/scale_probe (items.json).
For each step: does a candidate source contain the GT next state, and how large is it?
  enum     all states reachable from the current GT state in <=2 two-electron moves,
           plus a third move from up to 300 of the depth-2 states (lowest total charge
           first); recall@N after ranking by target edit distance or by charge burden.
  sample   the model proposes the next step, 5 independent samples at temperature 1.
  list     one call: the model lists its 5 most plausible distinct next steps.
A proposal is written as bond/charge changes on the current state (the format used to
show steps elsewhere) and applied here; a hit is a state equal to the GT next state
(fukuyama_metrics.same_state); reaching a later GT state is counted separately.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
import json
import multiprocessing
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'agent_v2'))
sys.path.insert(0, str(HERE.parent / 'step_search'))
sys.path.insert(0, str(HERE.parent / 'scale_probe'))
import agent_v2 as A
import common as C
import moves as MV
import probe as P
from common import ROOT, chem, fm
from develop import metrics

OUT = ROOT / 'outputs' / 'recall_probe_20261101'
ITEMS = ROOT / 'outputs' / 'scale_probe_20261031' / 'items.json'
BASE_URL = 'https://LLM-GATEWAY-2.example/v1'
MODEL = 'gemini-3.8-flash-high'
K = 5
ORDER_KIND = {1.0: 'single', 1.5: 'aromatic', 2.0: 'double', 3.0: 'triple'}
FORMAT = ('Write a step as its changes on the current state, separated by semicolons, using the atom labels of the states: '
          '"form A-B" (new single bond), "break A-B", "A-B order 2->1", "charge A +0->+1". A step may be concerted '
          '(several changes at once); give the complete change from the current intermediate to the next intermediate.')


def pool_cases():
    return {c['case_id']: c for c in map(json.loads, (ROOT / 'outputs/fukuyama/clean.jsonl').read_text().splitlines())}


def gt_chain(case):
    return [chem.parse_pgl_literal(case['initial_state_pgl'])] + C.gt_states(case)


def view(state):
    return fm.state_view(state)


# ------------------------------------------------------------------ apply proposals
def apply_changes(state, text):
    """Apply 'form A-B; break C-D; A-B order 2->1; charge X +0->+1' to a copy of state; None if invalid."""
    state = copy.deepcopy(state)
    atoms = dict(chem.pgl.iter_atoms(state))
    try:
        for part in [p.strip().strip('.') for p in re.split(r'[;\n]', text) if p.strip()]:
            locations = chem._bond_locations(state)
            m = re.fullmatch(r'form\s+(\S+?)-(\S+?)(?:\s+order\s+([\d.]+))?', part)
            if m:
                a, b = m.group(1), m.group(2)
                if a not in atoms or b not in atoms or chem._bond_key(a, b) in locations:
                    return None
                order = float(m.group(3) or 1)
                chem._find_substructure(state, a).bondlist.append(chem.Bond(a, b, ORDER_KIND.get(order, 'single'), order=order))
                continue
            m = re.fullmatch(r'break\s+(\S+?)-(\S+)', part)
            if m:
                sub, bond = locations[chem._bond_key(m.group(1), m.group(2))]
                sub.bondlist.remove(bond)
                continue
            m = re.fullmatch(r'(\S+?)-(\S+?)\s+order\s+([\d.]+)\s*->\s*([\d.]+)', part)
            if m:
                _, bond = locations[chem._bond_key(m.group(1), m.group(2))]
                new = float(m.group(4))
                if new == 0:
                    sub, bond = locations[chem._bond_key(m.group(1), m.group(2))]
                    sub.bondlist.remove(bond)
                else:
                    bond.order, bond.kind = new, ORDER_KIND.get(new, 'single')
                continue
            m = re.fullmatch(r'charge\s+(\S+)\s+([+-]?\d+)\s*->\s*([+-]?\d+)', part)
            if m:
                atoms[m.group(1)].charge = int(m.group(3))
                continue
            return None
    except (KeyError, AttributeError, ValueError):
        return None
    return state


def classify(new, gt, pos):
    """'next' if new equals GT state pos+1, 'later' if a later GT state, else 'miss'."""
    if new is None:
        return 'invalid'
    v = view(new)
    if fm.same_state(v, view(gt[pos + 1])):
        return 'next'
    if any(fm.same_state(v, view(g)) for g in gt[pos + 2:]):
        return 'later'
    return 'miss'


# ------------------------------------------------------------------ enumeration
def enum_one(args):
    case, item = args
    cur = chem.parse_pgl_literal(item['current'])
    gt = gt_chain(case)
    pos = item['step']
    nxt = gt[pos + 1]
    tgt = chem.parse_pgl_literal(case['target_state_pgl'])
    reg = MV.region(case, gt[0])
    goal = view(nxt)
    seen = {MV.signature(cur)}
    found, rows = None, []

    def visit(states, depth):
        nonlocal found
        fresh = []
        for st in states:
            sig = MV.signature(st)
            if sig in seen:
                continue
            seen.add(sig)
            hit = fm.same_state(view(st), goal)
            if hit and found is None:
                found = depth
            b = P.burden(st)
            rows.append({'depth': depth, 'valid': b is not None, 'burden': b if b is not None else 99,
                         'tdist': P.target_distance(st, tgt), 'hit': hit})
            fresh.append(st)
        return fresh

    d1 = visit([s for _, _, s in MV.enumerate_moves(case, cur, reg | MV.region(case, cur))], 1)
    d2 = visit([s for st in d1 for _, _, s in MV.enumerate_moves(case, st, reg | MV.region(case, st))], 2)
    charge = lambda st: sum(abs(int(a.charge or 0)) for _, a in chem.pgl.iter_atoms(st))
    seeds = sorted(d2, key=charge)[:300]
    visit([s for st in seeds for _, _, s in MV.enumerate_moves(case, st, reg | MV.region(case, st))], 3)
    return {'case_id': item['case_id'], 'step': pos, 'found_depth': found, 'rows': rows}


def run_enum(items, cases, procs):
    dest = OUT / 'enum.jsonl'
    done = {(r['case_id'], r['step']) for r in map(json.loads, dest.read_text().splitlines())} if dest.exists() else set()
    todo = [(cases[it['case_id']], it) for it in items if (it['case_id'], it['step']) not in done]
    with multiprocessing.Pool(procs) as pool, dest.open('a') as fh:
        for k, rec in enumerate(pool.imap_unordered(enum_one, todo), 1):
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            print(json.dumps({'enum_done': k, 'of': len(todo), 'found_depth': rec['found_depth'], 'n': len(rec['rows'])}), flush=True)


# ------------------------------------------------------------------ LLM proposals
def context(case, item):
    cur = chem.parse_pgl_literal(item['current'])
    hist = '\n'.join(f'  {k + 1}. {h}' for k, h in enumerate(item['history'])) or '  (none yet)'
    return (f"INITIAL:\n{case['initial_state_pgl']}\nTARGET:\n{case['target_state_pgl']}\n"
            f"Conditions: {A.conditions_text(case)}\nSteps of the mechanism so far (changes on the INITIAL state, applied in order):\n{hist}\n"
            f"CURRENT species: {', '.join(C._molecule_smiles(cur, m) for m in C._components(cur))}\n\n{FORMAT}")


SAMPLE_ASK = ('\n\nWhat is the next step of the accepted mechanism under these conditions? Reason briefly, then give the step '
              'as <step>changes</step>.')
LIST_ASK = (f'\n\nList the {K} most plausible DISTINCT next steps under these conditions, most likely first (include '
            'alternatives a chemist would seriously consider, e.g. different sites, protonation first, concerted vs stepwise). '
            f'Reason briefly, then give them as <step1>changes</step1> ... <step{K}>changes</step{K}>.')


def llm_job(arm, r, case, item):
    dest = OUT / MODEL / arm / f"{item['case_id']}_{item['step']}_{r}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    model = C.Model(MODEL, OUT / MODEL / arm / 'traces' / f"{item['case_id']}_{item['step']}_{r}.jsonl", base_url=BASE_URL,
                    max_tokens=8192, temperature=1.0 if arm == 'sample' else 0.2)
    try:
        answer = model.complete([{'role': 'system', 'content': P.SYSTEM},
                                 {'role': 'user', 'content': context(case, item) + (SAMPLE_ASK if arm == 'sample' else LIST_ASK)}], arm)
    except Exception as exc:
        return {'error': type(exc).__name__}
    if arm == 'sample':
        steps = re.findall(r'<step>(.*?)</step>', answer, re.S)[-1:]
    else:
        steps = [m[-1] for k in range(1, K + 1) if (m := re.findall(rf'<step{k}>(.*?)</step{k}>', answer, re.S))]
    gt = gt_chain(case)
    cur = chem.parse_pgl_literal(item['current'])
    outcomes = []
    for text in steps:
        new = apply_changes(cur, text.strip())
        outcomes.append({'text': text.strip(), 'class': classify(new, gt, item['step']),
                         'sig': repr(MV.signature(new)) if new is not None else None})
    rec = {'case_id': item['case_id'], 'step': item['step'], 'arm': arm, 'r': r, 'proposals': outcomes, 'tokens': model.tokens}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    return rec


# ------------------------------------------------------------------ analysis
def analyze(items):
    ids = [(it['case_id'], it['step']) for it in items]
    enum = {(r['case_id'], r['step']): r for r in map(json.loads, (OUT / 'enum.jsonl').read_text().splitlines())} \
        if (OUT / 'enum.jsonl').exists() else {}
    report = {'steps': len(ids)}
    if enum:
        e = [enum[i] for i in ids if i in enum]
        block = {'steps': len(e), 'set_size_median': statistics.median(len(r['rows']) for r in e),
                 'set_size_valid_median': statistics.median(sum(x['valid'] for x in r['rows']) for r in e)}
        for d in (1, 2, 3):
            block[f'recall_depth<={d}'] = round(statistics.mean(r['found_depth'] is not None and r['found_depth'] <= d for r in e), 4)
        block['gt_passes_validity_filter'] = round(statistics.mean(any(x['hit'] and x['valid'] for x in r['rows']) for r in e), 4)
        for key in ('tdist', 'burden'):
            for n in (5, 10, 20, 50):
                hits = []
                for r in e:
                    rows = sorted([x for x in r['rows'] if x['valid']], key=lambda x: (x[key], x['tdist'], x['burden']))
                    hits.append(any(x['hit'] for x in rows[:n]))
                block[f'recall@{n}_rank_by_{key}'] = round(statistics.mean(hits), 4)
        report['enum'] = block
    for arm, reps in (('sample', range(K)), ('list', [0])):
        recs = {}
        for i in ids:
            fs = [OUT / MODEL / arm / f'{i[0]}_{i[1]}_{r}.json' for r in reps]
            if all(f.exists() for f in fs):
                recs[i] = [p for f in fs for p in json.loads(f.read_text())['proposals']]
        if not recs:
            continue
        block = {'steps': len(recs)}
        cls = collections.Counter(p['class'] for v in recs.values() for p in v)
        block['proposal_classes'] = dict(cls)
        for n in ((1, 3, 5) if True else ()):
            block[f'recall_next@{n}'] = round(statistics.mean(any(p['class'] == 'next' for p in v[:n]) for v in recs.values()), 4)
            block[f'recall_next_or_later@{n}'] = round(statistics.mean(any(p['class'] in ('next', 'later') for p in v[:n])
                                                                       for v in recs.values()), 4)
        block['distinct_valid_states_median'] = statistics.median(len({p['sig'] for p in v if p['sig']}) for v in recs.values())
        report[arm] = block
    if enum and all((OUT / MODEL / 'list' / f'{i[0]}_{i[1]}_0.json').exists() for i in ids):
        for n in (5, 10, 20):
            hits = []
            for i in ids:
                lst = json.loads((OUT / MODEL / 'list' / f'{i[0]}_{i[1]}_0.json').read_text())['proposals']
                samp = [p for r in range(K) if (OUT / MODEL / 'sample' / f'{i[0]}_{i[1]}_{r}.json').exists()
                        for p in json.loads((OUT / MODEL / 'sample' / f'{i[0]}_{i[1]}_{r}.json').read_text())['proposals']]
                rows = sorted([x for x in enum[i]['rows'] if x['valid']], key=lambda x: (x['tdist'], x['burden']))[:n]
                hits.append(any(p['class'] == 'next' for p in lst + samp) or any(x['hit'] for x in rows))
            report[f'union_list5+sample5+enum_top{n}_recall'] = round(statistics.mean(hits), 4)
    (OUT / 'SUMMARY.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['enum', 'llm', 'analyze'])
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--workers', type=int, default=110)
    cli = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = [it for it in json.loads(ITEMS.read_text()) if len(it['candidates']) == P.N_CAND]
    cases = pool_cases()
    if cli.what == 'enum':
        run_enum(items, cases, cli.procs)
    elif cli.what == 'llm':
        jobs = [(arm, r, cases[it['case_id']], it) for it in items for arm, reps in (('sample', range(K)), ('list', [0])) for r in reps]
        random.Random(11).shuffle(jobs)
        with concurrent.futures.ThreadPoolExecutor(cli.workers) as ex:
            for k, fut in enumerate(concurrent.futures.as_completed([ex.submit(llm_job, *j) for j in jobs]), 1):
                rec = fut.result()
                if 'error' in rec or k % 50 == 0:
                    print(json.dumps({'done': k, 'of': len(jobs), **({'error': rec['error']} if 'error' in rec else {})}), flush=True)
    analyze(items)


if __name__ == '__main__':
    main()
