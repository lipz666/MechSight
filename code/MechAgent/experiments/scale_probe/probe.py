"""Quick probe of the measurement-grounded paradigm (LLM-estimated parameters only).

Per annotated step of the 24 dev cases (GT state i -> GT state i+1), with the
GT prefix given, the model sees 8 candidate next steps: the GT step plus 7
legal successors of state i (<=2 two-electron moves) chosen as hard negatives: no
|charge| >= 2, no hydride, no neutral carbene/nitrene, lowest charge separation first; up to 4 of them touch an
atom the GT step touches. Shuffled; each shown as bond/charge changes + species.
Arms on identical candidates:
  direct  the model picks the step of the accepted mechanism.
  scale   the model only estimates physical-organic parameters for the polar
          events of every candidate (Mayr N/sN/E, pKa, nucleofugality Nf/sf with
          electrofugality Ef, or an activation free energy for other events);
          code turns them into log k (Mayr-Patz, Eigen, SN1 heterolysis, Eyring)
          and picks the candidate whose slowest event is fastest.
No experimental database is used yet: this checks whether routing the decision
through quantitative reactivity estimates beats direct choice at all.

--items matched: size-matched control. Negatives come from <=3 moves and must have
the same number of changes as the GT step (+-1), so step size cannot give the GT
away; steps without 7 such negatives are dropped. Outputs go to OUT/matched/.
--items hard: as matched, but the 7 negatives are the ones closest to the TARGET
(bond-order + charge edit distance), so "move toward the target" no longer singles
out the GT step. Outputs go to OUT/hard/; the distance heuristic is scored too.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'agent_v2'))
sys.path.insert(0, str(HERE.parent / 'step_search'))
import agent_v2 as A
import common as C
import moves as MV
from common import ROOT, chem, fm
from develop import metrics

OUT = ROOT / 'outputs' / 'scale_probe_20261031'
BASE_URL = 'http://127.0.0.1:2048/v1'
N_CAND = 8
SYSTEM = 'You are an expert physical organic chemist.'


def same(a, b):
    return fm.same_state(fm.state_view(a), fm.state_view(b))


def touched(before, after):
    changes, _ = C.step_text(before, after)
    out = set()
    for part in changes.split('; '):
        tok = part.replace('form ', '').replace('break ', '').replace('charge ', '').split(' ')[0]
        out |= set(tok.split('-'))
    return out - {''}


def burden(state):
    """Total |formal charge|, or None for implausible states (|q| >= 2, hydride, neutral carbene/nitrene)."""
    atoms, order, adj = MV.view(state)
    total = 0
    for name, a in atoms.items():
        q = int(a.charge or 0)
        if abs(q) >= 2 or (a.element == 'H' and q < 0):
            return None
        if q == 0 and a.element in {'C', 'N', 'O'} and MV.deficient(name, atoms, order, adj):
            return None
        total += abs(q)
    return total


def target_distance(state, target):
    sa, sb = chem.state_signature(state), chem.state_signature(target)
    ba = {k: v['order'] for k, v in sa['bonds'].items()}
    bb = {k: v['order'] for k, v in sb['bonds'].items()}
    d = sum(abs(ba.get(k, 0) - bb.get(k, 0)) for k in set(ba) | set(bb))
    return d + sum(abs(sa['atoms'][n]['charge'] - sb['atoms'][n]['charge']) for n in set(sa['atoms']) & set(sb['atoms']))


def n_changes(before, after):
    return len(C.step_text(before, after)[0].split('; '))


def build_items(matched=False, hard=False):
    pool = {c['case_id']: c for c in map(json.loads, (ROOT / 'outputs/fukuyama/clean.jsonl').read_text().splitlines())}
    items = []
    for cid in json.loads((ROOT / 'outputs/p23_planning_probe_20261024/cases.json').read_text()):
        case = pool[cid]
        ini = chem.parse_pgl_literal(case['initial_state_pgl'])
        gt = [ini] + C.gt_states(case)
        reg = MV.region(case, ini)
        tgt = chem.parse_pgl_literal(case['target_state_pgl'])
        for i in range(len(gt) - 1):
            cur, nxt = gt[i], gt[i + 1]
            if not C.step_text(cur, nxt)[0]:
                continue
            succ, seen = [], {MV.signature(cur), MV.signature(nxt)}
            first = MV.enumerate_moves(case, cur, reg | MV.region(case, cur))
            layer = [s for _, _, s in first]
            for st in layer[:40]:
                layer += [s for _, _, s in MV.enumerate_moves(case, st, reg | MV.region(case, st))]
            rng = random.Random(f'{cid}:{i}')
            if matched:
                for st in rng.sample(layer, min(40, len(layer))):
                    layer += [s for _, _, s in MV.enumerate_moves(case, st, reg | MV.region(case, st))]
            for st in layer:
                sig = MV.signature(st)
                if sig in seen or same(st, nxt) or same(st, cur):
                    continue
                seen.add(sig)
                succ.append(st)
            core = touched(cur, nxt)
            gt_n = n_changes(cur, nxt)
            succ = [s for s in succ if burden(s) is not None]
            if matched:
                succ = [s for s in succ if abs(n_changes(cur, s) - gt_n) <= 1]
                if len(succ) < N_CAND - 1:
                    continue
            rng.shuffle(succ)
            succ.sort(key=burden)
            if hard:
                succ.sort(key=lambda st: target_distance(st, tgt))
                pick = succ[:N_CAND - 1]
            else:
                near = [s for s in succ if touched(cur, s) & core]
                pick = near[:4]
                pick += [s for s in succ if s not in pick][:N_CAND - 1 - len(pick)]
            cands = [nxt] + pick
            order = list(range(len(cands)))
            rng.shuffle(order)
            items.append({'case_id': cid, 'step': i, 'gt_index': order.index(0), 'n_changes': gt_n,
                          'cand_changes': [n_changes(cur, cands[k]) for k in order],
                          'cand_target_distance': [target_distance(cands[k], tgt) for k in order],
                          'current': repr(cur), 'candidates': [repr(cands[k]) for k in order],
                          'history': [C.step_text(gt[k], gt[k + 1])[0] for k in range(i)]})
    return items, pool


def context(case, item):
    cur = chem.parse_pgl_literal(item['current'])
    hist = '\n'.join(f'  {k + 1}. {h}' for k, h in enumerate(item['history'])) or '  (none yet)'
    lines = []
    for k, c in enumerate(item['candidates']):
        changes, species = C.step_text(cur, chem.parse_pgl_literal(c))
        lines.append(f'[{k}] {changes}  ->  {species}')
    return (f"INITIAL:\n{case['initial_state_pgl']}\nTARGET:\n{case['target_state_pgl']}\n"
            f"Conditions: {A.conditions_text(case)}\nSteps of the mechanism so far:\n{hist}\n"
            f"CURRENT species: {', '.join(C._molecule_smiles(cur, m) for m in C._components(cur))}\n\n"
            'Candidate next steps (atom labels as in the states above; a candidate may combine several arrows):\n' + '\n'.join(lines))


DIRECT_ASK = ('\n\nWhich candidate is the next step of the accepted mechanism under these conditions? Reason briefly, '
              'then answer with exactly <choice>k</choice>.')
SCALE_ASK = '''

Do NOT decide which candidate is the textbook answer. Instead, act as a measuring physical organic chemist: for EVERY
candidate, decompose it into its polar elementary events and estimate quantitative parameters for each event by analogy
with the closest compounds measured in Mayr's reactivity database, Bordwell/iBonD pKa tables and Mayr's nucleofuge /
electrofuge scales (name the reference compound briefly), in the solvent of the conditions. Event types:
  attack           nucleophile site + electrophile site form a bond: give N, sN (nucleophile) and E (electrophile)
  proton_transfer  give pKa_acid (the acid that loses H+) and pKa_base_conj (pKa of the conjugate acid of the base), same solvent
  ionization       heterolysis R-X -> R+ + X-: give Nf, sf (nucleofuge X-) and Ef (electrofuge R+)
  other            anything else (pericyclic, 1,2-shift, concerted, metal-centred): give dG_act_kcal (activation free energy)
Reason briefly, then output exactly one JSON object between <json> and </json>:
{"candidates": [{"id": 0, "events": [{"type": "attack", "N": 13.4, "sN": 0.7, "E": -5.5}, ...]}, ...]}'''


def log_k(ev):
    t = ev.get('type')
    try:
        if t == 'attack':
            return float(ev['sN']) * (float(ev['N']) + float(ev['E']))
        if t == 'proton_transfer':
            d = float(ev['pKa_base_conj']) - float(ev['pKa_acid'])
            return 10.0 + min(0.0, d)
        if t == 'ionization':
            return float(ev['sf']) * (float(ev['Nf']) + float(ev['Ef']))
        if t == 'other':
            return 12.79 - float(ev['dG_act_kcal']) / 1.364
    except (KeyError, TypeError, ValueError):
        return None
    return None


def scale_pick(answer, n):
    found = re.findall(r'<json>(.*?)</json>', answer, re.S)
    if not found:
        return None, {}
    try:
        data = json.loads(found[-1])
    except json.JSONDecodeError:
        return None, {}
    scores = {}
    for c in data.get('candidates', []):
        ks = [log_k(e) for e in c.get('events', [])]
        if c.get('id') in range(n) and ks and all(k is not None for k in ks):
            scores[c['id']] = min(ks)
    if not scores:
        return None, scores
    return max(scores, key=scores.get), scores


def job(model_name, arm, case, item, out, base_url):
    dest = out / model_name / arm / f"{item['case_id']}_{item['step']}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    model = C.Model(model_name, out / model_name / arm / 'traces' / f"{item['case_id']}_{item['step']}.jsonl",
                    base_url=base_url, max_tokens=8192)
    try:
        answer = model.complete([{'role': 'system', 'content': SYSTEM},
                                 {'role': 'user', 'content': context(case, item) + (DIRECT_ASK if arm == 'direct' else SCALE_ASK)}], arm)
    except Exception as exc:
        return {'error': type(exc).__name__}
    n = len(item['candidates'])
    if arm == 'direct':
        m = re.findall(r'<choice>\s*\[?\s*(\d+)\s*\]?\s*</choice>', answer)
        choice, scores = (int(m[-1]) if m and int(m[-1]) < n else None), {}
    else:
        choice, scores = scale_pick(answer, n)
    rec = {'case_id': item['case_id'], 'step': item['step'], 'arm': arm, 'choice': choice, 'gt_index': item['gt_index'],
           'correct': choice == item['gt_index'], 'scores': scores, 'tokens': model.tokens}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    return rec


def analyze(items, models, out, arms):
    report = {'steps': len(items), 'candidates_per_step': statistics.mean(len(i['candidates']) for i in items)}
    if all('cand_changes' in it for it in items):
        report['mean_changes_gt'] = round(statistics.mean(it['n_changes'] for it in items), 2)
        report['mean_changes_negatives'] = round(statistics.mean(c for it in items for k, c in enumerate(it['cand_changes'])
                                                                 if k != it['gt_index']), 2)
    if all('cand_target_distance' in it for it in items):
        acc = []
        for it in items:
            d = it['cand_target_distance']
            winners = [k for k, x in enumerate(d) if x == min(d)]
            acc.append((it['gt_index'] in winners) / len(winners))
        report['target_distance_heuristic_top1'] = round(statistics.mean(acc), 4)
    for m in models:
        rows = []
        for it in items:
            recs = {arm: out / m / arm / f"{it['case_id']}_{it['step']}.json" for arm in arms}
            if all(p.exists() for p in recs.values()):
                r = {arm: json.loads(p.read_text()) for arm, p in recs.items()}
                rows.append((it, r))
        ids = [it['case_id'] for it, _ in rows]
        block = {'steps_scored': len(rows), 'chance': round(statistics.mean(1 / len(it['candidates']) for it, _ in rows), 3) if rows else None}
        for arm in arms:
            block[f'{arm}_top1'] = round(statistics.mean(float(r[arm]['correct']) for _, r in rows), 4) if rows else None
            block[f'{arm}_no_answer'] = sum(r[arm]['choice'] is None for _, r in rows)
        if len(arms) == 1 or not rows:
            by = collections.defaultdict(list)
            for it, r in rows:
                by['1-2 changes' if it['n_changes'] <= 2 else '3-4 changes' if it['n_changes'] <= 4 else '5+ changes'].append(r[arms[0]]['correct'])
            block['by_gt_step_size'] = {k: {'n': len(v), arms[0]: round(statistics.mean(v), 3)} for k, v in sorted(by.items())}
            report[m] = block
            continue
        diff = [float(r['scale']['correct']) - float(r['direct']['correct']) for _, r in rows]
        lo, hi = metrics.clustered_bootstrap_ci(diff, ids, seed=20261031)
        block['scale - direct'] = {'delta': round(statistics.mean(diff), 4), 'ci95': [round(lo, 4), round(hi, 4)],
                                   'scale_only': sum(d > 0 for d in diff), 'direct_only': sum(d < 0 for d in diff)}
        by = collections.defaultdict(list)
        for it, r in rows:
            by['1-2 changes' if it['n_changes'] <= 2 else '3-4 changes' if it['n_changes'] <= 4 else '5+ changes'].append(r)
        block['by_gt_step_size'] = {k: {'n': len(v), 'direct': round(statistics.mean(x['direct']['correct'] for x in v), 3),
                                        'scale': round(statistics.mean(x['scale']['correct'] for x in v), 3)} for k, v in sorted(by.items())}
        report[m] = block
    (out / f"SUMMARY_{'_'.join(arms)}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=C.MODELS)
    ap.add_argument('--workers', type=int, default=150)
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--items', choices=['base', 'matched', 'hard'], default='base')
    ap.add_argument('--arms', nargs='+', default=['direct', 'scale'])
    ap.add_argument('--base-url', default=BASE_URL)
    cli = ap.parse_args()
    out = OUT / cli.items if cli.items != 'base' else OUT
    out.mkdir(parents=True, exist_ok=True)
    items_file = out / 'items.json'
    if items_file.exists():
        items = json.loads(items_file.read_text())
        pool = {c['case_id']: c for c in map(json.loads, (ROOT / 'outputs/fukuyama/clean.jsonl').read_text().splitlines())}
    else:
        items, pool = build_items(matched=cli.items in ('matched', 'hard'), hard=cli.items == 'hard')
        items_file.write_text(json.dumps(items))
    items = [it for it in items if len(it['candidates']) == N_CAND]  # steps with no sane distractor are dropped
    print(json.dumps({'items': len(items)}), flush=True)
    if not cli.analyze_only:
        jobs = [(m, arm, pool[it['case_id']], it, out, cli.base_url) for m in cli.models for arm in cli.arms for it in items]
        random.Random(9).shuffle(jobs)
        with concurrent.futures.ThreadPoolExecutor(cli.workers) as ex:
            for k, fut in enumerate(concurrent.futures.as_completed([ex.submit(job, *j) for j in jobs]), 1):
                rec = fut.result()
                if 'error' in rec or k % 50 == 0:
                    print(json.dumps({'done': k, 'of': len(jobs), **({'error': rec['error']} if 'error' in rec else {})}), flush=True)
    analyze(items, cli.models, out, cli.arms)


if __name__ == '__main__':
    main()
