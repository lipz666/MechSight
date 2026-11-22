"""Direction-two pilot: enumerate legal moves -> LLM recognises the best -> search.

Per case, three rollouts: rollout r takes the LLM's rank-r move at the first
step and its rank-1 move afterwards (moves revisiting a state are dropped).
A rollout stops when the target state is reached, no legal move is left, or
after MAX_DEPTH moves. If several rollouts reach the target, they are
compared pairwise (both orders) by the same judge as MechAgent-v2; otherwise
the first rollout is kept. The LLM never writes arrows itself.

Pilot only (development cases); the baselines are the J3 arms of the same
model on the same cases.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'agent_v2'))
import agent_v2 as A
import common as C
import moves as MV
from common import ROOT, chem, fm
from develop import metrics

OUT = ROOT / 'outputs' / 'step_search_pilot'
MAX_DEPTH = 24
SYSTEM = ('You are an expert physical organic chemist. A mechanism is being built one two-electron move at a time; at '
          'each point you choose among ALL legal moves, listed with the species each one produces. Atom labels refer to '
          'the states shown. Choose moves that a chemist would accept as the actual mechanism under the conditions, not '
          'merely the shortest route to the target.')


def state_species(state):
    return ', '.join(C._molecule_smiles(state, comp) for comp in C._components(state))


def rank_prompt(case, state, history, cands):
    lines = []
    for i, (label, _, new) in enumerate(cands):
        _, species = C.step_text(state, new)
        lines.append(f'[{i}] {label}  ->  {species}')
    hist = '\n'.join(f'  {k + 1}. {h}' for k, h in enumerate(history)) or '  (none yet)'
    return (f"INITIAL:\n{case['initial_state_pgl']}\nTARGET:\n{case['target_state_pgl']}\n"
            f"Conditions: {A.conditions_text(case)}\nMoves made so far:\n{hist}\n"
            f"CURRENT species: {state_species(state)}\n\nLegal next moves:\n" + '\n'.join(lines) +
            '\n\nRank the three best next moves for the accepted mechanism. Reason briefly, then answer with exactly '
            '<rank>i,j,k</rank> using the bracketed numbers.')


def parse_rank(answer, n):
    found = re.findall(r'<rank>\s*([0-9,\s]+)\s*</rank>', answer)
    if not found:
        return []
    out = []
    for tok in found[-1].split(','):
        tok = tok.strip()
        if tok.isdigit() and int(tok) < n and int(tok) not in out:
            out.append(int(tok))
    return out


def rollout(model_name, case, r, run_dir):
    dest = run_dir / f"{case['case_id']}_r{r}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    model = C.Model(model_name, run_dir / 'traces' / f"{case['case_id']}_r{r}.jsonl", max_tokens=4096)
    state = chem.parse_pgl_literal(case['initial_state_pgl'])
    reg = MV.region(case, state)
    visited = {MV.signature(state)}
    actions, history, status = [], [], 'depth_cap'
    for depth in range(MAX_DEPTH):
        cands = [c for c in MV.enumerate_moves(case, state, reg | MV.region(case, state)) if MV.signature(c[2]) not in visited]
        if not cands:
            status = 'no_moves'
            break
        try:
            answer = model.complete([{'role': 'system', 'content': SYSTEM},
                                     {'role': 'user', 'content': rank_prompt(case, state, history, cands)}], f'rank{depth}')
        except Exception as exc:
            status = f'api_error:{type(exc).__name__}'
            break
        ranking = parse_rank(answer, len(cands))
        if not ranking:
            status = 'unparseable_rank'
            break
        pick = ranking[min(r, len(ranking) - 1)] if depth == 0 else ranking[0]
        label, action, state = cands[pick]
        visited.add(MV.signature(state))
        actions.append(repr(action))
        history.append(label)
        if MV.reached(state, case):
            status = 'reached_target'
            break
    rec = {'case_id': case['case_id'], 'rollout': r, 'status': status, 'actions_pgl': actions, 'history': history,
           'calls': len(model.events), 'tokens': model.tokens}
    if not status.startswith('api_error'):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rec, indent=1))
    return rec


def select(model_name, case, rolls, run_dir):
    """Pick one rollout: among those reaching the target, a pairwise judge decides."""
    good = [r for r in rolls if r['status'] == 'reached_target']
    if not good:
        return rolls[0], 'none_reached'
    if len(good) == 1:
        return good[0], 'single_reached'
    best = good[0]
    for other in good[1:]:
        if A.sigma_signature(case, best['actions_pgl']) == A.sigma_signature(case, other['actions_pgl']):
            continue
        wins = 0
        for order in ('best_first', 'other_first'):
            f = run_dir / 'judge' / f"{case['case_id']}_{best['rollout']}v{other['rollout']}_{order}.json"
            if f.exists():
                choice = json.loads(f.read_text())['choice']
            else:
                model = C.Model(model_name, run_dir / 'judge' / 'traces' / f.name.replace('.json', '.jsonl'), max_tokens=4096)
                ta, tb = A.render_actions(case, best['actions_pgl']), A.render_actions(case, other['actions_pgl'])
                choice = A.judge(model, case, *((ta, tb) if order == 'best_first' else (tb, ta)))
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps({'choice': choice}))
            other_letter = 'B' if order == 'best_first' else 'A'
            wins += choice == other_letter
        if wins == 2:
            best = other
    return best, 'judged'


def pilot_cases():
    pool = {}
    for src in [ROOT / 'outputs/fukuyama/clean.jsonl', ROOT / 'outputs/heldout_fukuyama_20261024/heldout.jsonl']:
        for line in src.read_text().splitlines():
            c = json.loads(line)
            pool[c['case_id']] = c
    dev = json.loads((ROOT / 'outputs/p23_planning_probe_20261024/cases.json').read_text())
    j3 = sorted(p.stem for p in (ROOT / 'outputs/agent_v2_J3/gemini-4-argon-build/direct@1').glob('*.json'))
    rest = sorted(set(j3) - set(dev), key=lambda c: hashlib.sha256(f'pilot:{c}'.encode()).hexdigest())
    return {cid: pool[cid] for cid in dev + rest[:6]}


def analyze(cases, model_name, run_dir):
    j3 = ROOT / 'outputs/agent_v2_J3' / model_name
    rows = {}
    for cid, case in cases.items():
        rolls = [run_dir / f'{cid}_r{r}.json' for r in range(3)]
        if not all(p.exists() for p in rolls):
            continue
        rolls = [json.loads(p.read_text()) for p in rolls]
        chosen, how = select(model_name, case, rolls, run_dir)
        def load(arm):
            return json.loads((j3 / arm / f'{cid}.json').read_text())['actions_pgl']
        d = [load(f'direct@{i}') for i in (1, 2, 3)]
        jr = [json.loads((j3 / 'judge' / f'{cid}_{o}.json').read_text()) for o in ('hint_first', 'base_first')]
        v2 = load('hint') if (not jr[0]['identical'] and all(r.get('picked_hint') for r in jr)) else d[0]
        arms = {'search': chosen['actions_pgl'], 'direct': d[0], 'vote3': d[A.vote(case, d)], 'v2': v2}
        rows[cid] = {arm: fm.score_case(case, acts) for arm, acts in arms.items()}
        rows[cid]['meta'] = {'how': how, 'reached': [r['status'] for r in rolls], 'calls': sum(r['calls'] for r in rolls),
                             'any_rollout_sigma': any(fm.score_case(case, r['actions_pgl'])['sigma_L1'] for r in rolls),
                             'any_rollout_in_order': any(fm.score_case(case, r['actions_pgl'])['intermediates_in_order'] for r in rolls)}
    ids = sorted(rows)
    report = {'cases': len(ids), 'model': model_name,
              'rollout_status': dict(__import__('collections').Counter(s for c in ids for s in rows[c]['meta']['reached'])),
              'mean_calls_per_case_search': round(statistics.mean(rows[c]['meta']['calls'] for c in ids), 1) if ids else None}
    for met in ('sigma_L1', 'intermediates_in_order'):
        report[met] = {arm: round(statistics.mean(float(rows[c][arm][met]) for c in ids), 4) for arm in ('search', 'direct', 'vote3', 'v2')}
        report[met]['any_of_3_rollouts'] = round(statistics.mean(float(rows[c]['meta'][f"any_rollout_{'sigma' if met == 'sigma_L1' else 'in_order'}"]) for c in ids), 4)
        for base in ('direct', 'vote3', 'v2'):
            diff = [float(rows[c]['search'][met]) - float(rows[c][base][met]) for c in ids]
            lo, hi = metrics.clustered_bootstrap_ci(diff, ids, seed=20261031)
            report[met][f'search - {base}'] = {'delta': round(statistics.mean(diff), 4), 'ci95': [round(lo, 4), round(hi, 4)],
                                               'better': sum(x > 0 for x in diff), 'worse': sum(x < 0 for x in diff)}
    (run_dir / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gemini-4-argon-build')
    ap.add_argument('--workers', type=int, default=90)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    cases = pilot_cases()
    if cli.limit:
        cases = dict(list(cases.items())[:cli.limit])
    run_dir = OUT / cli.model
    run_dir.mkdir(parents=True, exist_ok=True)
    (OUT / 'cases.json').write_text(json.dumps(list(cases)))
    if not cli.analyze_only:
        jobs = [(cli.model, c, r, run_dir) for c in cases.values() for r in range(3)]
        random.Random(5).shuffle(jobs)
        with concurrent.futures.ThreadPoolExecutor(cli.workers) as pool:
            for k, fut in enumerate(concurrent.futures.as_completed([pool.submit(rollout, *j) for j in jobs]), 1):
                rec = fut.result()
                print(json.dumps({'done': k, 'of': len(jobs), 'case': rec['case_id'], 'r': rec['rollout'],
                                  'status': rec['status'], 'moves': len(rec['actions_pgl'])}), flush=True)
    analyze(cases, cli.model, run_dir)


if __name__ == '__main__':
    main()
