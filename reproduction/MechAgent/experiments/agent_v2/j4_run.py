"""J4: pre-registered confirmatory run of MechAgent-v2 on oMe-Gold (fresh data).

Refuses to run unless outputs/agent_v2_J4/REGISTRATION.json exists and the
hashes it records for this script, agent_v2.py, motifs.py, j4_score.py,
the motif library and cases.jsonl all match.

Arms per model and case (same as J3): direct@1..3, hint (motif library mined
from all FukuyamaBench cases), judge (direct@1 vs hint, both orders). The
target is the principal product only (oMe-Gold omits byproducts).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent_v2 as A
import common as C
import j4_score as S
import motifs as M
from common import ROOT
from develop import metrics

OUT = ROOT / 'outputs' / 'agent_v2_J4'
LIB = ROOT / 'outputs' / 'agent_v2_J2' / 'library_all.json'
HINT_K = 12
FROZEN = {'experiments/agent_v2/j4_run.py': HERE / 'j4_run.py', 'experiments/agent_v2/agent_v2.py': HERE / 'agent_v2.py',
          'experiments/agent_v2/motifs.py': HERE / 'motifs.py', 'experiments/agent_v2/j4_score.py': HERE / 'j4_score.py',
          'experiments/agent_v2/common.py': HERE / 'common.py',
          'outputs/agent_v2_J2/library_all.json': LIB, 'outputs/agent_v2_J4/cases.jsonl': OUT / 'cases.jsonl'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_registration():
    reg_path = OUT / 'REGISTRATION.json'
    if not reg_path.exists():
        raise SystemExit('No REGISTRATION.json: J4 must be pre-registered before any model call.')
    reg = json.loads(reg_path.read_text())
    for name, path in FROZEN.items():
        if reg['sha256'].get(name) != sha(path):
            raise SystemExit(f'Frozen file changed since registration: {name}')
    return reg


def target_block(case):
    return ('TARGET (principal product only; byproducts, counter-ions and spectators are NOT shown, so atoms of the '
            f"INITIAL state that are missing here end up in unshown byproducts):\n{case['target_state_pgl']}\n")


def gen_job(model_name, arm, case, lib):
    dest = OUT / model_name / arm / f"{case['case_id']}.json"
    if dest.exists():
        return 'cached'
    model = C.Model(model_name, OUT / model_name / arm / 'traces' / f"{case['case_id']}.jsonl")
    keys = M.select(lib.rank(case), HINT_K, 0.0) if arm == 'hint' else None
    try:
        actions, errors = A.generate(model, case, A.motif_hint(keys) if keys else None, target_block(case), phase=arm)
    except Exception as exc:
        return f'error:{type(exc).__name__}'
    rec = {'case_id': case['case_id'], 'arm': arm, 'model': model_name, 'actions_pgl': actions, 'parse_errors': errors,
           'hint_keys': [list(k) for k in keys] if keys else None, 'tokens': model.tokens, 'calls': len(model.events)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    return 'ok'


def judge_job(model_name, case, order):
    dest = OUT / model_name / 'judge' / f"{case['case_id']}_{order}.json"
    if dest.exists():
        return 'cached'
    base = json.loads((OUT / model_name / 'direct@1' / f"{case['case_id']}.json").read_text())['actions_pgl']
    hint = json.loads((OUT / model_name / 'hint' / f"{case['case_id']}.json").read_text())['actions_pgl']
    if A.sigma_signature(case, base) == A.sigma_signature(case, hint):
        rec = {'case_id': case['case_id'], 'order': order, 'choice': None, 'identical': True, 'tokens': 0}
    else:
        tb, th = A.render_actions(case, base), A.render_actions(case, hint)
        a, b = (th, tb) if order == 'hint_first' else (tb, th)
        model = C.Model(model_name, OUT / model_name / 'judge' / 'traces' / f"{case['case_id']}_{order}.jsonl", max_tokens=4096)
        try:
            choice = A.judge(model, case, a, b, target_block(case))
        except Exception as exc:
            return f'error:{type(exc).__name__}'
        picked = (choice == 'A') if order == 'hint_first' else (choice == 'B')
        rec = {'case_id': case['case_id'], 'order': order, 'choice': choice, 'identical': False,
               'picked_hint': picked if choice else None, 'tokens': model.tokens}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    return 'ok'


def run_pool(jobs, fn, workers):
    done = 0
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for fut in concurrent.futures.as_completed([pool.submit(fn, *j) for j in jobs]):
            done += 1
            status = fut.result()
            if status.startswith('error') or done % 50 == 0:
                print(json.dumps({'done': done, 'of': len(jobs), 'status': status}), flush=True)


def analyze(cases, models, reg):
    metrics_ = ['in_order_fraction', 'in_order_full', 'product_reached', 'oMeS_partial']
    report = {'registration': reg.get('primary')}
    for m in models:
        rows = {}
        for cid, case in cases.items():
            def load(arm):
                f = OUT / m / arm / f'{cid}.json'
                return json.loads(f.read_text())['actions_pgl'] if f.exists() else None
            d = [load(f'direct@{i}') for i in (1, 2, 3)]
            h = load('hint')
            js = [OUT / m / 'judge' / f'{cid}_{o}.json' for o in ('hint_first', 'base_first')]
            if any(x is None for x in d + [h]) or not all(j.exists() for j in js):
                continue
            jr = [json.loads(j.read_text()) for j in js]
            take_hint = (not jr[0]['identical']) and all(r.get('picked_hint') for r in jr)
            v2 = h if take_hint else d[0]
            sc = {arm: S.score_case(case, acts) for arm, acts in
                  (('direct', d[0]), ('vote3', d[A.vote(case, d)]), ('hint', h), ('v2', v2))}
            rows[cid] = {met: {arm: float(sc[arm][met]) for arm in sc} for met in metrics_}
            rows[cid]['took_hint'] = take_hint
        ids = sorted(rows)
        block = {'cases': len(ids), 'missing': len(cases) - len(ids),
                 'took_hint_rate': round(statistics.mean(rows[c]['took_hint'] for c in ids), 4) if ids else None}
        for met in metrics_:
            block[met] = {arm: round(statistics.mean(rows[c][met][arm] for c in ids), 4) for arm in ('direct', 'vote3', 'hint', 'v2')} if ids else {}
            for a, b in (('vote3', 'v2'), ('direct', 'v2'), ('direct', 'hint'), ('direct', 'vote3')):
                diff = [rows[c][met][b] - rows[c][met][a] for c in ids]
                if diff:
                    lo, hi = metrics.clustered_bootstrap_ci(diff, ids, seed=20261030)
                    block[met][f'{b} - {a}'] = {'delta': round(statistics.mean(diff), 4), 'ci95': [round(lo, 4), round(hi, 4)],
                                                'better': sum(x > 0 for x in diff), 'worse': sum(x < 0 for x in diff)}
        p = reg.get('primary', {})
        if p and ids:
            ci = block[p['metric']][p['contrast']]['ci95']
            block['preregistered_verdict'] = 'SUPPORTED: CI excludes 0' if ci[0] > 0 else 'NOT SUPPORTED: CI includes 0' if ci[1] >= 0 else 'NOT SUPPORTED: CI below 0'
        report[m] = block
    (OUT / 'ANALYSIS.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=C.MODELS)
    ap.add_argument('--workers', type=int, default=96)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    reg = check_registration()
    cases = {c['case_id']: c for c in map(json.loads, (OUT / 'cases.jsonl').read_text().splitlines())}
    lib = M.MotifLibrary.from_json(json.loads(LIB.read_text()))
    if not cli.analyze_only:
        gen = [(m, arm, c, lib) for m in cli.models for c in cases.values() for arm in ('direct@1', 'direct@2', 'direct@3', 'hint')]
        random.Random(4).shuffle(gen)
        run_pool(gen, gen_job, cli.workers)
        jud = [(m, c, o) for m in cli.models for c in cases.values() for o in ('hint_first', 'base_first')
               if (OUT / m / 'direct@1' / f"{c['case_id']}.json").exists() and (OUT / m / 'hint' / f"{c['case_id']}.json").exists()]
        run_pool(jud, judge_job, cli.workers)
    analyze(cases, cli.models, reg)


if __name__ == '__main__':
    main()
