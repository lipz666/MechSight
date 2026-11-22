"""T4: direct-generation baselines (E prompt + audit) on the benchmark sets.

Sets:
  oMe   oMe-Gold, 169 reactions converted to PGL (agent_v2_J4/cases.jsonl); scored with
        j4_score (species-level in-order, product reached, type-agnostic oMeS)
  T3    FukuyamaBench held-out T3 (95; GT confidence < 0.3); scored with fukuyama_metrics,
        split once into dev (47) / confirm (48) with seed 20261103 (saved to splits.json)
Models: gemini (two-endpoint failover) and gpt-6-luna (vps).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot as PL  # noqa: E402
from pilot import A, C, P, fm, phase2  # noqa: E402
import failover as F  # noqa: E402
import j4_score  # noqa: E402

ROOT = P.ROOT
OUT = ROOT / 'outputs' / 'benchmarks_20261103'
VPS = 'https://LLM-GATEWAY-2.example/v1'


def load_sets():
    ome = {json.loads(l)['case_id']: json.loads(l) for l in (ROOT / 'outputs/agent_v2_J4/cases.jsonl').read_text().splitlines()}
    reg = json.loads((ROOT / 'outputs/heldout_fukuyama_20261024/REGISTRATION.json').read_text())
    pool = {json.loads(l)['case_id']: json.loads(l) for l in (ROOT / 'outputs/heldout_fukuyama_20261024/heldout.jsonl').read_text().splitlines()}
    t3 = {cid: pool[cid] for cid in sorted(reg['sets']['sensitivity_T3'])}
    split_file = OUT / 'splits.json'
    if not split_file.exists():
        ids = sorted(t3)
        random.Random(20261103).shuffle(ids)
        OUT.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps({'T3_dev': sorted(ids[:47]), 'T3_confirm': sorted(ids[47:]),
                                          'note': 'fixed 2026-11-03 before any T3 method run; seed 20261103'}, indent=1))
    return {'oMe': ome, 'T3': t3}


def make_model(model, trace):
    if model == 'gemini':
        return F.FailoverModel(trace, max_tokens=P.MAX_TOKENS)
    return C.Model(model, trace, base_url=VPS, max_tokens=P.MAX_TOKENS)


def job(dataset, model, case):
    dest = OUT / dataset / model / 'E' / f"{case['case_id']}.json"
    if dest.exists():
        return 'cached'
    m = make_model(model, OUT / dataset / model / 'E' / 'traces' / f"{case['case_id']}.jsonl")
    try:
        actions, errors = phase2.generate(m, A.user_prompt(case), 'E')
    except Exception as exc:
        return f'error:{type(exc).__name__}'
    used = m.endpoints_used if hasattr(m, 'endpoints_used') else [VPS]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({'case_id': case['case_id'], 'model': model, 'endpoints': used,
                                'actions_pgl': actions, 'parse_errors': errors, 'tokens': m.tokens}, indent=1))
    return 'ok'


def score(dataset, case, actions):
    if dataset == 'oMe':
        return j4_score.score_case(case, actions)
    s = fm.score_case(case, actions)
    s['exact_path'] = bool(s['intermediates_in_order_strict']) and s['executed_steps'] == s['gold_steps']
    return s


def analyze(sets, models):
    rep = {}
    for ds, cases in sets.items():
        for model in models:
            rows = []
            for cid, case in cases.items():
                rec = PL.load(OUT / ds / model / 'E', cid)
                if rec:
                    rows.append((cid, score(ds, case, rec['actions_pgl'])))
            if not rows:
                continue
            ids = [c for c, _ in rows]
            keys = (['in_order_fraction', 'in_order_full', 'product_reached', 'oMeS_total', 'oMeS_partial'] if ds == 'oMe'
                    else ['sigma_L1', 'intermediates_in_order', 'exact_path', 'final_state_match'])
            rep[f'{ds}/{model}'] = {'cases': len(rows), 'of': len(cases),
                                   **{k: phase2.ci([float(s.get(k) or 0) for _, s in rows], ids) for k in keys}}
    (OUT / 'SUMMARY_baselines.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['gemini', 'gpt-6-luna'])
    ap.add_argument('--sets', nargs='+', default=['oMe', 'T3'])
    ap.add_argument('--workers', type=int, default=200)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    sets = {k: v for k, v in load_sets().items() if k in cli.sets}
    if cli.limit:
        sets = {k: dict(list(v.items())[:cli.limit]) for k, v in sets.items()}
    if not cli.analyze_only:
        PL.pool(job, [(ds, m, c) for ds, cs in sets.items() for m in cli.models for c in cs.values()], cli.workers)
    analyze(sets, cli.models)


if __name__ == '__main__':
    main()
