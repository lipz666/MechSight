"""P18: Fig. 1a / Fig. 3a - specialised models vs LLM vs Mech-Agent, in-template vs out-of-template.

Specialists from run_spec.py on the 4060 (synced to OUT/spec_v5/{chrimp,flower}); scoring rules of
spec_score.py (ChRIMP top-1 mechanism; FlowER best path to the main product). Benchmarks:
uspto31k (200, mech-USPTO-31k sample = the specialists' training distribution), T3_confirm (48),
oMe-Gold (196). Metric: all GT intermediates present (species level, same scorer for every arm),
plus product reached. Each comparison uses the cases the specialist finished (coverage reported).
"""
from __future__ import annotations

import json
import os

import common4 as K
import p11_mechagent as M
import spec_score as SS

SPEC = K.OUT / os.environ.get('SPEC_DIR', 'spec_v5')  # spec_v6_4090 = 2026-11-08 4090 rerun, 900 s limit
BENCH = {'uspto31k': K.load_uspto, 'T3_confirm': lambda: K.load_fukuyama('T3_confirm'), 'gold': K.load_gold}


def steps(states):
    return [{'intermediate_smiles': s} for s in states]


def reached(case, smis):
    target = SS.main_keys(case)
    return float(bool(target) and any(target <= K.species_keys(s) for s in smis if s))


def resource_failure(rec):
    """GPU out-of-memory after a retry: a hardware limit of our host, counted as missing coverage, not as a model miss."""
    return 'CUDA' in (rec.get('error') or '')


def arms(ds, cid, case):
    out = {}
    f = SPEC / 'chrimp' / f'{ds}__{cid}.json'
    if f.exists() and not resource_failure(rec := json.loads(f.read_text())):
        out['ChRIMP'] = SS.chrimp_states(rec, case)
    f = SPEC / 'flower' / f'{ds}__{cid}.json'
    if f.exists() and not resource_failure(rec := json.loads(f.read_text())):
        out['FlowER'] = SS.flower_path(rec['graph'], case) if rec.get('graph') else []
    for arm, ans in (('LLM_direct', K.cached(f'{ds}/base_default', cid)), ('MechAgent', M.final(ds, cid, 2))):
        m = K.parse_mech(ans) if ans is not None else None
        if ans is not None:
            out[arm] = [s.get('intermediate_smiles') or '' for s in (m or [])]
    return out


def main():
    rep = {}
    for ds, load in BENCH.items():
        cases = load()
        rows = {}
        for cid, case in cases.items():
            for arm, smis in arms(ds, cid, case).items():
                s = K.score(case, steps(smis))
                rows.setdefault(arm, []).append((cid, {'all_present': s['all_present'], 'reached': reached(case, smis)}))
        block = {'n_cases': len(cases)}
        for spec in ('ChRIMP', 'FlowER'):
            done = {c for c, _ in rows.get(spec, [])}
            if not done:
                continue
            sub = {a: [(c, m) for c, m in r if c in done] for a, r in rows.items()}
            block[spec] = {'coverage': len(done),
                           **{a: K.summarize(r, ['all_present', 'reached']) for a, r in sub.items() if len(r) == len(done)}}
            for a in ('LLM_direct', 'MechAgent'):
                if len(sub.get(a, [])) == len(done):
                    block[spec][f'{a}_minus_{spec}'] = K.paired(sub[a], sub[spec], 'all_present')
        rep[ds] = block
    (K.OUT / ('SUMMARY_P18_SPEC.json' if SPEC.name == 'spec_v5' else f'SUMMARY_P18_SPEC_{SPEC.name}.json')).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
