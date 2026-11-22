"""T3 contamination probe: product prediction on original problems vs. minimal homologues.

A memorised benchmark answer should not survive a perturbation the answer key never
contained. Each J3 case with an eligible spectator carbon (ks.analog_case: one CH3 >= 3 bonds
from every atom any GT step touches, mechanism unchanged) is asked, from reactants and
conditions only, for the major product(s). Original and homologue are scored with the
sigma-graph key (heavy-atom connectivity + H counts; charges and bond orders ignored) on the
product species that are new relative to the starting materials.
Registration: outputs/contamination_20261103/REGISTRATION.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'inert_knowledge'))
import pilot as PL  # noqa: E402
from pilot import A, C, P, chem, phase2  # noqa: E402
import baselines as B  # noqa: E402
import ks  # noqa: E402

OUT = P.ROOT / 'outputs' / 'contamination_20261103'
TASK = ('Reactants and reagents (SMILES): {reactants}\n'
        'Reaction conditions: {conditions}\n\n'
        'Predict the major organic product(s) of this reaction. Reason briefly, then give the product SMILES, one '
        'per line, inside <products> and </products>.')


def new_product_keys(case):
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    target = chem.parse_pgl_literal(case['target_state_pgl'])
    start = {P.state_component_keys(initial, c)[0] for c in C._components(initial)}
    return {k for c in C._components(target) if (k := P.state_component_keys(target, c)[0]) and k not in start
            and len(c) > 3}


def job(model, version, case):
    dest = OUT / model / version / f"{case['case_id']}.json"
    if dest.exists():
        return 'cached'
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    prompt = TASK.format(reactants=P.endpoint_smiles(initial), conditions=A.conditions_text(case))
    m = B.make_model(model, OUT / model / version / 'traces' / f"{case['case_id']}.jsonl")
    try:
        answer = m.complete([{'role': 'system', 'content': P.FREE_SYSTEM}, {'role': 'user', 'content': prompt}], version)
    except Exception as exc:
        return f'error:{type(exc).__name__}'
    found = re.search(r'<products>(.*?)</products>', answer, re.S)
    smiles = [s.strip().strip('`') for s in (found.group(1) if found else '').splitlines() if s.strip()]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({'case_id': case['case_id'], 'version': version, 'products': smiles, 'prompt': prompt,
                                'endpoints': getattr(m, 'endpoints_used', [B.VPS])}, indent=1))
    return 'ok'


def correct(case, rec):
    want = new_product_keys(case)
    got = {k for s in rec['products'] for k, _ in P.smiles_keys(s)}
    return bool(want) and want <= got


def analyze(pairs, models):
    rep = {}
    for model in models:
        rows = []
        for cid, (orig, hom) in pairs.items():
            ro, rh = PL.load(OUT / model / 'original', cid), PL.load(OUT / model / 'homologue', cid)
            if ro and rh and new_product_keys(orig) and new_product_keys(hom):
                rows.append({'case': cid, 'orig': float(correct(orig, ro)), 'hom': float(correct(hom, rh))})
        ids = [r['case'] for r in rows]
        rep[model] = {'pairs': len(rows), 'original': phase2.ci([r['orig'] for r in rows], ids),
                      'homologue': phase2.ci([r['hom'] for r in rows], ids),
                      'original_minus_homologue': phase2.ci([r['orig'] - r['hom'] for r in rows], ids),
                      'only_original_correct': sum(r['orig'] > r['hom'] for r in rows),
                      'only_homologue_correct': sum(r['hom'] > r['orig'] for r in rows)} if rows else {'pairs': 0}
        (OUT / model).mkdir(parents=True, exist_ok=True)
        (OUT / model / 'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    (OUT / 'SUMMARY.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['gemini', 'gpt-6-luna'])
    ap.add_argument('--workers', type=int, default=200)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    sys.path.insert(0, str(HERE.parent / 'agent_v2'))
    import j3_dev
    cases = j3_dev.load_cases()
    pairs = {cid: (c, h) for cid, c in cases.items() if (h := ks.analog_case(c))}
    if cli.limit:
        pairs = dict(list(pairs.items())[:cli.limit])
    OUT.mkdir(parents=True, exist_ok=True)
    if not cli.analyze_only:
        PL.pool(job, [(m, v, c) for m in cli.models for cid, (o, h) in pairs.items() for v, c in (('original', o), ('homologue', h))],
                cli.workers)
    analyze(pairs, cli.models)


if __name__ == '__main__':
    main()
