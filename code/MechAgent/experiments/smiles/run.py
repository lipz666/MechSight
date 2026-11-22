"""SMILES-mode experiments (main representation from 2026-11-03).

Datasets: J3 (147 Fukuyama, diagnostics), T3 (95 Fukuyama held-out; dev/confirm split in
benchmarks_20261103/splits.json), oMe (oMe-Gold 169).
Models: gemini (two-gateway client), gpt-6-luna and claude-opus-5-5-high (vps),
gemini-4-argon (117 gateway).
Conditions:
  M1, M2   direct stepwise SMILES mechanism (smiles_mech.TASK), two samples
  Vp, V0   M1 prompt + one observed intermediate: GT hidden species (Vp) or GT visible
           species (V0), chosen as in the PGL visibility experiment (J3 only)
Outputs: outputs/smiles_main_20261103/<dataset>/<model>/<cond>/<case>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'mechsight'))
sys.path.insert(0, str(HERE.parent / 'inert_knowledge'))
import smiles_mech as SM  # noqa: E402
from smiles_mech import A, C, P, PL, chem, phase2  # noqa: E402
import failover as F  # noqa: E402
import baselines as B  # noqa: E402

OUT = P.ROOT / 'outputs' / 'smiles_main_20261103'
GATEWAYS = {'vps': ('https://LLM-GATEWAY-2.example/v1', 'OPENAI_API_KEY'), '117': ('http://LLM-GATEWAY-1.example:8080/v1', 'FALLBACK_API_KEY')}
MODELS = {'gpt-6-luna': 'vps', 'claude-opus-5-5-high': 'vps', 'gemini-4-argon': '117'}
OBS = ('\nExperimental observation: the following intermediate was detected during this reaction '
       '(SMILES): {smiles}')


def make_model(model, trace):
    if model == 'gemini':
        return F.FailoverModel(trace, max_tokens=P.MAX_TOKENS)
    from openai import OpenAI
    url, key_env = GATEWAYS[MODELS[model]]
    m = C.Model(model, trace, base_url=url, max_tokens=P.MAX_TOKENS)
    m.client = OpenAI(api_key=os.environ[key_env], base_url=url, timeout=900, max_retries=0)
    m.endpoint = url
    return m


def load_cases():
    sys.path.insert(0, str(P.ROOT / 'experiments' / 'agent_v2'))
    import j3_dev
    sets = B.load_sets()
    return {'J3': j3_dev.load_cases(), 'T3': sets['T3'], 'oMe': sets['oMe']}


def shown(case):
    sys.path.insert(0, str(HERE.parent / 'inert_knowledge'))
    import visibility as V
    return V.shown_species(case)


def prompt_for(case):
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    target = chem.parse_pgl_literal(case['target_state_pgl'])
    return SM.TASK.format(reactants=P.endpoint_smiles(initial), products=P.endpoint_smiles(target),
                          conditions=A.conditions_text(case))


def job(dataset, model, cond, case):
    dest = OUT / dataset / model / cond / f"{case['case_id']}.json"
    if dest.exists():
        return 'cached'
    prompt = prompt_for(case)
    extra = None
    if cond in ('Vp', 'V0'):
        s = shown(case)
        if not s or (cond == 'V0' and not s[2]):
            return 'skip'
        extra = s[0] if cond == 'Vp' else s[2]
        prompt += OBS.format(smiles=extra)
    m = make_model(model, OUT / dataset / model / cond / 'traces' / f"{case['case_id']}.jsonl")
    try:
        ans = m.complete([{'role': 'system', 'content': P.FREE_SYSTEM}, {'role': 'user', 'content': prompt}], cond)
    except Exception as exc:
        return f'error:{type(exc).__name__}'
    found = re.search(r'<mechanism>(.*?)</mechanism>', ans, re.S)
    lines = [ln.strip().strip('`').strip() for ln in (found.group(1) if found else '').splitlines()]
    lines = [re.sub(r'^(step\s*\d+\s*[:.)-]\s*|\d+\s*[:.)]\s*)', '', ln, flags=re.I) for ln in lines if ln]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({'case_id': case['case_id'], 'model': model, 'cond': cond, 'shown': extra,
                                'states': lines, 'tag_found': bool(found),
                                'endpoints': getattr(m, 'endpoints_used', None) or [getattr(m, 'endpoint', '')]}, indent=1))
    return 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--datasets', nargs='+', default=['J3'])
    ap.add_argument('--conds', nargs='+', default=['M1', 'M2'])
    ap.add_argument('--workers', type=int, default=100)
    ap.add_argument('--limit', type=int, default=0)
    cli = ap.parse_args()
    sets = load_cases()
    jobs = []
    for ds in cli.datasets:
        cases = list(sets[ds].values())[:cli.limit or None]
        jobs += [(ds, cli.model, cond, c) for cond in cli.conds for c in cases]
    PL.pool(job, jobs, cli.workers)


if __name__ == '__main__':
    main()
