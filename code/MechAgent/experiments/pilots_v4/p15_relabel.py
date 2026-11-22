"""P15: format-only relabelling of step type/subtype on oMe-Gold (DEVIATION_v5_confirm.md).

The revision prompts of Self-refine, M_ms_none and Mech-Agent did not carry the official
type/subtype list, so their labels are free text and official oMeS (S_partial) scores them ~0.
One call per final path assigns an official pair to every step; the intermediate SMILES are
copied from the original path in code, so species-level metrics cannot change. A label outside
the allowed list keeps the original one.
"""
from __future__ import annotations

import json
import re

import common4 as K
import p11_mechagent as M

ALLOWED = json.loads(re.search(r'Allowed type/subtype pairs:\s*(\[.*?\])\s*\n\n', K.PROMPTS['default'], re.S).group(1))
PAIRS = {(d['type'], d['subtype']) for d in ALLOWED}
PROMPT = ('Below is a stepwise mechanism (intermediate SMILES after each step). Do not change the mechanism. For every '
          'step, choose the type/subtype pair from the allowed list that describes it.\n\nAllowed type/subtype pairs:\n'
          '{allowed}\n\nMechanism:\n{steps}\n\nReturn between [ANSWER] and [/ANSWER] a JSON list with one entry per step, '
          'in order: {{"step": n, "type": ..., "subtype": ..., "intermediate_smiles": ...}} (copy the SMILES).')
ARMS = {'SR2': lambda cid: K.cached('gold/SR2', cid) or K.cached('gold/SR1', cid),
        'M_ms_none': lambda cid: K.cached('gold/M_ms_none', cid),
        'MechAgent': lambda cid: M.final('gold', cid, 2)}


def relabelled(arm, cid):
    """Original path with official labels (None if the original does not parse)."""
    orig = K.parse_mech(ARMS[arm](cid))
    if not orig:
        return None
    lab = K.parse_mech(K.cached(f'gold/RL_{arm}', cid)) or []
    out = []
    for i, s in enumerate(orig):
        t = (lab[i].get('type'), lab[i].get('subtype')) if i < len(lab) else (None, None)
        if t not in PAIRS:
            t = (s.get('type'), s.get('subtype'))
        out.append({'step': i + 1, 'type': t[0], 'subtype': t[1], 'intermediate_smiles': s['intermediate_smiles']})
    return out


def job(arm, cid):
    orig = K.parse_mech(ARMS[arm](cid))
    if not orig:
        return 'noparse'
    steps = '\n'.join(f"  step {j}: {s['intermediate_smiles']}" for j, s in enumerate(orig, 1))
    return 'ok' if K.call(f'gold/RL_{arm}', cid, PROMPT.format(allowed=json.dumps(ALLOWED, indent=1), steps=steps)) else 'error'


if __name__ == '__main__':
    gold = K.load_gold()
    K.run_pool(job, [(a, cid) for a in ARMS for cid in gold], 600)
