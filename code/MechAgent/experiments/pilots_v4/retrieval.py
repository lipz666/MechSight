"""Mechanism memory for MechSight: oMe-Template (167) + oMe-Silver (2493), Silver-dev removed.

Retrieval combines (1) names: the model's own identification of the target (named
reactions / classes) matched to corpus names, and (2) structure: Tanimoto similarity of
RDKit reaction difference fingerprints. One analog per distinct name, up to k.
strict=True removes every corpus entry that shares the target's own name (used on
Silver-dev, where the corpus otherwise holds ~20 siblings of each name).
"""
from __future__ import annotations

import functools
import json
import re

from rdkit import DataStructs, RDLogger
from rdkit.Chem import rdChemReactions

import common4 as K

RDLogger.DisableLog('rdApp.*')

IDENTIFY = ('Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: '
            '{conditions}\n\nIdentify this transformation. Give up to three named reactions or well-known reaction '
            'types it belongs to (most likely first), using standard textbook names. Answer only with JSON: '
            '{{"names": ["...", "..."]}}')


def diff_fp(case):
    try:
        rxn = rdChemReactions.ReactionFromSmarts('.'.join(case['reactants']) + '>>' + '.'.join(case['products']),
                                                 useSmiles=True)
        return rdChemReactions.CreateDifferenceFingerprintForReaction(rxn)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def corpus():
    dev = set(K.silver_dev())
    items = [c for c in K.load_template() + K.load_silver() if c['id'] not in dev]
    for c in items:
        c['norm'] = K.norm_name(c['name'])
        c['fp'] = diff_fp(c)
    return items


def identify(case, ds):
    ans = K.call(f'{ds}/identify', case['id'], IDENTIFY.format(
        reactants='.'.join(case['reactants']), products='.'.join(case['products']), conditions=case['conditions']))
    m = re.search(r'\{.*\}', ans or '', re.S)
    try:
        return [K.norm_name(x) for x in json.loads(m.group(0)).get('names', [])][:3] if m else []
    except Exception:
        return []


def _name_match(a, b):
    if not a or not b:
        return False
    if a == b or (len(a) > 6 and len(b) > 6 and (a in b or b in a)):
        return True
    ta, tb = set(a.split()) - {'reaction', 'the', 'of'}, set(b.split()) - {'reaction', 'the', 'of'}
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= 0.6


def retrieve(case, names, k=3, strict=False):
    own = K.norm_name(case.get('name'))
    fp = diff_fp(case)
    pool = [c for c in corpus() if not (strict and own and _name_match(c['norm'], own))]
    sim = lambda c: DataStructs.TanimotoSimilarity(fp, c['fp']) if fp is not None and c['fp'] is not None else 0.0
    picked, seen = [], set()
    for n in names:
        hits = sorted((c for c in pool if _name_match(c['norm'], n)), key=sim, reverse=True)
        if hits and hits[0]['norm'] not in seen:
            picked.append(hits[0])
            seen.add(hits[0]['norm'])
        if len(picked) == k:
            return picked
    for c in sorted(pool, key=sim, reverse=True):
        if c['norm'] not in seen:
            picked.append(c)
            seen.add(c['norm'])
        if len(picked) == k:
            break
    return picked


def render(analogs):
    out = []
    for i, a in enumerate(analogs, 1):
        steps = '\n'.join(f"  {j}. {s['subtype']}: {s['intermediate_smiles']}" for j, s in enumerate(a['gold'], 1))
        out.append(f"Analog {i}: {a['name']}\nReactants: {'.'.join(a['reactants'])}\nProducts: "
                   f"{'.'.join(a['products'])}\nConditions: {a['conditions']}\nMechanism (subtype: intermediate):\n{steps}")
    return '\n\n'.join(out)
