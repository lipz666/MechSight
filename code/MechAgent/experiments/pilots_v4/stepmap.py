"""Heavy-atom bond changes per mechanism step by RXNMapper (run with external/env-map).

A step is written as `previous species . reactant/reagent species >> next species` (the official
format keeps one intermediate per step, so partners come from the reactant list). Changes =
bonds formed + bonds broken between mapped atoms + bonds from mapped atoms to atoms that leave
(unmapped). Requests and results are plain JSON so the llm venv never imports rxnmapper:
  in : OUT/stepmap_requests.jsonl  {"rxn": "...>>..."}
  out: OUT/stepmap_cache.json      {rxn: n_changes or null}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / 'outputs' / 'pilots_v4_20261103'
# STEPMAP_REQ / STEPMAP_CACHE let a parallel pipeline use its own files (no write races)
REQ = Path(os.environ.get('STEPMAP_REQ', OUT / 'stepmap_requests.jsonl'))
CACHE = Path(os.environ.get('STEPMAP_CACHE', OUT / 'stepmap_cache.json'))


def n_changes(mapped):
    from rdkit import Chem
    r, p = mapped.split('>>')
    rm, pm = Chem.MolFromSmiles(r, sanitize=False), Chem.MolFromSmiles(p, sanitize=False)
    if rm is None or pm is None:
        return None
    in_product = {a.GetAtomMapNum() for a in pm.GetAtoms() if a.GetAtomMapNum()}

    def bonds(m, keep):
        out = set()
        for b in m.GetBonds():
            a, c = b.GetBeginAtom(), b.GetEndAtom()
            if a.GetSymbol() == 'H' or c.GetSymbol() == 'H':
                continue
            x, y = a.GetAtomMapNum(), c.GetAtomMapNum()
            if keep(x, y):
                out.add(tuple(sorted((x, y))))
        return out
    both = lambda x, y: x in in_product and y in in_product  # noqa: E731
    br, bp = bonds(rm, both), bonds(pm, lambda x, y: bool(x and y))
    leaving = len(bonds(rm, lambda x, y: (x in in_product) != (y in in_product)))
    return len(bp - br) + len(br - bp) + leaving


def main(batch=16):
    from rxnmapper import RXNMapper
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = sorted({json.loads(l)['rxn'] for l in REQ.read_text().splitlines() if l} - cache.keys())
    print(len(todo), 'to map', flush=True)
    mapper = RXNMapper()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        try:
            res = mapper.get_attention_guided_atom_maps(chunk, canonicalize_rxns=False)
        except Exception:
            res = []
            for rxn in chunk:
                try:
                    res.append(mapper.get_attention_guided_atom_maps([rxn], canonicalize_rxns=False)[0])
                except Exception:
                    res.append(None)
        for rxn, r in zip(chunk, res):
            try:
                cache[rxn] = n_changes(r['mapped_rxn']) if r else None
            except Exception:
                cache[rxn] = None
        if i // batch % 20 == 0:
            CACHE.write_text(json.dumps(cache))
            print(i + len(chunk), '/', len(todo), flush=True)
    CACHE.write_text(json.dumps(cache))
    print('done', len(cache), flush=True)


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:]))
