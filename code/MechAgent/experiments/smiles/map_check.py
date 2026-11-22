"""V1: can SMILES-only mechanisms be turned back into sigma-bond events by atom mapping?

GT FukuyamaBench paths (PGL, exact) are rendered as one dot-joined SMILES per state; every
consecutive pair is atom-mapped with RXNMapper (no GT maps given), and the heavy-atom bonds
formed/broken are compared with the GT step, as element-pair multisets.
Run with the env-map interpreter (rxnmapper); the PGL side is pre-exported by --export
under the llm venv.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs' / 'smiles_validation_20261103'


def export():
    sys.path.insert(0, str(ROOT / 'experiments' / 'agent_v2'))
    import common as C
    import j3_dev
    from common import chem, fm
    rows = []
    for cid, case in j3_dev.load_cases().items():
        initial = chem.parse_pgl_literal(case['initial_state_pgl'])
        states = [initial] + C.gt_states(case)
        smiles = ['.'.join(C._molecule_smiles(s, c) for c in C._components(s)) for s in states]
        atoms = [dict(chem.pgl.iter_atoms(s)) for s in states]
        steps = []
        for k, step in enumerate(fm.gold_steps_from_clean(case)):
            el = lambda lab: atoms[k + 1].get(lab, atoms[k].get(lab)).element
            heavy = lambda a, b: el(a) != 'H' and el(b) != 'H'
            steps.append({'form': sorted('-'.join(sorted((el(a), el(b)))) for a, b in step['form'] if heavy(a, b)),
                          'break': sorted('-'.join(sorted((el(a), el(b)))) for a, b in step['break'] if heavy(a, b))})
        rows.append({'case': cid, 'smiles': smiles, 'steps': steps})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'gt_smiles_paths.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    print(len(rows), 'cases exported')


def bond_changes(mapped):
    from rdkit import Chem
    r, p = mapped.split('>>')
    def bonds(smi):
        m = Chem.MolFromSmiles(smi, sanitize=False)
        out = {}
        for b in m.GetBonds():
            a, c = b.GetBeginAtom(), b.GetEndAtom()
            if a.GetAtomMapNum() and c.GetAtomMapNum():
                out[tuple(sorted((a.GetAtomMapNum(), c.GetAtomMapNum())))] = tuple(sorted((a.GetSymbol(), c.GetSymbol())))
        return out
    br, bp = bonds(r), bonds(p)
    return (sorted('-'.join(bp[k]) for k in bp.keys() - br.keys()), sorted('-'.join(br[k]) for k in br.keys() - bp.keys()))


def check():
    from rxnmapper import RXNMapper
    mapper = RXNMapper()
    rows = [json.loads(l) for l in (OUT / 'gt_smiles_paths.jsonl').read_text().splitlines()]
    n_step = ok_step = n_case = ok_case = failed = 0
    per = []
    for row in rows:
        all_ok = True
        for k, gt in enumerate(row['steps']):
            rxn = f"{row['smiles'][k]}>>{row['smiles'][k + 1]}"
            n_step += 1
            try:
                mapped = mapper.get_attention_guided_atom_maps([rxn])[0]['mapped_rxn']
                form, brk = bond_changes(mapped)
                good = Counter(form) == Counter(gt['form']) and Counter(brk) == Counter(gt['break'])
            except Exception:
                failed += 1
                good = False
            ok_step += good
            all_ok &= good
        n_case += 1
        ok_case += all_ok
        per.append({'case': row['case'], 'all_steps_ok': all_ok})
    rep = {'steps': n_step, 'steps_recovered': ok_step, 'step_rate': round(ok_step / n_step, 3),
           'mapper_failures': failed, 'cases': n_case, 'cases_all_steps_recovered': ok_case,
           'case_rate': round(ok_case / n_case, 3),
           'note': 'heavy-atom bond changes as element-pair multisets; proton transfers not checked'}
    (OUT / 'MAP_CHECK.json').write_text(json.dumps(rep, indent=2))
    (OUT / 'map_check_rows.jsonl').write_text('\n'.join(json.dumps(r) for r in per) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    export() if '--export' in sys.argv else check()
