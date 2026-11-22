"""(1) Automated audit of species MechSight-Auto adds to the Direct pathway on the held-out benchmarks.
(2) Candidate showcase reactions: Direct fails, Auto succeeds, Direct has a >=3-bond step that Auto splits
    into <=2-bond steps through a reference intermediate.
Run from MechAgent/experiments/pilots_v4."""
import json, sys
sys.path.insert(0, '.')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
import common4 as K
import p11_mechagent as M
import p12_controls as C

S = sys.argv[1]


def ekeys(smi):
    out = set()
    for s in (smi or '').split('.'):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            out.add(Chem.MolToInchiKey(m).split('-')[0])
        except Exception:
            pass
    return out


def refs(case):
    if 'pgl' in case:
        h, v = C.gt_intermediates(case)
        return h + v, set(h)
    return [g['intermediate_smiles'] for g in case['gold']], set()


audit = {'exact': 0, 'equivalent': 0, 'not_in_reference': 0}
per_set = {}
cands = []
for ds, cases in M.sets(('T3_confirm', 'gold')).items():
    for cid, case in cases.items():
        d_ans, a_ans = K.cached(f'{ds}/base_default', cid), M.final(ds, cid, 2)
        if d_ans is None or a_ans is None:
            continue
        dm, am = K.parse_mech(d_ans) or [], K.parse_mech(a_ans) or []
        rlist, hid = refs(case)
        rk = [K.species_keys(r) for r in rlist]
        rek = [ekeys(r) for r in rlist]
        dkeys = [K.species_keys(s.get('intermediate_smiles')) for s in dm]
        ends = set().union(*[K.species_keys(x) for x in case['reactants'] + case['products']])
        for s in am:
            smi = s.get('intermediate_smiles') or ''
            k = K.species_keys(smi)
            if not k or k <= ends or any(k == dk for dk in dkeys):
                continue  # not a newly inserted species
            if any(r and r <= k for r in rk):
                c = 'exact'
            elif any(r and r <= ekeys(smi) for r in rek):
                c = 'equivalent'
            else:
                c = 'not_in_reference'
            audit[c] += 1
            per_set.setdefault(ds, {}).setdefault(c, 0)
            per_set[ds][c] += 1
        # showcase search
        if K.score(case, dm)['all_present'] or not K.score(case, am)['all_present']:
            continue
        dg = M.gaps(case, dm)
        if not any('bonds' in r for _, r in dg):
            continue
        ag = M.gaps(case, am)
        heavy_atoms = max((Chem.MolFromSmiles(x).GetNumHeavyAtoms() for x in case['reactants'] if Chem.MolFromSmiles(x)), default=99)
        cands.append({'ds': ds, 'id': cid, 'name': case.get('name', ''), 'n_ref': len(rlist), 'max_reactant_heavy': heavy_atoms,
                      'auto_gaps_left': len(ag), 'reactants': case['reactants'], 'products': case['products'],
                      'conditions': case.get('conditions', ''),
                      'direct': [(s.get('intermediate_smiles'), M.changes(r)) for s, r in zip(dm, M.step_rxns(case, dm))],
                      'auto': [(s.get('intermediate_smiles'), M.changes(r)) for s, r in zip(am, M.step_rxns(case, am))],
                      'reference': rlist, 'hidden': sorted(hid)})
tot = sum(audit.values())
print('inserted species', tot, {k: (v, round(100 * v / tot, 1)) for k, v in audit.items()}, per_set)
cands.sort(key=lambda c: (c['auto_gaps_left'], c['n_ref'], c['max_reactant_heavy']))
json.dump({'audit': audit, 'per_set': per_set, 'cands': cands}, open(S, 'w'), indent=1)
for c in cands[:12]:
    print(c['ds'], c['id'], c['name'][:60], 'nref', c['n_ref'], 'heavy', c['max_reactant_heavy'], 'gaps_left', c['auto_gaps_left'])
