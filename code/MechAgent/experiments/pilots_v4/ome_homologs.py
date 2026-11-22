"""Build minimal homologues of oMe-Gold reactions for the P5 contamination probe.

Run with external/env-map (rxnmapper). The overall reaction (reactants >> products) is
atom-mapped; atoms whose mapped neighbourhood changes are the reaction centre. One CH3 is put
on an uncharged carbon carrying H that is >= 3 bonds from every centre atom (the farthest
one), in the reactant and on the mapped atom of the product that contains it.
Usage: env-map/bin/python -I ome_homologs.py <oMe_Gold.json> <out.json>
"""
import json
import sys
from collections import deque

from rdkit import Chem, RDLogger
from rxnmapper import RXNMapper

RDLogger.DisableLog('rdApp.*')


def neighbours(mols):
    nb = {}
    for m in mols:
        for a in m.GetAtoms():
            if a.GetAtomMapNum():
                nb[a.GetAtomMapNum()] = {n.GetAtomMapNum() for n in a.GetNeighbors()}
    return nb


def add_methyl(mol, idx):
    rw = Chem.RWMol(mol)
    c = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(idx, c, Chem.BondType.SINGLE)
    a = rw.GetAtomWithIdx(idx)
    a.SetNoImplicit(False)
    a.SetNumExplicitHs(max(0, a.GetNumExplicitHs() - 1))
    m = rw.GetMol()
    for at in m.GetAtoms():
        at.SetAtomMapNum(0)
    Chem.SanitizeMol(m)
    return Chem.MolToSmiles(m)


def build(rec, mapper):
    rs, ps = rec['reactants_smiles'], rec['products_smiles']
    mapped = mapper.get_attention_guided_atom_maps(['.'.join(rs) + '>>' + '.'.join(ps)])[0]['mapped_rxn']
    left, right = mapped.split('>>')
    rmols = [Chem.MolFromSmiles(s) for s in left.split('.')]
    pmols = [Chem.MolFromSmiles(s) for s in right.split('.')]
    if any(m is None for m in rmols + pmols):
        return None
    rn, pn = neighbours(rmols), neighbours(pmols)
    centre = {k for k in rn if rn[k] != pn.get(k, set())} | {k for k in pn if k not in rn}
    best = None
    for ri, m in enumerate(rmols):
        dist, q = {}, deque()
        for a in m.GetAtoms():
            if a.GetAtomMapNum() in centre:
                dist[a.GetIdx()] = 0
                q.append(a.GetIdx())
        if not q:
            continue
        while q:
            i = q.popleft()
            for n in m.GetAtomWithIdx(i).GetNeighbors():
                if n.GetIdx() not in dist:
                    dist[n.GetIdx()] = dist[i] + 1
                    q.append(n.GetIdx())
        for a in m.GetAtoms():
            d = dist.get(a.GetIdx(), 0)
            if a.GetSymbol() == 'C' and a.GetTotalNumHs() >= 1 and a.GetFormalCharge() == 0 and d >= 3 \
                    and a.GetAtomMapNum() and (best is None or d > best[0]):
                best = (d, ri, a.GetIdx(), a.GetAtomMapNum())
    if best is None:
        return None
    d, ri, idx, mapnum = best
    hit = [(pi, a.GetIdx()) for pi, m in enumerate(pmols) for a in m.GetAtoms() if a.GetAtomMapNum() == mapnum]
    if not hit:
        return None
    pi, pidx = hit[0]
    new_r = [add_methyl(m, idx) if i == ri else Chem.MolToSmiles(_strip(m)) for i, m in enumerate(rmols)]
    new_p = [add_methyl(m, pidx) if i == pi else Chem.MolToSmiles(_strip(m)) for i, m in enumerate(pmols)]
    return {'reaction_id': rec['reaction_id'], 'distance': d, 'reactants': new_r, 'products': new_p,
            'orig_reactants': rs, 'orig_products': ps, 'conditions': rec.get('conditions', ''),
            'changed_product_index': pi}


def _strip(m):
    m = Chem.Mol(m)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return m


if __name__ == '__main__':
    mapper = RXNMapper()
    out = []
    for rec in json.load(open(sys.argv[1])):
        try:
            h = build(rec, mapper)
        except Exception:
            h = None
        if h:
            out.append(h)
    json.dump(out, open(sys.argv[2], 'w'), indent=1)
    print(len(out), 'homologues')
