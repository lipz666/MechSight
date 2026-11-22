"""Build the chemist review payload for the inert-knowledge experiments.

Tasks (one item per heavy-atom hidden event unless noted):
  match   C-4: the model's free-text intermediate that the sigma-key matched to the GT hidden
          species. Is it really the same intermediate?
  class   C-3: which class of hidden intermediate is this, and is the GT step chemically sound?
  analog  KS-3: does the CH3 homologue keep the mechanism unchanged? (20 sampled cases)

Drawings: GT mechanism frames come from develop/viz/render_rdkit.py (bonds about to break in red,
bonds about to form as dashed lines, shared layout across steps). Single species are drawn with
RDKit; the transient bond is highlighted.

Output: review/data/index.json plus one review/data/cases/<case>.json per case.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / 'agent_v2'))
import ks  # noqa: E402
import probe as P  # noqa: E402
from probe import C, chem, fm  # noqa: E402

VIZ = P.ROOT / 'develop' / 'viz'
sys.path.insert(0, str(VIZ))
import render_rdkit as R  # noqa: E402
from extract import state_dict, union_coords  # noqa: E402

from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdDepictor  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402
from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog('rdApp.*')
DATA = HERE / 'data'
ORDER = {1.0: Chem.BondType.SINGLE, 2.0: Chem.BondType.DOUBLE, 3.0: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}
HL = (0.18, 0.62, 0.42)
ANALOG_SAMPLE = 20


# ------------------------------------------------------------------ drawing
def _svg(mol, width=380, height=260, atoms=(), bonds=()):
    try:
        rdDepictor.Compute2DCoords(mol)
    except Exception:
        return ''
    d = rdMolDraw2D.MolDraw2DSVG(width, height)
    o = d.drawOptions()
    o.useBWAtomPalette()
    o.clearBackground = False
    o.bondLineWidth = 1.6
    o.minFontSize = 12
    o.highlightBondWidthMultiplier = 14
    o.highlightRadius = 0.32
    try:
        d.DrawMolecule(mol, highlightAtoms=list(atoms), highlightAtomColors={i: HL for i in atoms},
                       highlightBonds=list(bonds), highlightBondColors={i: HL for i in bonds})
    except Exception:
        return ''
    d.FinishDrawing()
    return d.GetDrawingText().replace('#000000', 'currentColor')


def smiles_svg(smi, width=380, height=260):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return ''
        mol.UpdatePropertyCache(strict=False)
    return _svg(mol, width, height)


def component_svg(state, members, transient_pairs):
    """One GT species, heavy atoms only, with the transient bond(s) highlighted."""
    atoms = dict(chem.pgl.iter_atoms(state))
    bmap = chem.pgl.bond_map(state)
    heavy = sorted(n for n in members if atoms[n].element != 'H')
    hcount = {n: int(atoms[n].Hcount or 0) for n in heavy}
    for (x, y) in bmap:
        if x in hcount and atoms[y].element == 'H':
            hcount[x] += 1
        if y in hcount and atoms[x].element == 'H':
            hcount[y] += 1
    rw = Chem.RWMol()
    idx = {}
    for n in heavy:
        a = Chem.Atom(atoms[n].element)
        a.SetFormalCharge(int(atoms[n].charge or 0))
        a.SetNumExplicitHs(hcount[n])
        a.SetNoImplicit(True)
        idx[n] = rw.AddAtom(a)
    hl_atoms, hl_bonds = set(), []
    for (x, y), bond in bmap.items():
        if x in idx and y in idx:
            rw.AddBond(idx[x], idx[y], ORDER.get(float(bond.order or 1.0), Chem.BondType.SINGLE))
            if P.anon_pair(x, y) in transient_pairs:
                hl_bonds.append(rw.GetBondBetweenAtoms(idx[x], idx[y]).GetIdx())
                hl_atoms |= {idx[x], idx[y]}
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    return _svg(mol, 420, 300, hl_atoms, hl_bonds), C._molecule_smiles(state, members)


def mechanism_frames(case):
    """GT path as rendered frames: each frame shows what the next step breaks and forms."""
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    states = [initial] + C.gt_states(case)
    sds = [state_dict(s) for s in states]
    coords = union_coords(sds)
    canvas = R.canvas_for(coords)
    world = R.global_transform(coords)
    frames = []
    for k, sd in enumerate(sds):
        nxt = sds[k + 1] if k + 1 < len(sds) else None
        try:
            drawn = R.render_frame({'state': sd, 'transfers': []}, coords, canvas, world, next_state=nxt, show_labels=False)
            svg = drawn['svg']
        except Exception as exc:
            svg = f'<text x="20" y="30">render failed: {type(exc).__name__}</text>'
        change = ''
        if k < len(case['gt_step_changes']):
            s = case['gt_step_changes'][k]
            parts = [f'form {a}–{b}' for a, b, _ in s['bonds_formed']] + [f'break {a}–{b}' for a, b, _ in s['bonds_broken']]
            change = '; '.join(parts)
        frames.append({'svg': svg, 'next_change': change})
    return {'canvas': canvas, 'frames': frames}


# ------------------------------------------------------------------ items
def endpoints(case):
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    target = chem.parse_pgl_literal(case['target_state_pgl'])
    r, p = P.endpoint_smiles(initial), P.endpoint_smiles(target)
    return {'reactants': r, 'products': p, 'reactants_svg': smiles_svg(r, 560, 260), 'products_svg': smiles_svg(p, 560, 260),
            'conditions': P.A.conditions_text(case)}


def hidden_cards(case):
    """Per heavy hidden event: window, species drawing, and step indices."""
    gold = fm.gold_steps_from_clean(case)
    per_step = [fm.sigma_events(s) for s in gold]
    states = C.gt_states(case)
    out = {}
    for t, info in P.hidden_events(case).items():
        if not info['heavy'] or not info['keys']:
            continue
        touched = [i for i, ev in enumerate(per_step) if any((a, b) == t for _, a, b in ev)]
        k = touched[0]
        st = states[k]
        comp = next((c for c in C._components(st) if set(t) & c), None)
        svg, smi = component_svg(st, comp, {t}) if comp else ('', '')
        out[t] = {'pair': '–'.join(t), 'window': [touched[0] + 1, touched[-1]], 'species_svg': svg, 'species_smiles': smi}
    return out


def build():
    cases = P.__dict__.get('_cases') or __import__('j3_dev').load_cases()
    rows = {(r['case'], r['pair']): r for r in map(json.loads, (P.OUT / 'rows.jsonl').read_text().splitlines())}
    prelabels = {}
    pre = HERE / 'prelabels.json'
    if pre.exists():
        prelabels = json.loads(pre.read_text())
    (DATA / 'cases').mkdir(parents=True, exist_ok=True)
    index = {'tasks': {'match': [], 'class': [], 'analog': []}}
    analog_pool = []
    for cid, case in cases.items():
        cards = hidden_cards(case)
        if not cards:
            continue
        free = {c: (P.load(c, cid) or {}).get('intermediates', []) for c in ('S', 'G')}
        events = []
        for t, card in cards.items():
            info = P.hidden_events(case)[t]
            row = rows.get((cid, '-'.join(t)), {})
            matched = []
            for cond in ('S', 'G'):
                for smi in free[cond]:
                    if any(k in info['keys'] for k, _ in P.smiles_keys(smi)):
                        matched.append({'cond': cond, 'smiles': smi, 'svg': smiles_svg(smi)})
                        break
            item_id = f"{cid}:{'-'.join(t)}"
            ev = {**card, 'id': item_id, 'used_in_E': bool(row.get('E')), 'known_free_text': bool(row.get('S') or row.get('G')),
                  'matched': matched, 'prelabel': prelabels.get(item_id)}
            events.append(ev)
            index['tasks']['class'].append({'id': item_id, 'case': cid, 'pair': card['pair']})
            if matched:
                index['tasks']['match'].append({'id': item_id, 'case': cid, 'pair': card['pair']})
        payload = {'case': cid, 'partition': case['partition'], **endpoints(case), 'mechanism': mechanism_frames(case),
                   'events': events,
                   'free_text': {c: [{'smiles': s, 'svg': smiles_svg(s, 300, 210)} for s in free[c]] for c in ('S', 'G')}}
        a = ks.analog_case(case)
        if a:
            analog_pool.append(cid)
            init = chem.parse_pgl_literal(a['initial_state_pgl'])
            payload['analog'] = {'site': a['analog_site'], 'distance': a['analog_site_distance'],
                                 'reactants': P.endpoint_smiles(init), 'reactants_svg': smiles_svg(P.endpoint_smiles(init), 560, 260)}
        (DATA / 'cases' / f'{cid}.json').write_text(json.dumps(payload))
        print(cid, len(events), flush=True)
    for cid in sorted(random.Random(20261102).sample(sorted(analog_pool), min(ANALOG_SAMPLE, len(analog_pool)))):
        index['tasks']['analog'].append({'id': f'{cid}:analog', 'case': cid, 'pair': 'CH3 homologue'})
    (DATA / 'index.json').write_text(json.dumps(index, indent=1))
    print({k: len(v) for k, v in index['tasks'].items()})


if __name__ == '__main__':
    build()
