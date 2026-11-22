"""Legal elementary moves on a PGL state (direction two: enumerate -> recognise -> search).

Two-electron moves only (polar mechanisms), in the spirit of Synthegy's
ionisation/attack move set:
  ionize(x, y -> y)    a bond x-y loses one order; the pair goes to y
  attack(u -> v)       a lone pair on u forms (or strengthens) a bond u-v
  pi_attack(a=b -> v)  pi electrons of a=b form a bond b-v (a loses one order)
A concerted annotated step is reached as a short sequence of such moves, so
the intermediates of the annotation are still visited.

Moves are restricted to the reaction region of the endpoints (atoms whose
bonds or charge differ, their heavy neighbours, catalysts/reagents of other
elements, hydrogens on them and free protons) and kept only if every touched
atom stays electron-sane: no negative or odd non-bonding count, octet for
period-2 atoms, duet for H, at most 12 electrons for heavier main-group atoms.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from develop import fukuyama_metrics as fm
from develop import standalone_chemistry as chem

VALENCE = {'H': 1, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Br': 7, 'I': 7, 'Se': 6,
           'Li': 1, 'Na': 1, 'K': 1, 'Mg': 2, 'Al': 3, 'Sn': 4}
PERIOD2 = {'B', 'C', 'N', 'O', 'F'}
METALS = {'Li', 'Na', 'K', 'Mg', 'Ca', 'Al', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Rh', 'Pd', 'Ag', 'Sn',
          'Os', 'Pt', 'Au', 'Hg', 'Ce', 'Sm', 'In', 'Pb', 'Tl'}


def view(state):
    atoms = dict(chem.pgl.iter_atoms(state))
    order = collections.defaultdict(float)
    adj = collections.defaultdict(set)
    for (x, y), b in chem.pgl.bond_map(state).items():
        o = float(b.order or 1.0)
        order[frozenset((x, y))] = o
        adj[x].add(y)
        adj[y].add(x)
    return atoms, order, adj


def electrons(name, atoms, order, adj):
    """(bonding-order sum incl. recorded H, non-bonding electrons) for one atom."""
    a = atoms[name]
    bonded = sum(order[frozenset((name, m))] for m in adj[name]) + int(a.Hcount or 0)
    nonbonding = VALENCE.get(a.element, 0) - int(a.charge or 0) - bonded
    return bonded, nonbonding


def sane(name, atoms, order, adj):
    a = atoms[name]
    if a.element in METALS or a.element not in VALENCE:
        return True
    bonded, nb = electrons(name, atoms, order, adj)
    if nb < 0 or abs(nb - round(nb)) > 1e-6 or round(nb) % 2:
        return False
    total = 2 * bonded + nb
    if a.element == 'H':
        return total <= 2
    if a.element in PERIOD2:
        return total <= 8
    return total <= 12


def deficient(name, atoms, order, adj):
    """Fewer electrons than a closed shell: carbocations, Cl+, free H+, trivalent B/Al."""
    a = atoms[name]
    if a.element not in VALENCE or a.element in METALS:
        return False
    bonded, nb = electrons(name, atoms, order, adj)
    total = 2 * bonded + nb
    return total < (2 if a.element == 'H' else 8)


def region(case, state):
    """Reaction-region atoms (from the supplied endpoints) present in this state."""
    ini = chem.parse_pgl_literal(case['initial_state_pgl'])
    tgt = chem.parse_pgl_literal(case['target_state_pgl'])
    ia, ta = dict(chem.pgl.iter_atoms(ini)), dict(chem.pgl.iter_atoms(tgt))
    bi, bt = set(chem.pgl.bond_map(ini)), set(chem.pgl.bond_map(tgt))
    core = {x for p in bi ^ bt for x in p} | {n for n in ta if n in ia and ta[n].charge != ia[n].charge}
    atoms, order, adj = view(state)
    heavy = {n for n in core if n in atoms and atoms[n].element != 'H'}
    heavy |= {m for n in list(heavy) for m in adj[n] if atoms[m].element != 'H'}
    heavy |= {n for n, a in atoms.items() if a.element != 'H' and a.element not in {'C', 'H', 'O', 'N'}}
    hyd = {m for n in heavy for m in adj[n] if atoms[m].element == 'H'}
    free_h = {n for n, a in atoms.items() if a.element == 'H' and not adj[n]}
    return heavy | hyd | free_h


def _transfer(source, target, kind):
    return chem.ReactionTransformation('reaction', [chem.ElectronTransfer(source=source, target=target, electrons=2, kind=kind)])


def enumerate_moves(case, state, region_atoms=None):
    """[(label, ReactionTransformation, resulting_state)] for every sane move."""
    atoms, order, adj = view(state)
    reg = region_atoms if region_atoms is not None else region(case, state)
    cands = []
    for pair, o in order.items():
        x, y = tuple(pair)
        if not (x in reg and y in reg) or o <= 0:
            continue
        for a, b in ((x, y), (y, x)):
            kind = 'pi' if o >= 2 else 'sigma'
            cands.append((f'ionize {a}-{b} ({kind}), electron pair to {b}', _transfer((a, b, f'{kind}_bond'), b, 'bond_breaking'), {a, b}))
    donors = [n for n in reg if atoms[n].element != 'H' and electrons(n, atoms, order, adj)[1] >= 2]
    acceptors = [n for n in reg if deficient(n, atoms, order, adj)
                 or atoms[n].element in METALS or atoms[n].element in {'Si', 'P', 'S', 'Sn'}]
    for u in donors:
        for v in acceptors:
            if u == v:
                continue
            cands.append((f'{u} lone pair attacks {v} (forms {u}-{v})', _transfer(u, (u, v, 'new_bond'), 'bond_forming'), {u, v}))
    for pair, o in order.items():
        if o < 2:
            continue
        a, b = tuple(pair)
        if not (a in reg and b in reg):
            continue
        for x, y in ((a, b), (b, a)):
            for v in acceptors:
                if v in (x, y):
                    continue
                cands.append((f'pi electrons of {x}={y} attack {v} (forms {y}-{v})', _transfer((x, y, 'pi_bond'), (y, v, 'new_bond'), 'pi_attack'), {x, y, v}))
    out = []
    seen = set()
    for label, action, touched in cands:
        try:
            new, _ = chem.apply_action(state, action.to_dict())
        except Exception:
            continue
        na, no, nadj = view(new)
        if not all(sane(n, na, no, nadj) for n in touched if n in na):
            continue
        sig = signature(new)
        if sig in seen:
            continue
        seen.add(sig)
        out.append((label, action, new))
    return out


def signature(state):
    s = chem.state_signature(state)
    return (tuple(sorted((k, v['charge']) for k, v in s['atoms'].items())),
            tuple(sorted((k, v['order']) for k, v in s['bonds'].items())))


def reached(state, case):
    return fm.same_state(fm.state_view(state), fm.state_view(chem.parse_pgl_literal(case['target_state_pgl'])))
