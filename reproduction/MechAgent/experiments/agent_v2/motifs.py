"""Mechanistic motif library: which kinds of atom pairs form transient bonds.

Mined by counting over expert-annotated mechanisms (no model is trained).
For every structural candidate pair (the enumerator of the shortlist pilot:
reaction region + first shell + catalysts/reagents, H* on region atoms) a
motif describes the two atoms in the INITIAL state at several levels of
detail; the library stores, per motif, how often such a candidate turned out
to be a transient bond (formed and later broken) in the annotated mechanism.

Scoring backs off from the most specific motif with enough support to coarser
ones, with add-one smoothing toward the global base rate.
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'p23_planning_probe'))
from develop import fukuyama_metrics as fm
from develop import standalone_chemistry as chem

import run as P23  # candidate_transients, pair_key

MIN_SUPPORT = 3


def _graph(state):
    atoms = dict(chem.pgl.iter_atoms(state))
    adj = collections.defaultdict(set)
    orders = {}
    for (x, y), bond in chem.pgl.bond_map(state).items():
        adj[x].add(y)
        adj[y].add(x)
        orders[frozenset((x, y))] = float(bond.order or 1.0)
    return atoms, adj, orders


METALS = {'Li', 'Na', 'K', 'Mg', 'Ca', 'B', 'Al', 'Si', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Rh', 'Pd',
          'Ag', 'Sn', 'Os', 'Pt', 'Au', 'Hg', 'Ce', 'Sm', 'In', 'Pb', 'Tl', 'Se', 'Te'}


def _hcount(name, atoms, adj):
    return int(atoms[name].Hcount or 0) + sum(1 for m in adj[name] if atoms[m].element == 'H')


def _endpoint_roles(initial, target):
    """Per heavy atom: net H change, net charge change, bonds formed/broken net (endpoints only)."""
    ia, iadj, _ = _graph(initial)
    ta, tadj, _ = _graph(target)
    roles = {}
    for n, a in ia.items():
        if a.element == 'H':
            continue
        if n not in ta:
            roles[n] = ('gone',)
            continue
        dh = _hcount(n, ta, tadj) - _hcount(n, ia, iadj)
        dq = int(ta[n].charge or 0) - int(a.charge or 0)
        hi = {m for m in iadj[n] if ia[m].element != 'H'}
        ht = {m for m in tadj[n] if ta[m].element != 'H'}
        formed, broken = len(ht - hi), len(hi - ht)
        bond = 'fb' if formed and broken else 'f' if formed else 'b' if broken else '0'
        roles[n] = (max(-1, min(1, dh)), max(-1, min(1, dq)), bond)
    return roles


def _descriptor(name, atoms, adj, orders, core, roles):
    a = atoms[name]
    heavy = [m for m in adj[name] if atoms[m].element != 'H']
    aromatic = any(abs(orders[frozenset((name, m))] - 1.5) < 1e-6 for m in adj[name])
    unsat = any(orders[frozenset((name, m))] >= 2 for m in adj[name])
    q = int(a.charge or 0)
    kind = 'ar' if aromatic else ('pi' if unsat else 'sat')
    cls = 'metal' if a.element in METALS else 'core' if name in core else 'shell'
    role = roles.get(name, ('?',))
    fine = f"{a.element}{q:+d}|H{min(_hcount(name, atoms, adj), 3)}|D{len(heavy)}|{kind}|{cls}|{role}"
    mid = f"{a.element}{q:+d}|{kind}|{cls}|{role[-1]}"
    coarse = f"{a.element}|{cls}"
    return fine, mid, coarse


def _same_molecule(a, b, adj):
    frontier, seen = {a}, {a}
    while frontier:
        if b in frontier:
            return True
        frontier = {m for n in frontier for m in adj[n]} - seen
        seen |= frontier
    return False


def _distance(a, b, adj, cap=4):
    if a == b:
        return 0
    frontier, seen, d = {a}, {a}, 0
    while frontier and d < cap:
        d += 1
        frontier = {m for n in frontier for m in adj[n]} - seen
        if b in frontier:
            return d
        seen |= frontier
    return 'far' if _same_molecule(a, b, adj) else 'diffmol'


def candidate_motifs(case):
    """[(pair_key, (motif_L3, motif_L2, motif_L1, motif_L0))] for every structural candidate."""
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    target = chem.parse_pgl_literal(case['target_state_pgl'])
    atoms, adj, orders = _graph(initial)
    ta = dict(chem.pgl.iter_atoms(target))
    bi, bt = set(chem.pgl.bond_map(initial)), set(chem.pgl.bond_map(target))
    core = {x for pair in bi ^ bt for x in pair} | {n for n in ta if n in atoms and ta[n].charge != atoms[n].charge}
    roles = _endpoint_roles(initial, target)
    out = []
    for a, b in P23.candidate_transients(case):
        key = P23.pair_key(a, b)
        if a == 'H*' or b == 'H*':
            heavy = b if a == 'H*' else a
            f, m, c = _descriptor(heavy, atoms, adj, orders, core, roles)
            motifs = (('H', f), ('H', m), ('H', c, roles.get(heavy, ('?',))[0]), ('H', atoms[heavy].element))
        else:
            fa, ma, ca = _descriptor(a, atoms, adj, orders, core, roles)
            fb, mb, cb = _descriptor(b, atoms, adj, orders, core, roles)
            dist = _distance(a, b, adj)
            dist = dist if dist in (2, 3, 'diffmol') else ('far' if dist == 'far' or (isinstance(dist, int) and dist >= 4) else dist)
            motifs = (tuple(sorted((fa, fb))) + (dist,), tuple(sorted((ma, mb))) + (dist,),
                      tuple(sorted((ca, cb))) + (dist == 'diffmol',), tuple(sorted((atoms[a].element, atoms[b].element))))
        out.append((key, motifs))
    return out


def gold_transients(case):
    keys = P23.transient_bonds([P23.sx._step_from_clean(s) for s in case['gt_step_changes']])
    return {tuple(k[1:]) for k in keys}


class MotifLibrary:
    def __init__(self):
        self.pos = [collections.Counter() for _ in range(4)]
        self.tot = [collections.Counter() for _ in range(4)]
        self.base_pos = self.base_tot = 0

    def add_case(self, case):
        gold = gold_transients(case)
        for key, motifs in candidate_motifs(case):
            hit = key in gold
            self.base_pos += hit
            self.base_tot += 1
            for level, motif in enumerate(motifs):
                self.tot[level][motif] += 1
                self.pos[level][motif] += hit
        return self

    def probability(self, motifs):
        prior = (self.base_pos + 1) / (self.base_tot + 2)
        p = prior
        # back off from coarse to fine: each level shrinks toward the coarser estimate
        for level in (3, 2, 1, 0):
            n = self.tot[level][motifs[level]]
            k = self.pos[level][motifs[level]]
            if n >= MIN_SUPPORT or level == 3:
                p = (k + 2 * p) / (n + 2)
        return p

    def rank(self, case):
        scored = [(self.probability(motifs), key) for key, motifs in candidate_motifs(case)]
        return sorted(scored, key=lambda x: (-x[0], x[1]))

    def to_json(self):
        return {'base': [self.base_pos, self.base_tot],
                'levels': [[[list(m) if isinstance(m, tuple) else m, self.pos[l][m], self.tot[l][m]] for m in self.tot[l]] for l in range(4)]}

    @classmethod
    def from_json(cls, data):
        lib = cls()
        lib.base_pos, lib.base_tot = data['base']
        def as_key(m):
            return tuple(as_key(x) if isinstance(x, list) else x for x in m) if isinstance(m, list) else m
        for l, rows in enumerate(data['levels']):
            for m, k, n in rows:
                m = as_key(m)
                lib.pos[l][m], lib.tot[l][m] = k, n
        return lib


def select(ranked, k_max=8, threshold=0.0):
    return [key for p, key in ranked[:k_max] if p >= threshold]
