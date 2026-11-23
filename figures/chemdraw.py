"""ChemDraw-style structure renderer + CDXML export for the MechSight manuscript figures (run with external/venv-llm/bin/python).

RDKit is used only for 2D coordinates. Every mark is drawn here following ChemDraw's ACS Document 1996 conventions:

  document settings      bond length 14.4 pt, line 0.6 pt, margin 1.6 pt, bond spacing 18 %, chain angle 120°,
                         Arial 10 pt labels (standard ACS Document 1996 journal styling)
  figure scale           one uniform 69 % scale for every structure in every figure
                         (figure units: 1 unit = 175 mm / 1000 = 0.496 pt -> bond 20 units)
  double bonds           ring: outer full line + inner line offset 18 % toward the ring centre, trimmed on the
                         angle bisectors; acyclic C=X and tetrasubstituted C=C: two centred lines; other acyclic
                         C=C: offset toward the substituents
  labels                 heteroatoms only; element symbol centred on the atom; implicit H written on the free
                         side (OH / HO / NH above or below); bonds stop a 1.6 pt margin short of the glyphs
  charges / radicals     "+" / "–" at the upper right of a label, or beside a bare vertex (no "C" is shown for
                         a charged carbon); radical as a dot on the most open side
Every structure is also written as .cdxml with the ACS 1996 document settings, so it opens in ChemDraw as a native,
editable drawing (File > Open; then Structure > Clean Up if desired).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point2D

# ---- ACS Document 1996, scaled 69 % into 0.496-pt figure units
PT = 0.496                       # pt per figure unit
SCALE = 0.69
L = 14.4 * SCALE / PT            # bond length          ~20.0 units
FONT = 10 * SCALE / PT           # label size           ~13.9 units
LINE = 0.6 * SCALE / PT          # line width           ~0.83 units
MARGIN = 1.6 * SCALE / PT        # label margin         ~2.2 units
SPACING = 0.18 * L               # double-bond spacing  ~3.6 units
CAP = 0.716                      # Arial cap height / em
FAMILY = 'Arial, Helvetica, sans-serif'

# Arial advance widths (em) for the glyphs we need
ADV = {'C': .722, 'N': .722, 'O': .778, 'S': .667, 'B': .667, 'r': .333, 'I': .278, 'F': .611, 'l': .222,
       'i': .222, 'P': .667, 'H': .722, 'K': .667, 'a': .556, 'u': .556, 'e': .556, 'g': .556, 'n': .556,
       '2': .556, '3': .556, '4': .556, '+': .584, '−': .584}

# abbreviations: dummy atoms [1*], [2*], ... in a spec's "draw" SMILES; (label when bonded from the left, from the right)
ABBR = {1: ('Ph', 'Ph'), 2: ('SiPh3', 'Ph3Si'), 3: ('C3F7', 'F7C3'), 4: ('CO2Me', 'MeO2C'), 5: ('OSiPh3', 'Ph3SiO'), 6: ('Tol', 'Tol'), 7: ('PPh3', 'Ph3P'), 8: ('E', 'E'), 9: ('PMP', 'PMP'), 10: ('TT+', 'TT+'), 11: ('TT•', 'TT•'), 12: ('Si(TMS)3', '(TMS)3Si'), 13: ('Pz', 'Pz'), 14: ('Ar', 'Ar')}
ABBR_HEAD = {1: 'P', 2: 'Si', 3: 'C', 4: 'C', 5: 'O', 6: 'T', 7: 'P', 8: 'E', 9: 'P', 10: 'T', 11: 'T', 12: 'S', 13: 'P', 14: 'A'}     # glyphs that sit on the attachment point

def _sub_digits(t, size):
    """Digits as subscripts; the baseline is restored on the text that follows (not on an empty tspan)."""
    import re as _re
    out, down = [], False
    for tok in _re.findall(r'\d+|\D+', t):
        if tok.isdigit():
            out.append(f'<tspan font-size="{size * 0.72:.2f}" dy="{size * 0.2:.2f}">{tok}</tspan>'); down = True
        else:
            out.append(f'<tspan dy="{-size * 0.2:.2f}">{tok}</tspan>' if down else tok); down = False
    return ''.join(out)


def adv(s, size=FONT):
    return sum(ADV.get(c, .6) for c in s) * size


# --------------------------------------------------------------------------------------------- coordinates
def _coords(smi, template=None, rot=0.0, flip=None, coordgen=True, template_coordgen=True, tweaks=None, coords=None):
    m = Chem.MolFromSmiles(smi)
    if template:
        tm = Chem.MolFromSmiles(template[0]); rdDepictor.SetPreferCoordGen(template_coordgen); rdDepictor.Compute2DCoords(tm)
        core = Chem.MolFromSmarts(template[1])
        a, b = m.GetSubstructMatch(core), tm.GetSubstructMatch(core)
        cmap = {i: Point2D(tm.GetConformer().GetAtomPosition(j).x, tm.GetConformer().GetAtomPosition(j).y)
                for i, j in zip(a, b)}
        rdDepictor.SetPreferCoordGen(False)
        rdDepictor.Compute2DCoords(m, coordMap=cmap)
    else:
        rdDepictor.SetPreferCoordGen(coordgen)
        rdDepictor.Compute2DCoords(m)
    rdDepictor.SetPreferCoordGen(True)
    conf = m.GetConformer()
    P = [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(m.GetNumAtoms())]
    if flip == 'x':
        P = [[-x, y] for x, y in P]
    elif flip == 'y':
        P = [[x, -y] for x, y in P]
    if rot:
        t = math.radians(rot)
        P = [[x * math.cos(t) - y * math.sin(t), x * math.sin(t) + y * math.cos(t)] for x, y in P]
    bl = [math.dist(P[b.GetBeginAtomIdx()], P[b.GetEndAtomIdx()]) for b in m.GetBonds()]
    s = L / (sum(bl) / len(bl)) if bl else L
    P = [[x * s, -y * s] for x, y in P]          # SVG: y down
    if coords:                                    # hand-placed atoms, in bond lengths, y down (bridged cages)
        for k, (cx, cy) in coords.items():
            P[int(k)] = [cx * L, cy * L]
    if tweaks:
        for k, (dx, dy) in tweaks.items():
            P[int(k)][0] += dx; P[int(k)][1] += dy
    Chem.Kekulize(m, clearAromaticFlags=True)
    return m, P


# --------------------------------------------------------------------------------------------- geometry helpers
def _ang(v):
    return math.atan2(v[1], v[0])

def _unit(v):
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n) if n else (0.0, 0.0)

def _free_dir(P, i, nbrs, prefs=None):
    """Direction (radians, SVG frame) of the most open sector around atom i."""
    if not nbrs:
        return 0.0
    angs = sorted(_ang((P[j][0] - P[i][0], P[j][1] - P[i][1])) for j in nbrs)
    if prefs:
        best, bd = None, -1
        for a in prefs:
            d = min(abs((a - b + math.pi) % (2 * math.pi) - math.pi) for b in angs)
            if d > bd + 1e-6:
                best, bd = a, d
        return best
    if len(angs) == 1:
        return angs[0] + math.pi
    gaps = [(angs[(k + 1) % len(angs)] - angs[k]) % (2 * math.pi) for k in range(len(angs))]
    k = max(range(len(gaps)), key=gaps.__getitem__)
    return angs[k] + gaps[k] / 2

def _clip(p, q, box):
    """Move p toward q until it leaves the axis-aligned box (x0, y0, x1, y1)."""
    if box is None:
        return p
    x0, y0, x1, y1 = box
    dx, dy = q[0] - p[0], q[1] - p[1]
    ts = []
    for t in ((x0 - p[0]) / dx if dx else None, (x1 - p[0]) / dx if dx else None,
              (y0 - p[1]) / dy if dy else None, (y1 - p[1]) / dy if dy else None):
        if t is not None and t > 0:
            x, y = p[0] + t * dx, p[1] + t * dy
            if x0 - 1e-6 <= x <= x1 + 1e-6 and y0 - 1e-6 <= y <= y1 + 1e-6:
                ts.append(t)
    t = min(ts) if ts else 0
    return (p[0] + t * dx, p[1] + t * dy)

def _intersect(p, d, q, e):
    """Intersection of p + s d and q + t e (returns s) or None."""
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-9:
        return None
    return ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den


# --------------------------------------------------------------------------------------------- renderer
def draw(smi, *, template=None, rot=0.0, flip=None, coordgen=True, template_coordgen=True, tweaks=None, wedges=None, hashes=None, coords=None):
    m, P = _coords(smi, template, rot, flip, coordgen, template_coordgen, tweaks, coords)
    wedges = {(int(a), int(b)) for a, b in (wedges or [])}
    hashes = {(int(a), int(b)) for a, b in (hashes or [])}
    n = m.GetNumAtoms()
    nb = {i: [x.GetIdx() for x in m.GetAtomWithIdx(i).GetNeighbors()] for i in range(n)}
    ri = m.GetRingInfo()
    marks, texts, boxes = [], [], {}

    # ---- labels
    labeled = {}
    for a in m.GetAtoms():
        i, el = a.GetIdx(), a.GetSymbol()
        if el != 'C' or a.GetDegree() == 0 or (a.GetFormalCharge() and a.GetDegree() == 1):   # terminal C- / C+ -> H2C-
            labeled[i] = el
    for i, el in labeled.items():
        a = m.GetAtomWithIdx(i)
        x, y = P[i]
        if a.GetAtomicNum() == 0:                       # abbreviation (Ph, SiPh3, ...)
            k = a.GetIsotope()
            from_right = bool(nb[i]) and P[nb[i][0]][0] > x
            lab = ABBR[k][1] if from_right else ABBR[k][0]
            head = ABBR_HEAD[k]
            hw_ = adv(head)
            tw = sum(adv(c, FONT * (0.72 if c.isdigit() else 1)) for c in lab)
            x0 = (x + hw_ / 2 - tw) if from_right else (x - hw_ / 2)
            boxes[i] = (x - hw_ / 2 - MARGIN, y - CAP * FONT / 2 - MARGIN, x + hw_ / 2 + MARGIN, y + CAP * FONT / 2 + MARGIN)
            texts.append(f'<text x="{x0:.2f}" y="{y + CAP * FONT / 2:.2f}" font-size="{FONT:.2f}">{_sub_digits(lab, FONT)}</text>')
            continue
        w = adv(el)
        boxes[i] = (x - w / 2 - MARGIN, y - CAP * FONT / 2 - MARGIN, x + w / 2 + MARGIN, y + CAP * FONT / 2 + MARGIN)
        base = y + CAP * FONT / 2
        texts.append(f'<text x="{x - w / 2:.2f}" y="{base:.2f}" font-size="{FONT:.2f}">{el}</text>')
        h = a.GetTotalNumHs()
        hbox = None
        if h:
            sub = f'<tspan font-size="{FONT * 0.72:.2f}" dy="{FONT * 0.2:.2f}">{h}</tspan>' if h > 1 else ''
            hw = adv('H') + (adv(str(h), FONT * 0.72) if h > 1 else 0)
            R, Lf, U, Dn = 0.0, math.pi, -math.pi / 2, math.pi / 2
            d = _free_dir(P, i, nb[i], prefs=[R, Lf, Dn, U])
            if d == R:
                texts.append(f'<text x="{x + w / 2:.2f}" y="{base:.2f}" font-size="{FONT:.2f}">H{sub}</text>')
                hbox = (x + w / 2, base - CAP * FONT, x + w / 2 + hw, base)
            elif d == Lf:
                texts.append(f'<text x="{x - w / 2 - hw:.2f}" y="{base:.2f}" font-size="{FONT:.2f}">H{sub}</text>')
                hbox = (x - w / 2 - hw, base - CAP * FONT, x - w / 2, base)
            else:
                yy = base + (FONT * 1.02 if d == Dn else -FONT * 1.02)
                texts.append(f'<text x="{x - adv("H") / 2:.2f}" y="{yy:.2f}" font-size="{FONT:.2f}">H{sub}</text>')
                hbox = (x - adv('H') / 2, yy - CAP * FONT, x - adv('H') / 2 + hw, yy)
        # charge: ChemDraw puts it at the upper right; otherwise the first free corner (bonds and H avoided)
        q = a.GetFormalCharge()
        if q:
            sign = '+' if q > 0 else '−'
            busy = [_ang((P[j][0] - x, P[j][1] - y)) for j in nb[i]]
            if hbox:
                busy.append(_ang(((hbox[0] + hbox[2]) / 2 - x, (hbox[1] + hbox[3]) / 2 - y)))
            cands = [-math.pi / 4, 0, -3 * math.pi / 4, math.pi, -math.pi / 2, math.pi / 4, 3 * math.pi / 4, math.pi / 2]
            sep = lambda c: min((abs((c - b_ + math.pi) % (2 * math.pi) - math.pi) for b_ in busy), default=math.pi)
            c = next((c for c in cands if sep(c) >= math.radians(50)), max(cands, key=sep))
            cs = FONT * 0.8
            rx, ry = w / 2 + adv(sign, cs) * 0.55, CAP * FONT / 2 + cs * 0.22
            if hbox and abs(c) < math.pi / 2 and hbox[0] > x:      # H written on the right: go past it
                rx = hbox[2] - x + adv(sign, cs) * 0.55
            cx, cy = x + math.cos(c) * rx * 1.05, y + math.sin(c) * ry * 1.25
            if abs(math.sin(c)) < 0.1:                       # beside the label: superscript height, as in Br+
                cy = y - CAP * FONT * 0.42
            if hbox and hbox[0] < x < hbox[2]:              # H above/below the atom: charge rides on the H
                cx, cy = hbox[2] + adv(sign, cs) * 0.6, hbox[1] + cs * 0.12
            texts.append(f'<text x="{cx:.2f}" y="{cy + cs * 0.36:.2f}" font-size="{cs:.2f}" text-anchor="middle">{sign}</text>')
        if a.GetNumRadicalElectrons():
            d = _free_dir(P, i, nb[i])
            marks.append(f'<circle cx="{x + math.cos(d) * (w / 2 + 3.2):.2f}" cy="{y + math.sin(d) * (CAP * FONT / 2 + 3.2):.2f}" r="{0.075 * L:.2f}"/>')

    # ---- bare-vertex charges and radicals (no "C" label, ChemDraw style)
    for a in m.GetAtoms():
        i = a.GetIdx()
        if i in labeled:
            continue
        x, y = P[i]
        if a.GetFormalCharge():
            d = _free_dir(P, i, nb[i])
            sign = '+' if a.GetFormalCharge() > 0 else '−'
            cx, cy = x + math.cos(d) * 0.32 * L, y + math.sin(d) * 0.32 * L
            texts.append(f'<text x="{cx:.2f}" y="{cy + FONT * 0.28:.2f}" font-size="{FONT * 0.8:.2f}" text-anchor="middle">{sign}</text>')
        if a.GetNumRadicalElectrons():
            d = _free_dir(P, i, nb[i])
            marks.append(f'<circle cx="{x + math.cos(d) * 0.3 * L:.2f}" cy="{y + math.sin(d) * 0.3 * L:.2f}" r="{0.075 * L:.2f}"/>')

    # ---- bonds
    def seg(p, q):
        marks.append(f'<line x1="{p[0]:.2f}" y1="{p[1]:.2f}" x2="{q[0]:.2f}" y2="{q[1]:.2f}"/>')

    def wedge(p, q, hw=2.3):
        v = _unit((q[0] - p[0], q[1] - p[1]))
        nrm = (-v[1], v[0])
        p1 = (q[0] + nrm[0] * hw, q[1] + nrm[1] * hw)
        p2 = (q[0] - nrm[0] * hw, q[1] - nrm[1] * hw)
        marks.append(f'<polygon points="{p[0]:.2f},{p[1]:.2f} {p1[0]:.2f},{p1[1]:.2f} {p2[0]:.2f},{p2[1]:.2f}"/>')

    def hash_bond(p, q, hw=2.3, n_steps=5):
        v = _unit((q[0] - p[0], q[1] - p[1]))
        nrm = (-v[1], v[0])
        for k in range(1, n_steps + 1):
            t = k / n_steps
            cx = p[0] + (q[0] - p[0]) * t
            cy = p[1] + (q[1] - p[1]) * t
            w = hw * t
            marks.append(f'<line x1="{cx + nrm[0] * w:.2f}" y1="{cy + nrm[1] * w:.2f}" x2="{cx - nrm[0] * w:.2f}" y2="{cy - nrm[1] * w:.2f}"/>')

    def endpoints(i, j, off=(0.0, 0.0)):
        p = (P[i][0] + off[0], P[i][1] + off[1]); q = (P[j][0] + off[0], P[j][1] + off[1])
        return _clip(p, q, boxes.get(i)), _clip(q, p, boxes.get(j))

    def trim_to_neighbour(end, other, start_pt, dvec, side_sign, normal):
        """For a centred line at an unlabeled vertex: stop it where it meets the adjacent bond on its side."""
        best = None
        for k in nb[end]:
            if k == other:
                continue
            e = (P[k][0] - P[end][0], P[k][1] - P[end][1])
            if (e[0] * normal[0] + e[1] * normal[1]) * side_sign <= 0:
                continue
            s = _intersect(start_pt, dvec, P[end], _unit(e))
            if s is not None and abs(s) < 0.5 * L:
                best = s if best is None or abs(s) < abs(best) else best
        return best

    ring_sets = [set(r) for r in ri.AtomRings()]
    cdx_bonds = []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        order = b.GetBondTypeAsDouble()
        p, q = endpoints(i, j)
        v = _unit((P[j][0] - P[i][0], P[j][1] - P[i][1]))
        nrm = (-v[1], v[0])
        pos = 'Center'
        if order == 1:
            if (i, j) in wedges:
                wedge(p, q)
            elif (j, i) in wedges:
                wedge(q, p)
            elif (i, j) in hashes:
                hash_bond(p, q)
            elif (j, i) in hashes:
                hash_bond(q, p)
            else:
                seg(p, q)
        elif order == 2:
            rings = [r for r in ring_sets if i in r and j in r]
            di, dj = m.GetAtomWithIdx(i).GetDegree(), m.GetAtomWithIdx(j).GetDegree()
            centred = not rings and (i in labeled or j in labeled or di == 1 or dj == 1 or (di >= 3 and dj >= 3))
            if centred:
                for sgn in (1, -1):
                    off = (nrm[0] * SPACING / 2 * sgn, nrm[1] * SPACING / 2 * sgn)
                    a1, a2 = endpoints(i, j, off)
                    if i not in labeled and di > 1:
                        s = trim_to_neighbour(i, j, (P[i][0] + off[0], P[i][1] + off[1]), v, sgn, nrm)
                        if s is not None:
                            a1 = (P[i][0] + off[0] + v[0] * s, P[i][1] + off[1] + v[1] * s)
                    if j not in labeled and dj > 1:
                        s = trim_to_neighbour(j, i, (P[j][0] + off[0], P[j][1] + off[1]), v, sgn, nrm)
                        if s is not None:
                            a2 = (P[j][0] + off[0] + v[0] * s, P[j][1] + off[1] + v[1] * s)
                    seg(a1, a2)
            else:
                seg(p, q)
                if rings:
                    r = min(rings, key=len)
                    cx = sum(P[k][0] for k in r) / len(r); cy = sum(P[k][1] for k in r) / len(r)
                    side = 1 if (cx - P[i][0]) * nrm[0] + (cy - P[i][1]) * nrm[1] > 0 else -1
                    inner_angle = math.pi * (len(r) - 2) / len(r)
                else:
                    subs = [k for k in nb[i] + nb[j] if k not in (i, j)]
                    tot = sum((P[k][0] - P[i][0]) * nrm[0] + (P[k][1] - P[i][1]) * nrm[1] for k in subs)
                    side = 1 if tot > 0 else -1
                    inner_angle = 2 * math.pi / 3
                off = (nrm[0] * SPACING * side, nrm[1] * SPACING * side)
                a1, a2 = endpoints(i, j, off)
                cut = SPACING / math.tan(inner_angle / 2)
                if i not in labeled:
                    a1 = (a1[0] + v[0] * cut, a1[1] + v[1] * cut)
                if j not in labeled:
                    a2 = (a2[0] - v[0] * cut, a2[1] - v[1] * cut)
                seg(a1, a2)
                pos = 'Right' if side > 0 else 'Left'
        elif order == 3:
            seg(p, q)
            for sgn in (1, -1):
                off = (nrm[0] * SPACING * sgn, nrm[1] * SPACING * sgn)
                seg(*endpoints(i, j, off))
        cdx_bonds.append((i, j, int(order), pos))

    # ---- bounding box from marks and labels
    xs, ys, trects = [], [], []
    for s in marks:
        for a_, b_ in re.findall(r'(?:x1|x2|cx)="([-\d.]+)"\s+(?:y1|y2|cy)="([-\d.]+)"', s):
            xs.append(float(a_)); ys.append(float(b_))
        if 'points="' in s:
            for a_, b_ in re.findall(r'([-\d.]+),([-\d.]+)', s):
                xs.append(float(a_)); ys.append(float(b_))
    for t in texts:
        mt = re.search(r'x="([-\d.]+)" y="([-\d.]+)" font-size="([\d.]+)"', t)
        x, y, fs = map(float, mt.groups())
        label = re.sub(r'<[^>]+>', '', t)
        w = adv(label, fs)
        if 'text-anchor="end"' in t:
            x -= w
        elif 'text-anchor="middle"' in t:
            x -= w / 2
        xs += [x, x + w]; ys += [y - CAP * fs, y + 0.25 * fs]
        trects.append((x, y - CAP * fs, x + w, y + 0.25 * fs))
    pad = 2.5
    x0, y0, x1, y1 = min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad
    W, H = x1 - x0, y1 - y0
    body = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{W:.2f}px' height='{H:.2f}px' viewBox='{x0:.2f} {y0:.2f} {W:.2f} {H:.2f}'>"
            f'<g stroke="#000" stroke-width="{LINE:.3f}" stroke-linecap="round" fill="#000">' + ''.join(marks) + '</g>'
            f'<g font-family="{FAMILY}" fill="#000">' + ''.join(texts) + '</g></svg>')
    atoms = {i: [P[i][0] - x0, P[i][1] - y0] for i in range(n)}
    # geometry for arrow routing: bonds (atom pairs) and label rectangles, in the same frame as atoms
    geo_bonds = [[int(i), int(j)] for i, j, _o, _p in cdx_bonds]
    geo_text = [[round(r[0] - x0, 2), round(r[1] - y0, 2), round(r[2] - x0, 2), round(r[3] - y0, 2)] for r in trects]
    return body, dict(w=W, h=H, atoms=atoms, bonds=geo_bonds, text=geo_text, smiles=smi), (m, P, cdx_bonds, labeled)


# --------------------------------------------------------------------------------------------- CDXML (ChemDraw)
CDX_HEAD = ('<?xml version="1.0" encoding="UTF-8" ?>\n'
            '<!DOCTYPE CDXML SYSTEM "http://www.cambridgesoft.com/xml/cdxml.dtd" >\n'
            '<CDXML CreationProgram="MechSight figures" BondLength="14.4" LineWidth="0.6" BoldWidth="2" '
            'HashSpacing="2.5" MarginWidth="1.6" BondSpacing="18" ChainAngle="120" '
            'LabelFont="3" LabelSize="10" LabelFace="96" CaptionFont="3" CaptionSize="10" CaptionFace="0" '
            'LabelJustification="Auto" CaptionJustification="Left" color="0" bgcolor="1" '
            'ShowAtomQuery="yes" ShowAtomStereo="no" ShowBondQuery="yes" ShowBondStereo="no">'
            '<colortable><color r="1" g="1" b="1"/><color r="0" g="0" b="0"/></colortable>'
            '<fonttable><font id="3" charset="iso-8859-1" name="Arial"/></fonttable><page id="1" '
            'HeaderPosition="36" FooterPosition="36" PrintTrimMarks="yes" HeightPages="1" WidthPages="1">')
CDX_TAIL = '</page></CDXML>\n'


def cdxml_fragment(m, P, bonds, labeled, ox, oy, ids, caption=None):
    k = 14.4 / L
    out = ['<fragment id="%d">' % next(ids)]
    nid = {}
    for a in m.GetAtoms():
        i = a.GetIdx(); nid[i] = next(ids)
        x, y = ox + P[i][0] * k, oy + P[i][1] * k
        attrs = [f'id="{nid[i]}"', f'p="{x:.2f} {y:.2f}"']
        if a.GetSymbol() != 'C':
            attrs.append(f'Element="{a.GetAtomicNum()}"')
            attrs.append(f'NumHydrogens="{a.GetTotalNumHs()}"')
        if a.GetFormalCharge():
            attrs.append(f'Charge="{a.GetFormalCharge()}"')
        if a.GetNumRadicalElectrons():
            attrs.append('Radical="Doublet"')
        inner = ''
        if i in labeled:
            h = a.GetTotalNumHs()
            lab = a.GetSymbol() + ('H' if h else '') + (str(h) if h > 1 else '')
            inner = (f'<t p="{x - 3.5:.2f} {y + 3.6:.2f}" LabelJustification="Left">'
                     f'<s font="3" size="10" face="96">{escape(lab)}</s></t>')
        out.append(f'<n {" ".join(attrs)}>{inner}</n>' if inner else f'<n {" ".join(attrs)}/>')
    for i, j, order, pos in bonds:
        extra = f' Order="{order}"' if order > 1 else ''
        if order == 2:
            extra += f' DoublePosition="{pos}"'
        out.append(f'<b id="{next(ids)}" B="{nid[i]}" E="{nid[j]}"{extra}/>')
    out.append('</fragment>')
    if caption:
        ys = [P[i][1] * k for i in range(m.GetNumAtoms())]
        xs = [P[i][0] * k for i in range(m.GetNumAtoms())]
        out.append(f'<t id="{next(ids)}" p="{ox + min(xs):.2f} {oy + max(ys) + 22:.2f}"><s font="3" size="10" face="0">{escape(caption)}</s></t>')
    return ''.join(out)


def render_set(spec, outdir: Path):
    """spec: {name: {"smi", "template"?, "rot"?, "flip"?}} -> <name>.svg, <name>.cdxml, all.cdxml, meta.json"""
    outdir.mkdir(parents=True, exist_ok=True)
    meta, frags = {}, []
    import itertools
    ids_all = itertools.count(1000)
    ox = 40.0
    for name, s in spec.items():
        body, mt, (m, P, bonds, labeled) = draw(s.get('draw', s['smi']), template=tuple(s['template']) if s.get('template') else None,
                                               rot=s.get('rot', 0), flip=s.get('flip'), coordgen=s.get('coordgen', True),
                                               template_coordgen=s.get('template_coordgen', True), tweaks=s.get('tweaks'),
                                               wedges=s.get('wedges'), hashes=s.get('hashes'), coords=s.get('coords'))
        (outdir / f'{name}.svg').write_text(body)
        mt['smiles'] = s['smi']
        meta[name] = mt
        if 'draw' in s:                                  # CDXML gets the fully expanded structure
            _, _, (m, P, bonds, labeled) = draw(s['smi'], rot=s.get('rot', 0), flip=s.get('flip'))
        one = CDX_HEAD + cdxml_fragment(m, P, bonds, labeled, 100, 100, itertools.count(10)) + CDX_TAIL
        (outdir / f'{name}.cdxml').write_text(one)
        xs = [p[0] for p in P]
        frags.append(cdxml_fragment(m, P, bonds, labeled, ox - min(xs) * 14.4 / L, 160, ids_all, caption=name))
        ox += (max(xs) - min(xs)) * 14.4 / L + 50
    (outdir / 'all_structures.cdxml').write_text(CDX_HEAD + ''.join(frags) + CDX_TAIL)
    (outdir / 'meta.json').write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == '__main__':
    import sys
    spec = json.loads(Path(sys.argv[1]).read_text())
    meta = render_set(spec, Path(sys.argv[2]))
    print(json.dumps({k: (round(v['w']), round(v['h'])) for k, v in meta.items()}))
