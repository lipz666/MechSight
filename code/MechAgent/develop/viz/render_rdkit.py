"""Render one mechanism frame as an SVG: RDKit draws the structure, we draw the electrons.

RDKit owns everything that makes a structure look like chemistry — bond angles,
CH3/OH condensation, font metrics, label avoidance. It has no notion of a lone
pair or a curly arrow, so those are drawn on top, in the very same coordinate
frame: ``MolDraw2D.GetDrawCoords`` exposes the exact affine map from molecule
space to pixels, so an electron mark anchored to an atom lands on that atom.

Three rules make the arrows read like a printed mechanism rather than like
graph edges:

* An arrow starts *on the electrons it moves* — the dots of a lone pair, or the
  line of the bond — never at a floating offset from an atom centre.
* An arrow ends *on what receives them* — pointing at the attacked atom, or at
  the bond that is forming — never at the midpoint of a line that is not there
  yet.
* Everything an atom carries (lone pair, charge, PGL label) is placed in a
  direction free of bonds, of the arrows already drawn, and of the other marks,
  so nothing is drawn through a glyph.

Carbon carries its charge as a mark of ours rather than in the RDKit label, so a
carbocation reads as the usual bare vertex with a plus beside it, and a
carbanion also gets the lone pair its charge implies.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D

BOND_TYPE = {
    "single": Chem.BondType.SINGLE,
    "double": Chem.BondType.DOUBLE,
    "triple": Chem.BondType.TRIPLE,
    "aromatic": Chem.BondType.AROMATIC,
    "transition": Chem.BondType.SINGLE,
    "dative": Chem.BondType.SINGLE,
}

# Sentinel colours: RDKit only speaks RGB, the page only speaks tokens. These go
# in as literal hex and come out as CSS variables, so one render serves both themes.
HL_ATOM = (1.0, 0.0, 1.0)             # #FF00FF -> var(--accent-soft)
HL_BREAK = (1.0, 0.0, 0.0)            # #FF0000 -> var(--break-soft)

SCREEN_TOKENS = (
    ("#FF00FF", "var(--accent-soft)"),
    ("#ff00ff", "var(--accent-soft)"),
    ("#FF0000", "var(--break-soft)"),
    ("#ff0000", "var(--break-soft)"),
    ("#000000", "var(--ink)"),
)

BOND_LENGTH_PX = 46.0                 # fixed, so every frame shares one scale
RD_BOND_UNITS = 1.5                   # RDKit's own bond length in molecule units


# ── molecule construction ──────────────────────────────────────────────────

def _neighbours(state: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for b in state["bonds"]:
        out.setdefault(b["a"], []).append(b["b"])
        out.setdefault(b["b"], []).append(b["a"])
    return out


def collapsed_hydrogens(state: Dict[str, Any], keep: Sequence[str]) -> Dict[str, str]:
    """Hydrogens folded into their heavy atom's label, as {H label: heavy label}.

    A hydrogen stays explicit when this step's arrows touch it; everything else
    becomes part of an ``OH`` / ``CH3`` so the drawing reads as a structure
    rather than as a graph.
    """
    keep_set = set(keep)
    nb = _neighbours(state)
    folded: Dict[str, str] = {}
    for label, atom in state["atoms"].items():
        if atom["element"] != "H" or label in keep_set:
            continue
        partners = nb.get(label, [])
        if len(partners) == 1 and state["atoms"].get(partners[0], {}).get("element") != "H":
            folded[label] = partners[0]
    return folded


def build_mol(
    state: Dict[str, Any],
    coords: Dict[str, List[float]],
    keep: Sequence[str],
) -> Tuple[Chem.Mol, Dict[str, int], Dict[str, str], Dict[str, int]]:
    """A drawable RDKit mol for one ReactionState, on the path's shared coordinates.

    Charged carbons go to RDKit as neutral: a carbocation should read as a bare
    vertex with a plus beside it, which is what a chemist draws, not as an atom
    label ``C+``. The charge comes back as one of our own marks.
    """
    folded = collapsed_hydrogens(state, keep)
    shown = [l for l in sorted(state["atoms"]) if l not in folded and l in coords]

    rw = Chem.RWMol()
    index: Dict[str, int] = {}
    own_charge: Dict[str, int] = {}
    for label in shown:
        info = state["atoms"][label]
        charge = int(info["charge"])
        atom = Chem.Atom(info["element"])
        atom.SetNoImplicit(True)
        if info["element"] == "C" and charge:
            own_charge[label] = charge
        else:
            atom.SetFormalCharge(charge)
        atom.SetNumExplicitHs(sum(1 for h, heavy in folded.items() if heavy == label))
        index[label] = rw.AddAtom(atom)

    for b in state["bonds"]:
        if b["a"] in folded or b["b"] in folded:
            continue
        if b["a"] not in index or b["b"] not in index:
            continue
        rw.AddBond(index[b["a"]], index[b["b"]], BOND_TYPE.get(b["kind"], Chem.BondType.SINGLE))

    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for label, i in index.items():
        x, y = coords[label]
        conf.SetAtomPosition(i, (float(x), float(-y), 0.0))   # payload y grows downward
    mol.AddConformer(conf)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol, index, folded, own_charge


# ── drawing ────────────────────────────────────────────────────────────────

def _draw(mol: Chem.Mol, width: int, height: int, highlight_atoms, highlight_bonds):
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.clearBackground = False
    opts.prepareMolsBeforeDrawing = False
    opts.fixedBondLength = BOND_LENGTH_PX
    opts.bondLineWidth = 1.5
    opts.multipleBondOffset = 0.16
    opts.additionalAtomLabelPadding = 0.08
    opts.minFontSize = 11
    opts.maxFontSize = 15
    opts.highlightRadius = 0.32
    opts.highlightBondWidthMultiplier = 11
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(highlight_atoms),
        highlightAtomColors={i: HL_ATOM for i in highlight_atoms},
        highlightBonds=list(highlight_bonds),
        highlightBondColors={i: HL_BREAK for i in highlight_bonds},
    )
    drawer.FinishDrawing()
    origin = drawer.GetDrawCoords(Point2D(0.0, 0.0))
    unit = drawer.GetDrawCoords(Point2D(1.0, 0.0))
    return drawer.GetDrawingText(), (origin.x, origin.y, unit.x - origin.x)


_SVG_OPEN = re.compile(r"<svg[^>]*>", re.I)
_HEADER = re.compile(r"<\?xml.*?\?>", re.S)


def _inner(svg: str) -> str:
    body = _HEADER.sub("", svg)
    body = _SVG_OPEN.sub("", body, count=1)
    body = body.replace("</svg>", "")
    return re.sub(r"<!--.*?-->", "", body, flags=re.S).strip()


def _tokenize(svg: str, tokens) -> str:
    for literal, token in tokens:
        svg = svg.replace(literal, token)
    return svg


# ── small vector helpers ───────────────────────────────────────────────────

def _v(a, b):
    return (b[0] - a[0], b[1] - a[1])


def _norm(v):
    n = math.hypot(*v) or 1.0
    return (v[0] / n, v[1] / n)


def _add(a, v, k=1.0):
    return (a[0] + v[0] * k, a[1] + v[1] * k)


def _mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _seg_distance(p, a, b) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    wx, wy = p[0] - a[0], p[1] - a[1]
    denom = vx * vx + vy * vy
    t = 0.0 if denom == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    return math.hypot(p[0] - (a[0] + vx * t), p[1] - (a[1] + vy * t))


def _curve_points(src, ctrl, dst, n=11):
    out = []
    for i in range(1, n):
        t = i / n
        u = 1 - t
        out.append(
            (
                u * u * src[0] + 2 * u * t * ctrl[0] + t * t * dst[0],
                u * u * src[1] + 2 * u * t * ctrl[1] + t * t * dst[1],
            )
        )
    return out


def _arrow_kind(kind: str) -> str:
    text = str(kind or "")
    if "forming" in text or text == "coordination":
        return "form"
    if "breaking" in text:
        return "break"
    return "shift"


# ── the frame renderer ─────────────────────────────────────────────────────

def render_frame(
    frame: Dict[str, Any],
    coords: Dict[str, List[float]],
    canvas: Tuple[int, int],
    world: Tuple[float, float, float],
    next_state: Optional[Dict[str, Any]] = None,
    *,
    show_labels: bool = True,
    show_ghost: bool = True,
    highlight: bool = True,
    tokens: Sequence[Tuple[str, str]] = SCREEN_TOKENS,
) -> Dict[str, Any]:
    """One frame -> {svg body, focus bbox}.

    RDKit re-centres every molecule it draws, which would make the structure
    jump from step to step. Both the structure and the electron marks are
    therefore wrapped in one transform onto the path's shared world frame, so an
    atom sits at the same place in every step.
    """
    state = frame["state"]
    transfers = frame.get("transfers") or []
    atoms = state["atoms"]

    active: List[str] = []
    for t in transfers:
        for label in t["source"]["atoms"] + t["target"]["atoms"]:
            if label in atoms and label not in active:
                active.append(label)

    mol, index, folded, own_charge = build_mol(state, coords, keep=active)
    shown = set(index)
    neigh = _neighbours(state)

    def bond_key(b):
        return tuple(sorted((b["a"], b["b"])))

    breaking, forming = set(), []
    if next_state is not None:
        now = {bond_key(b) for b in state["bonds"]}
        later = {bond_key(b) for b in next_state["bonds"]}
        breaking = now - later
        forming = [k for k in later - now if k[0] in coords and k[1] in coords]

    highlight_bonds = []
    if highlight:
        for b in state["bonds"]:
            if bond_key(b) in breaking and b["a"] in index and b["b"] in index:
                bond = mol.GetBondBetweenAtoms(index[b["a"]], index[b["b"]])
                if bond is not None:
                    highlight_bonds.append(bond.GetIdx())
    highlight_atoms = [index[l] for l in active if l in index] if highlight else []

    svg, (ox, oy, scale) = _draw(mol, canvas[0], canvas[1], highlight_atoms, highlight_bonds)

    def px(label):
        x, y = coords[label]
        return (ox + x * scale, oy + y * scale)

    # a hydrogen folded into a label still needs an anchor: use its heavy atom
    def anchor(label: str) -> Optional[str]:
        return label if label in shown else folded.get(label)

    def glyph_radius(label: str) -> float:
        """Roughly how far the drawn glyph reaches from the atom centre."""
        info = atoms.get(label, {})
        element = str(info.get("element", "C"))
        if element == "C" and label not in own_charge:
            return 5.0                                       # a bare vertex
        radius = 9.0 + 2.5 * (len(element) - 1)
        if sum(1 for h, heavy in folded.items() if heavy == label):
            radius += 7.0
        if info.get("charge"):
            radius += 3.0
        return radius

    atom_points = [px(l) for l in shown]
    segments = [
        (px(b["a"]), px(b["b"]))
        for b in state["bonds"]
        if b["a"] in shown and b["b"] in shown
    ]
    occupied: List[Tuple[float, float]] = []                 # marks already placed

    def free_direction(label: str, prefer=None, spread: float = 24.0):
        """A direction off `label` that no bond, mark or arrow already uses."""
        p = px(label)
        best, best_score = (0.0, -1.0), -1e9
        for deg in range(0, 360, 10):
            rad = math.radians(deg)
            d = (math.cos(rad), math.sin(rad))
            probe = _add(p, d, spread)
            score = min(
                min((_seg_distance(probe, a, b) for a, b in segments), default=1e9),
                min((_dist(probe, o) for o in occupied), default=1e9),
                min((_dist(probe, q) for q in atom_points if _dist(q, p) > 1.0), default=1e9),
            )
            if prefer is not None:
                score += 11.0 * (d[0] * prefer[0] + d[1] * prefer[1])
            if score > best_score:
                best, best_score = d, score
        return best

    # ── lone pairs: the electrons an arrow is about to move ────────────────
    lone_pairs: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}

    def place_lone_pair(label: str, toward=None):
        host = anchor(label)
        if host is None:
            return None
        if host in lone_pairs:
            return lone_pairs[host]
        p = px(host)
        reach = glyph_radius(host) + 6.0
        prefer = None
        if toward is not None and _dist(toward, p) > 1.0:
            prefer = _norm(_v(p, toward))
        best, best_score = (0.0, -1.0), -1e9
        for deg in range(0, 360, 10):
            rad = math.radians(deg)
            d = (math.cos(rad), math.sin(rad))
            centre = _add(p, d, reach)
            probe = _add(p, d, reach + 6.0)
            score = min(
                min((_seg_distance(probe, a, b) for a, b in segments), default=1e9),
                min((_dist(probe, o) for o in occupied), default=1e9),
                min((_dist(probe, q) for q in atom_points if _dist(q, p) > 1.0), default=1e9),
            )
            if prefer is not None:
                score += 4.0 * (d[0] * prefer[0] + d[1] * prefer[1])
                # a pair parked right next to what it attacks leaves no room for
                # an arrow, so keep a drawable chord between them
                if _dist(centre, toward) < 30.0:
                    score -= 34.0
            if score > best_score:
                best, best_score = d, score
        centre = _add(p, best, reach)
        lone_pairs[host] = (centre, best)
        occupied.append(centre)
        return lone_pairs[host]

    for t in transfers:                                      # the pair that moves
        sa = [a for a in t["source"]["atoms"] if anchor(a)]
        ta = [a for a in t["target"]["atoms"] if anchor(a)]
        if len(sa) == 1 and ta:
            partner = next((a for a in ta if a != sa[0]), ta[0])
            host = anchor(partner)
            place_lone_pair(sa[0], toward=px(host) if host else None)
    for label, info in atoms.items():                        # the pair a charge implies
        if info.get("charge", 0) < 0 and label in shown:
            place_lone_pair(label)

    # ── arrows ────────────────────────────────────────────────────────────
    def on_bond(a: str, b: str, toward=None, off: float = 4.5):
        p, q = px(a), px(b)
        m = _mid(p, q)
        u = _norm(_v(p, q))
        perp = (-u[1], u[0])
        if toward is None:
            return m
        side = 1.0 if ((toward[0] - m[0]) * perp[0] + (toward[1] - m[1]) * perp[1]) >= 0 else -1.0
        return _add(m, perp, side * off)

    def at_atom(label: str, coming_from) -> Tuple[float, float]:
        """A point just outside an atom's glyph, on the side the arrow arrives from."""
        p = px(label)
        d = _norm(_v(p, coming_from)) if _dist(coming_from, p) > 1.0 else (0.0, -1.0)
        return _add(p, d, glyph_radius(label) + 6.0)

    arrows: List[str] = []
    arrow_points: List[Tuple[float, float]] = []
    for i, t in enumerate(transfers):
        sa = [a for a in t["source"]["atoms"] if anchor(a)]
        ta = [a for a in t["target"]["atoms"] if anchor(a)]
        if not sa or not ta:
            continue

        collapse = len(sa) >= 2 and len(ta) == 1 and ta[0] in sa
        if collapse:
            keeper, partner = anchor(ta[0]), anchor(next(a for a in sa if a != ta[0]))
            pk, pp = px(keeper), px(partner)
            u = _norm(_v(pk, pp))
            perp = (-u[1], u[0])
            side = 1.0
            best_clear = -1e9
            for candidate in (1.0, -1.0):
                probe = _add(_mid(pk, pp), perp, candidate * 14.0)
                clear = min(
                    min((_seg_distance(probe, a, b) for a, b in segments), default=1e9),
                    min((_dist(probe, q) for q in atom_points), default=1e9),
                )
                if clear > best_clear:
                    side, best_clear = candidate, clear
            src = _add(_add(pk, u, _dist(pk, pp) * 0.46), perp, side * 7.0)
            out = _norm((perp[0] * side * 0.85 - u[0] * 0.5, perp[1] * side * 0.85 - u[1] * 0.5))
            dst = _add(pk, out, glyph_radius(keeper) + 7.0)

        # where the electrons are now
        elif len(sa) >= 2:
            hint = px(anchor(ta[0]))
            src = on_bond(anchor(sa[0]), anchor(sa[1]), toward=hint)
        elif True:
            pair = lone_pairs.get(anchor(sa[0])) or place_lone_pair(sa[0])
            src = _add(pair[0], pair[1], 4.0)

        # where they are going
        if collapse:
            pass
        elif len(ta) >= 2:
            receiver = next((a for a in ta if a not in sa), None)
            if receiver is not None and len(sa) == 1:
                dst = at_atom(anchor(receiver), src)          # a nucleophile points at its target
            else:
                dst = on_bond(anchor(ta[0]), anchor(ta[1]), toward=src)
        else:
            keeper = anchor(ta[0])
            partner = next((a for a in sa if anchor(a) != keeper), None)
            dst = at_atom(keeper, px(anchor(partner)) if partner else src)

        chord = _dist(src, dst) or 1.0
        u = _norm(_v(src, dst))
        perp = (-u[1], u[0])
        m = _mid(src, dst)
        obstacles = atom_points + [c for c, _ in lone_pairs.values()]
        ctrl, best_score = _add(m, perp, 18.0), -1e9
        for frac in (0.20, 0.32, 0.45):
            bow = max(11.0, min(frac * chord, 55.0))
            for sign in (1.0, -1.0):
                candidate = _add(m, perp, sign * bow)
                sampled = _curve_points(src, candidate, dst)
                clear = min(
                    min(
                        min((_dist(p, o) for o in obstacles), default=1e9),
                        min((_seg_distance(p, a, b) for a, b in segments), default=1e9),
                    )
                    for p in sampled
                )
                score = clear - 0.045 * bow
                if score > best_score:
                    ctrl, best_score = candidate, score

        tail = _norm(_v(ctrl, dst))
        head_len = max(6.5, min(11.0, chord * 0.30))
        back = _add(dst, tail, -head_len)
        nperp = (-tail[1], tail[0])
        wing = head_len * 0.36
        notch = _add(dst, tail, -head_len * 0.62)
        b1 = _add(back, nperp, wing)
        b2 = _add(back, nperp, -wing)
        if int(t["electrons"]) == 1:                         # fishhook: half a head
            head = f"{dst[0]:.1f},{dst[1]:.1f} {b1[0]:.1f},{b1[1]:.1f} {notch[0]:.1f},{notch[1]:.1f}"
        else:
            head = (
                f"{dst[0]:.1f},{dst[1]:.1f} {b1[0]:.1f},{b1[1]:.1f} "
                f"{notch[0]:.1f},{notch[1]:.1f} {b2[0]:.1f},{b2[1]:.1f}"
            )
        shaft_end = _add(dst, tail, -head_len * 0.72)
        kind = _arrow_kind(t["kind"])
        delay = i * 0.16
        arrows.append(
            f'<path class="arrow {kind} draw" pathLength="100" '
            f'd="M {src[0]:.1f} {src[1]:.1f} Q {ctrl[0]:.1f} {ctrl[1]:.1f} '
            f'{shaft_end[0]:.1f} {shaft_end[1]:.1f}" style="animation-delay:{delay:.2f}s"/>'
            f'<polygon class="head {kind}" points="{head}" '
            f'style="animation-delay:{delay + 0.46:.2f}s"/>'
        )
        arrow_points.extend([src, dst] + _curve_points(src, ctrl, dst, 9))

    occupied.extend(arrow_points)

    # ── charges we own, lone-pair dots, PGL labels ────────────────────────
    marks: List[str] = []
    for label, charge in own_charge.items():
        d = free_direction(label, spread=17.0)
        p = _add(px(label), d, 12.0)
        sign = "+" if charge > 0 else "−"
        marks.append(f'<text class="charge" x="{p[0]:.1f}" y="{p[1]:.1f}">{sign}</text>')
        occupied.append(p)

    dots: List[str] = []
    for centre, d in lone_pairs.values():
        perp = (-d[1], d[0])
        for side in (1.0, -1.0):
            c = _add(centre, perp, side * 2.7)
            dots.append(f'<circle class="lp" cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="1.75"/>')

    labels: List[str] = []
    if show_labels:
        for label in active:
            host = anchor(label)
            if not host:
                continue
            d = free_direction(host, spread=26.0)
            p = _add(px(host), d, glyph_radius(host) + 15.0)
            labels.append(f'<text class="pgl" x="{p[0]:.1f}" y="{p[1]:.1f}">{label}</text>')
            occupied.append(p)

    ghosts: List[str] = []
    if show_ghost:
        for a, b in forming:
            pa, pb = anchor(a), anchor(b)
            if not pa or not pb:
                continue
            p, q = px(pa), px(pb)
            u = _norm(_v(p, q))
            s = _add(p, u, glyph_radius(pa) + 4)
            e = _add(q, u, -(glyph_radius(pb) + 4))
            ghosts.append(
                f'<line class="ghost" x1="{s[0]:.1f}" y1="{s[1]:.1f}" '
                f'x2="{e[0]:.1f}" y2="{e[1]:.1f}"/>'
            )

    focus = None
    if active:
        ring = {anchor(a) for a in active if anchor(a)}
        for label in list(ring):
            ring.update(m for m in neigh.get(label, []) if m in shown)
        pts = [px(l) for l in ring] + arrow_points
        if pts:
            focus = [
                min(p[0] for p in pts),
                min(p[1] for p in pts),
                max(p[0] for p in pts),
                max(p[1] for p in pts),
            ]

    gx, gy, gs = world
    k = gs / scale
    to_world = f"translate({gx:.2f},{gy:.2f}) scale({k:.4f}) translate({-ox:.2f},{-oy:.2f})"

    layers = f'<g class="mol-layer">{_tokenize(_inner(svg), tokens)}</g>'
    if ghosts:
        layers += f'<g class="ghost-layer">{"".join(ghosts)}</g>'
    if dots:
        layers += f'<g class="lp-layer">{"".join(dots)}</g>'
    if arrows:
        layers += f'<g class="arrow-layer">{"".join(arrows)}</g>'
    if marks or labels:
        layers += f'<g class="mark-layer">{"".join(marks)}{"".join(labels)}</g>'

    if focus:
        focus = [
            gx + (focus[0] - ox) * k,
            gy + (focus[1] - oy) * k,
            gx + (focus[2] - ox) * k,
            gy + (focus[3] - oy) * k,
        ]
    return {"svg": f'<g transform="{to_world}">{layers}</g>', "focus": focus}


def global_transform(coords: Dict[str, List[float]], pad: float = 58.0) -> Tuple[float, float, float]:
    """The one pixel frame every step of a path is normalized onto."""
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    px = BOND_LENGTH_PX / RD_BOND_UNITS
    return (pad - min(xs) * px, pad - min(ys) * px, px)


def canvas_for(coords: Dict[str, List[float]], pad: float = 58.0) -> Tuple[int, int]:
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    px = BOND_LENGTH_PX / RD_BOND_UNITS
    w = (max(xs) - min(xs)) * px + 2 * pad
    h = (max(ys) - min(ys)) * px + 2 * pad
    return int(max(w, 420)), int(max(h, 340))
