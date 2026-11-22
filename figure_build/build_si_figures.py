"""Generate 5 publication-grade multi-panel Supplementary Figures (Fig. S1 - Fig. S5) for Angew_MechSight_v1_SI.

Combines native SVG statistical charts with ACS 1996 chemical structures and electron-pushing arrows from `chemdraw.py`.
All nomenclature matches the V4 main manuscript (`FukuyamaBench`, `oMe-Gold`, `mech-USPTO-31k`,
`inert knowledge`, `blind spots`, `+ visible reference intermediates`, `+ hidden reference intermediates`).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figkit import *  # noqa

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / 'data' / 'figdata.json').read_text())
MOLS = HERE / 'si_mols' / 'mols'
META = mol_meta(MOLS)
OUT = HERE.parent / '02_LaTeX_Source' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

SUB = lambda s: f'<tspan baseline-shift="sub" font-size="8.5">{s}</tspan>'
SUP = lambda s: f'<tspan baseline-shift="super" font-size="8.5">{s}</tspan>'


def halo(F, name, idx, x, y, s, col, r=8.0, op=0.30):
    ax, ay = atom_xy(META, name, idx, x, y, s)
    F.circle(ax, ay, r, col, 'none', 0, op)


def bond_highlight(F, name, i, j, x, y, s, col, w=6.8, op=0.34):
    (x1, y1), (x2, y2) = atom_xy(META, name, i, x, y, s), atom_xy(META, name, j, x, y, s)
    F.line(x1, y1, x2, y2, col, w, cap='round', op=op)


def place_mol(F, name, cx, cy, s=1.0, pre_bonds=None):
    w, h = META[name]['w'] * s, META[name]['h'] * s
    x, y = cx - w / 2, cy - h / 2
    if pre_bonds:
        for i, j, col in pre_bonds:
            bond_highlight(F, name, i, j, x, y, s, col)
    F.mol(MOLS, name, x, y, META, s)
    return x, y, w, h


def build_figS1():
    """Fig. S1: Four-pillar origin-of-competence probes (M3 active-site blocking, M2 citation exposure,
    M1/M4 memory vs evidence override, and homologue/random-SMILES invariance; §S6, Tables S5-S8).
    """
    F = Fig(1000, 660)
    m2 = D['fig1']['m2']

    # =========================================================================
    # a | Counterfactual active-site blocking (n = 84 triplets: structures + subgroup bars)
    # =========================================================================
    F.letter(0, 18, 'a')
    F.text(24, 18, 'Counterfactual active-site blocking (n = 84 triplets, 504 trajectories)', 12, INK, 'start', 'bold')

    # Left: 3 ChemDraw triplet structures (Parent, Remote-preserving, Active-site-blocked)
    F.rect(16, 30, 184, 250, '#F8FAFC', GRID, 0.9, 6)
    S_m3 = 0.72
    cards = [
        ('m3_parent', 'Parent (intact)', '88.1%', NAVY, 1, None),
        ('m3_preserve', 'Remote edit (≥3 bonds)', '81.0%', BLUE_D, 5, ('preserve', 0)),
        ('m3_blocked', 'Site blocked (H→Me)', '9.5%', RED, 1, ('blocked', 0, 2)),
    ]
    for idx, (mname, mlab, mpct, mcol, ksite, extra) in enumerate(cards):
        cy = 76 + idx * 80
        F.text(24, cy - 30, mlab, 9.5, mcol, 'start', 'bold')
        F.text(192, cy - 30, mpct, 10.0, mcol, 'end', 'bold')
        mx, my, mw, mh = place_mol(F, mname, 90, cy + 12, S_m3)
        halo(F, mname, ksite, mx, my, S_m3, AMBER if extra is None or extra[0] == 'preserve' else RED, 6.8, 0.30)
        if extra and extra[0] == 'preserve':
            halo(F, mname, extra[1], mx, my, S_m3, BLUE, 6.8, 0.35)
            rx, ry = atom_xy(META, mname, ksite, mx, my, S_m3)
            F.check(rx + 28, ry + 6, TEAL, 4.5, 1.8)
        elif extra is None:
            rx, ry = atom_xy(META, mname, ksite, mx, my, S_m3)
            F.check(rx + 28, ry + 6, TEAL, 4.5, 1.8)
        elif extra and extra[0] == 'blocked':
            halo(F, mname, extra[1], mx, my, S_m3, RED, 5.8, 0.25)
            halo(F, mname, extra[2], mx, my, S_m3, RED, 5.8, 0.25)
            rx, ry = atom_xy(META, mname, ksite, mx, my, S_m3)
            F.cross(rx + 28, ry + 6, RED, 4.5, 1.8)

    # Right of Panel a: Subgroup bars
    X0, X1, Y0, Y1 = 238, 508, 62, 224
    Y = F.yaxis(X0, X1, Y0, Y1, 0, 100, (0, 25, 50, 75, 100), 'Target step adopted (%)', lx=28)
    subgroups = [
        ('All triplets', 'n = 84', 88.1, 81.0, 9.5, -71.4),
        ('Non-PubChem', 'n = 23', 89.1, 84.8, 6.5, -78.3),
        ('H → Me', 'n = 63', 88.9, 85.7, 8.7, -77.0),
        ('Deleted', 'n = 21', 85.7, 66.7, 11.9, -54.8),
    ]
    bw = 16
    for gi, (glab, gsub, vp, vpr, vbr, dval) in enumerate(subgroups):
        gx = X0 + 6 + gi * 67
        for ai, (v, col) in enumerate(((vp, NAVY), (vpr, BLUE), (vbr, RED))):
            bx = gx + ai * (bw + 2)
            F.bar(bx, Y1, v, None, None, Y, bw, col, ci=False)
            F.text(bx + bw / 2, Y(v) - 4, f'{v:.0f}', 9.0, col if v < 25 else INK, 'middle', 'bold')
        F.text(gx + 26, Y1 + 14, glab, 9.0, INK, 'middle', 'bold')
        F.text(gx + 26, Y1 + 25, gsub, 8.8, MUTED, 'middle')
        F.text(gx + 26, Y0 - 5, f'{dval:.1f}'.replace('-', '−'), 9.5, RED, 'middle', 'bold')

    F.rect(X0, 32, 9, 9, NAVY, 'none', 0, 2); F.text(X0 + 13, 40, 'Parent', 9.5, MUTED)
    F.rect(X0 + 62, 32, 9, 9, BLUE, 'none', 0, 2); F.text(X0 + 75, 40, 'Remote edit', 9.5, MUTED)
    F.rect(X0 + 150, 32, 9, 9, RED, 'none', 0, 2); F.text(X0 + 163, 40, 'Site blocked (Δ pp)', 9.5, RED, 'start', 'bold')

    # Bottom strip of Panel a: Fate of 168 blocked trajectories
    fy = 264
    fw = 270
    F.rect(X0, fy, fw * 0.899, 14, TEAL, 'none', 0, 2)
    F.rect(X0 + fw * 0.899, fy, fw * 0.065, 14, AMBER, 'none', 0, 0)
    F.rect(X0 + fw * 0.964, fy, fw * 0.036, 14, RED, 'none', 0, 2)
    F.text(X0 + 6, fy + 10.5, '89.9% redirected (151/168)', 9.2, '#fff', 'start', 'bold')
    F.text(X0 + fw, fy - 4, 'ghost structures: 3.6% (6/168)', 9.0, RED, 'end', 'bold')

    # =========================================================================
    # b | Citation-exposure stratification (N = 409 full-text papers)
    # =========================================================================
    BX = 532
    F.letter(BX, 18, 'b')
    F.text(BX + 24, 18, 'Citation-exposure stratification (N = 409 papers)', 12, INK, 'start', 'bold')

    X0b, X1b, Y0b, Y1b = BX + 44, 984, 62, 224
    Yb = F.yaxis(X0b, X1b, Y0b, Y1b, 0, 60, (0, 20, 40, 60), 'intermediate recall (%)', lx=28)
    strata_m2 = [
        ('Low cites', 'n = 85', m2['low']['Direct'], m2['low']['Auto']),
        ('Mid cites', 'n = 83', m2['mid']['Direct'], m2['mid']['Auto']),
        ('High cites', 'n = 76', m2['high']['Direct'], m2['high']['Auto']),
        ('2026 papers', 'n = 165', m2['new2026']['Direct'], m2['new2026']['Auto']),
    ]
    xs_m2 = {}
    for gi, (glab, gsub, vd, va) in enumerate(strata_m2):
        gx = X0b + 16 + gi * 100
        xs_m2[gi] = gx + 13
        F.bar(gx, Y1b, vd, None, None, Yb, 26, GRAY, ci=False)
        F.bar(gx + 29, Y1b, va, None, None, Yb, 26, TEAL, ci=False)
        F.text(gx + 13, Y1b - 6, f'{vd:.1f}', 9.5, '#fff', 'middle', 'bold')
        F.text(gx + 42, Y1b - 6, f'{va:.1f}', 9.5, '#fff', 'middle', 'bold')
        F.text(gx + 27, Y1b + 14, glab, 9.5, INK, 'middle', 'bold')
        F.text(gx + 27, Y1b + 25, gsub, 9.0, MUTED, 'middle')

    # Equivalence bracket between High cites and 2026 Post-cutoff
    top_eq = Yb(50)
    F.path(f'M{xs_m2[2]},{top_eq + 6} L{xs_m2[2]},{top_eq} L{xs_m2[3]},{top_eq} L{xs_m2[3]},{top_eq + 6}', TEAL, 1.2)
    F.text((xs_m2[2] + xs_m2[3]) / 2, top_eq - 5, 'Adjusted Δ = +0.1 pp [−8.9, 9.2] (±10 pp equiv.)', 9.5, TEAL, 'middle', 'bold')

    F.rect(X0b, 32, 9, 9, GRAY, 'none', 0, 2); F.text(X0b + 13, 40, 'Direct (slope +8.7 pp/10× cites, n.s.)', 9.5, MUTED)
    F.rect(X0b + 216, 32, 9, 9, TEAL, 'none', 0, 2); F.text(X0b + 229, 40, 'MechSight-Auto (+5.4 to +6.1 pp)', 9.5, TEAL, 'start', 'bold')

    # =========================================================================
    # c | Bibliographic recall vs. citation cueing (N = 409) & Contradictory evidence override (n = 250)
    # =========================================================================
    TOPC = 322
    F.line(14, TOPC - 18, 986, TOPC - 18, GRID, 0.8)
    F.letter(0, TOPC, 'c')
    F.text(24, TOPC, 'Bibliographic recall vs. citation cueing and counterfactual evidence override', 12, INK, 'start', 'bold')

    # Left subpanel: M1 Citation memory R1 vs Memory index C - A
    X0c1, X1c1, Y0c1, Y1c1 = 48, 250, TOPC + 42, TOPC + 206
    Yc1 = F.yaxis(X0c1, X1c1, Y0c1, Y1c1, -4, 24, (0, 8, 16, 24), 'rate / Δ recall (%)', lx=26)
    F.line(X0c1, Yc1(0), X1c1, Yc1(0), FAINT, 0.8, '2 2')
    m1_pts = [
        ('2026', 0.0, 1.0),
        ('Low', 6.5, 3.0),
        ('Mid', 9.6, -0.1),
        ('High', 16.5, 3.3),
    ]
    px_m1 = [X0c1 + 22 + i * 52 for i in range(4)]
    pts_r1 = [(px_m1[i], Yc1(m1_pts[i][1])) for i in range(4)]
    pts_ca = [(px_m1[i], Yc1(m1_pts[i][2])) for i in range(4)]
    F.path('M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in pts_r1), NAVY, 1.8)
    F.path('M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in pts_ca), TEAL, 1.8, dash='4 3')
    for i, (lab, r1, ca) in enumerate(m1_pts):
        x = px_m1[i]
        F.circle(x, Yc1(r1), 4.0, NAVY, '#fff', 0.9)
        F.diamond(x, Yc1(ca), 3.8, TEAL_L, TEAL, 1.2)
        if i == 0:
            F.text(x, Yc1(r1) + 14, f'{r1:.1f}', 9.0, NAVY, 'middle', 'bold')
            F.text(x, Yc1(ca) - 8, f'{ca:+.1f}', 9.0, TEAL, 'middle', 'bold')
        else:
            F.text(x, Yc1(r1) - 7, f'{r1:.1f}', 9.2, NAVY, 'middle', 'bold')
            F.text(x, Yc1(ca) + 14, f'{ca:+.1f}'.replace('-', '−'), 9.0, TEAL, 'middle', 'bold')
        F.text(x, Y1c1 + 15, lab, 9.5, INK, 'middle', 'bold')
    F.text((X0c1 + X1c1) / 2, Y1c1 + 28, 'Citation stratum (N = 409)', 9.2, MUTED, 'middle')
    F.circle(X0c1 + 6, TOPC + 15, 3.8, NAVY, '#fff', 0.8); F.text(X0c1 + 14, TOPC + 19, 'Bibliographic recall (+11.8 pp/10×)', 9.2, NAVY, 'start', 'bold')
    F.diamond(X0c1 + 6, TOPC + 28, 3.5, TEAL_L, TEAL, 1.1); F.text(X0c1 + 14, TOPC + 32, 'Citation-cueing gain (−0.5 pp/10×)', 9.2, TEAL, 'start', 'bold')

    # Right subpanel of c: M4 Contradictory polar evidence override on n = 250 radical papers
    X0c2, X1c2, Y0c2, Y1c2 = 298, 496, TOPC + 42, TOPC + 206
    Yc2 = F.yaxis(X0c2, X1c2, Y0c2, Y1c2, 0, 100, (0, 25, 50, 75, 100), 'polar pathway (%)', lx=26)
    m4_arms = [
        ('No clues', 'Direct', 20.1, GRAY),
        ('Radical clues', 'TEMPO/clock', 1.6, BLUE),
        ('Polar clues', 'Contradictory', 86.2, PURPLE),
    ]
    xs_m4 = []
    for i, (l1, l2, v, col) in enumerate(m4_arms):
        bx = X0c2 + 16 + i * 60
        xs_m4.append(bx + 18)
        F.bar(bx, Y1c2, v, None, None, Yc2, 36, col, ci=False)
        if v > 70:
            F.text(bx + 18, Yc2(v) + 15, f'{v:.1f}%', 9.5, '#fff', 'middle', 'bold')
        else:
            F.text(bx + 18, Yc2(v) - 5, f'{v:.1f}%', 9.5, col if v < 10 else INK, 'middle', 'bold')
        F.text(bx + 18, Y1c2 + 14, l1, 9.5, INK, 'middle', 'bold')
        F.text(bx + 18, Y1c2 + 25, l2, 8.8, MUTED, 'middle')
    top_m4 = Yc2(94)
    F.path(f'M{xs_m4[1]},{top_m4 + 6} L{xs_m4[1]},{top_m4} L{xs_m4[2]},{top_m4} L{xs_m4[2]},{top_m4 + 6}', PURPLE, 1.2)
    F.text((xs_m4[1] + xs_m4[2]) / 2, top_m4 - 5, '+84.6 pp override (n = 250)', 9.5, PURPLE, 'middle', 'bold')

    # =========================================================================
    # d | Homologue perturbation & randomized SMILES invariance (Table S8)
    # =========================================================================
    BX = 532
    F.letter(BX, TOPC, 'd')
    F.text(BX + 24, TOPC, 'Remote +CH3 homologues & randomized SMILES invariance', 12, INK, 'start', 'bold')

    # Small ChemDraw inset showing Parent vs Remote +CH3 homologue (>=3 bonds)
    F.rect(BX + 14, TOPC + 28, 122, 198, '#F8FAFC', GRID, 0.9, 6)
    F.text(BX + 75, TOPC + 44, 'Remote +CH' + SUB('3') + ' edit', 9.5, INK, 'middle', 'bold')
    mx1, my1, _, _ = place_mol(F, 'hom_parent', BX + 75, TOPC + 82, 0.66)
    halo(F, 'hom_parent', 1, mx1, my1, 0.66, AMBER, 6.5, 0.30)
    F.text(BX + 75, TOPC + 124, '↓ ≥ 3 bonds away', 8.8, BLUE_D, 'middle', 'bold')
    mx2, my2, _, _ = place_mol(F, 'hom_ch3', BX + 75, TOPC + 168, 0.66)
    halo(F, 'hom_ch3', 1, mx2, my2, 0.66, AMBER, 6.5, 0.30)
    halo(F, 'hom_ch3', 7, mx2, my2, 0.66, BLUE, 7.0, 0.38)

    X0d, X1d, Y0d, Y1d = BX + 170, 994, TOPC + 42, TOPC + 200
    Yd = F.yaxis(X0d, X1d, Y0d, Y1d, 0, 85, (0, 25, 50, 75), 'accuracy / recall (%)', lx=26)
    probes = [
        ('+CH3 prod', 'FukuyamaBench', 'n = 109', 39.4, 37.6, '+1.8 pp'),
        ('+CH3 prod', 'oMe-Gold', 'n = 133', 68.4, 63.2, '+5.3 pp'),
        ('+CH3 hid', 'FukuyamaBench', 'n = 74', 71.8, 73.2, '−1.4 pp'),
        ('Rand SMILES', 'FukuyamaBench', 'n = 147', 51.0, 47.6, '+3.4 pp'),
        ('5× Rand', 'FukuyamaBench', 'n = 48', 27.1, 25.1, '+2.0 pp'),
    ]
    for gi, (glab, gset, gn, v_can, v_pert, dstr) in enumerate(probes):
        gx = X0d + 8 + gi * 57
        F.bar(gx, Y1d, v_can, None, None, Yd, 19, NAVY, ci=False)
        F.bar(gx + 21, Y1d, v_pert, None, None, Yd, 19, BLUE_L, BLUE, 1.1, ci=False)
        F.text(gx + 20, Yd(max(v_can, v_pert)) - 5, dstr, 8.8, BLUE_D, 'middle', 'bold')
        F.text(gx + 20, Y1d + 13, glab, 8.2, INK, 'middle', 'bold')
        F.text(gx + 20, Y1d + 24, gset, 6.9, MUTED, 'middle')
        F.text(gx + 20, Y1d + 34, gn, 7.6, MUTED, 'middle')

    F.rect(X0d, TOPC + 14, 9, 9, NAVY, 'none', 0, 2); F.text(X0d + 13, TOPC + 22, 'Canonical / Parent', 9.2, MUTED)
    F.rect(X0d + 124, TOPC + 14, 9, 9, BLUE_L, BLUE, 1.1, 2); F.text(X0d + 137, TOPC + 22, 'Homologue / Permuted SMILES', 9.2, BLUE_D)

    return F.save(OUT / 'figS1_origin_probes.svg')


def build_figS2():
    """Fig. S2: Step-length controls and single-step elementary granularity audit with ChemDraw scheme (§S4, Table S4)."""
    F = Fig(1000, 370)

    # =========================================================================
    # a | Step-length controlled complete-pathway accuracy
    # =========================================================================
    F.letter(0, 18, 'a')
    F.text(24, 18, 'Step-length controlled pathway accuracy by hidden intermediates', 12, INK, 'start', 'bold')

    X0, X1, Y0, Y1 = 54, 514, 56, 286
    Y = F.yaxis(X0, X1, Y0, Y1, 0, 108, (0, 25, 50, 75, 100), 'Direct complete pathway (%)', lx=30)
    groups = [
        ('2 steps', '0 hid (n=35)', 80.0, NAVY),
        ('3 steps', '0 hid (n=13)', 76.9, NAVY),
        ('3 steps', '1–2 hid (n=39)', 48.7, BLUE),
        ('4–6 steps', '0 hid (n=30)', 100.0, TEAL),
        ('4–6 steps', '1–2 hid (n=45)', 57.8, AMBER),
        ('4–6 steps', '≥3 hid (n=8)', 12.5, RED),
        ('≥7 steps', '≥3 hid (n=33)', 21.2, RED),
    ]
    bw = 36
    for gi, (glab, gsub, val, col) in enumerate(groups):
        gx = X0 + 12 + gi * 64
        F.bar(gx, Y1, val, None, None, Y, bw, col, ci=False)
        F.text(gx + bw / 2, Y(val) - 5, f'{val:.1f}%', 9.5, col, 'middle', 'bold')
        F.text(gx + bw / 2, Y1 + 15, glab, 9.5, INK, 'middle', 'bold')
        F.text(gx + bw / 2, Y1 + 27, gsub, 8.8, MUTED, 'middle')

    # Highlight box & bracket around fixed 4-6 steps
    x_46_0 = X0 + 12 + 3 * 64 + bw / 2
    x_46_3 = X0 + 12 + 5 * 64 + bw / 2
    top_b = Y(100.0) - 16
    F.path(f'M{x_46_0},{top_b + 5} L{x_46_0},{top_b} L{x_46_3},{top_b} L{x_46_3},{top_b + 5}', RED, 1.3)
    F.text((x_46_0 + x_46_3) / 2, top_b - 5, 'Fixed 4–6 steps: −87.5 pp (30/30 → 1/8)', 9.8, RED, 'middle', 'bold')

    # =========================================================================
    # b | Single-step granularity audit (n = 115 steps) + ChemDraw +/-H+ merge illustration
    # =========================================================================
    BX = 548
    F.letter(BX, 18, 'b')
    F.text(BX + 24, 18, 'Single-step audit given reference S' + SUB('k−1') + ' (n = 115 steps)', 12, INK, 'start', 'bold')

    # Top stacked summary bar (single-pass n = 115: 84 + 25 + 6 = 115 = 100%)
    sx0, sy0, sw0, sh0 = BX + 14, 48, 424, 21
    w_ex = sw0 * (84.0 / 115.0)
    w_gr = sw0 * (25.0 / 115.0)
    w_er = sw0 - w_ex - w_gr
    F.text(sx0, sy0 - 8, 'Single-pass (n = 115):', 9.0, MUTED, 'start', 'bold')
    F.text(sx0 + sw0, sy0 - 8, '94.8% (109/115) chemically valid · 5.2% (6) error', 9.2, TEAL, 'end', 'bold')
    F.rect(sx0, sy0, w_ex - 1, sh0, TEAL, 'none', 0, 3)
    F.rect(sx0 + w_ex, sy0, w_gr - 1, sh0, BLUE, 'none', 0, 0)
    F.rect(sx0 + w_ex + w_gr, sy0, w_er, sh0, RED, 'none', 0, 3)
    F.text(sx0 + w_ex / 2, sy0 + 14.2, 'Exact match: 73.0% (84/115)', 9.5, '#fff', 'middle', 'bold')
    F.text(sx0 + w_ex + w_gr / 2, sy0 + 14.2, '21.7% (25)', 8.8, '#fff', 'middle', 'bold')

    # Clearly separated breakdown of the 43 pooled non-matching answers (2 runs: single-pass + 5-cand. select, sum = 43/43)
    F.text(sx0, 88, 'Breakdown of 43 non-matching answers (2 runs pooled, sum = 43/43):', 9.1, NAVY, 'start', 'bold')
    cats_nonmatch = [
        ('±H' + SUP('+') + ' proton-transfer merge', 16, 37.2, BLUE),
        ('Stepwise / concerted split', 11, 25.6, '#26A69A'),
        ('Symmetry / valid alternative', 7, 16.3, AMBER),
        ('Ambiguous event ordering', 2, 4.7, SLATE),
        ('True chemical / site error', 7, 16.3, RED),
    ]
    hx0, hx1 = BX + 178, 916
    xv = lambda v: hx0 + v / 45.0 * (hx1 - hx0)
    ry0 = 104
    for i, (lab, cnt, pct, col) in enumerate(cats_nonmatch):
        y = ry0 + i * 21.5
        F.text(hx0 - 8, y + 4, lab, 9.2, INK, 'end', 'bold' if i == 4 else 'normal')
        w = max(4.0, xv(pct) - hx0)
        F.rect(hx0, y - 5.5, w, 12.5, col, 'none', 0, 2)
        F.text(hx0 + w + 6, y + 4.5, f'{pct:.1f}% ({cnt}/43)', 8.9, col, 'start', 'bold')

    # Bottom ChemDraw box: visual illustration of acceptable ±H+ granularity merge
    cy_box = 216
    F.rect(BX + 14, cy_box, 424, 118, '#F8FAFC', GRID, 1.0, 6)
    F.text(BX + 26, cy_box + 17, 'Example of acceptable granularity (21.7%): ±H' + SUP('+') + ' proton-transfer bundling', 9.8, BLUE_D, 'start', 'bold')

    cy_m = cy_box + 72
    S_g = 0.85
    place_mol(F, 'gran_ketone', BX + 62, cy_m, S_g)
    F.text(BX + 62, cy_box + 108, ' S' + SUB('k−1') + ' (ketone)', 9.2, MUTED, 'middle')
    F.line(BX + 92, cy_m, BX + 150, cy_m, INK, 1.1, marker='aInk')
    F.text(BX + 121, cy_m - 7, '+ MeOH', 9.0, MUTED, 'middle')
    F.rect(BX + 158, cy_m - 26, 68, 52, AMBER_L, AMBER, 1.0, 5)
    place_mol(F, 'gran_alkoxide', BX + 192, cy_m, S_g)
    F.text(BX + 192, cy_box + 108, 'Ref. S' + SUB('k') + ' (alkoxide)', 9.2, AMBER, 'middle', 'bold')
    F.line(BX + 232, cy_m, BX + 286, cy_m, INK, 1.1, marker='aInk')
    F.text(BX + 259, cy_m - 7, '±H' + SUP('+'), 9.2, MUTED, 'middle', 'bold')
    F.rect(BX + 294, cy_m - 26, 72, 52, TEAL_L, TEAL, 1.1, 5)
    place_mol(F, 'gran_hemiacetal', BX + 330, cy_m, S_g)
    F.text(BX + 330, cy_box + 108, 'Hemiacetal (same σ-key)', 9.2, TEAL, 'middle', 'bold')
    F.check(BX + 382, cy_m, TEAL, 5.5, 2.0)
    F.text(BX + 408, cy_m + 4, 'Valid', 9.8, TEAL, 'middle', 'bold')

    # Curved top arrow showing LLM 1-step ±H+ bundle
    F.path(f'M{BX + 78},{cy_m - 22} Q{BX + 204},{cy_m - 52} {BX + 314},{cy_m - 26}', BLUE_D, 1.3, dash='4 2.5', marker='aInk')
    F.text(BX + 204, cy_m - 40, 'LLM 1-step ±H' + SUP('+') + ' bundle', 9.2, BLUE_D, 'middle', 'bold')

    return F.save(OUT / 'figS2_length_and_audit.svg')


def build_figS3():
    """Fig. S3: Falsified alternative paradigms, internal tracking of 251 missed hidden intermediates,
    and 8-family structural bifurcation with ChemDraw exemplars (§S5 & §S7, Tables S5 & S11).
    """
    F = Fig(1000, 450)

    # =========================================================================
    # a | Five falsified alternative paradigms vs. MechSight-Auto on Fukuyama-Test
    # =========================================================================
    F.letter(0, 18, 'a')
    F.text(24, 18, 'Falsified alternative paradigms vs. MechSight-Auto', 12, INK, 'start', 'bold')

    paradigms = [
        ('1. Mayr N/E & pKa scales', 'Numeric hallucination', -50.0, RED),
        ('2. Single-arrow tree search', '91% steps need ≥2 arrows', -6.7, RED),
        ('3. Enumerate-shortlist', 'Premature pruning', -0.8, '#E67E22'),
        ('4. Pairwise pathway judge', 'Favours shortcuts (33%)', +1.7, GRAY),
        ('5. Listwise pool judge', 'No cross-path splicing', +4.6, '#78909C'),
        ('MechSight-Auto', 'Δbond ≥ 3 + RDKit splicing', +18.8, TEAL),
    ]
    zx = 290
    ry0 = 56
    F.line(zx, ry0 - 12, zx, ry0 + 6 * 56 - 10, INK, 1.0)
    F.text(zx, ry0 - 18, 'Δ complete pathway vs. Direct (pp)', 9.5, MUTED, 'middle')
    for i, (lab, sub, dv, col) in enumerate(paradigms):
        y = ry0 + i * 56
        F.text(20, y + 2, lab, 10, TEAL if i == 5 else INK, 'start', 'bold' if i == 5 else 'normal')
        F.text(20, y + 15, sub, 9.0, MUTED, 'start')
        if dv < 0:
            w = min(78, max(6, abs(dv) * 1.5))
            F.rect(zx - w, y - 8, w, 20, '#FADBD8', RED, 1.0, 3)
            F.text(zx - w - 5, y + 6, f'{dv:+.1f} pp'.replace('-', '−'), 9.5, RED, 'end', 'bold')
        else:
            w = max(8, dv * 4.8)
            F.rect(zx, y - 8, w, 20, col, 'none', 0, 3)
            F.text(zx + w + 6, y + 6, f'+{dv:.1f} pp', 10 if i == 5 else 9.5, col, 'start', 'bold')

    # =========================================================================
    # b | Compact 2-tier flow of 251 omitted hidden intermediates
    # =========================================================================
    BX = 456
    F.letter(BX, 18, 'b')
    F.text(BX + 24, 18, 'Internal tracking of all 251 omitted hidden intermediates (n = 242 reactions)', 12, INK, 'start', 'bold')

    bx0, by0, bw0, bh0 = BX + 16, 36, 512, 24
    w_in = bw0 * (129 / 251)
    w_nw = bw0 * (2 / 251)
    w_bl = bw0 * (120 / 251)
    F.rect(bx0, by0, w_in - 1.5, bh0, TEAL, 'none', 0, 4)
    F.rect(bx0 + w_in, by0, w_nw, bh0, AMBER, 'none', 0, 0)
    F.rect(bx0 + w_in + w_nw + 1.5, by0, w_bl - 1.5, bh0, PURPLE, 'none', 0, 4)
    F.text(bx0 + w_in / 2, by0 + 16, 'Inert knowledge in pool: 129 / 251 (51.4%)', 9.8, '#fff', 'middle', 'bold')
    F.text(bx0 + w_in + w_nw + w_bl / 2, by0 + 16, 'Blind spots: 120 / 251 (47.8%)', 9.8, '#fff', 'middle', 'bold')
    F.text(bx0 + w_in + 2, by0 - 4, '2 (0.8%) gap completion', 8.8, AMBER, 'middle', 'bold')

    # Second compact bar: Recovery outcome (split labels cleanly on dark vs light segments)
    by2 = by0 + 32
    w_rec = w_in * (77 / 129)
    F.rect(bx0, by2, w_in, 22, TEAL_L, TEAL, 1.0, 4)
    F.rect(bx0, by2, w_rec, 22, TEAL, 'none', 0, 4)
    F.text(bx0 + w_rec / 2, by2 + 15, 'Auto splices 77/129 (59.7%)', 9.0, '#fff', 'middle', 'bold')
    F.text(bx0 + w_rec + (w_in - w_rec) / 2, by2 + 15, '52 unrecovered', 8.8, TEAL, 'middle', 'bold')

    bx_bl = bx0 + w_in + w_nw + 1.5
    F.rect(bx_bl, by2, w_bl - 1.5, 22, PURPLE_L, PURPLE, 1.0, 4)
    F.text(bx_bl + (w_bl - 1.5) / 2, by2 + 15, 'Auto: 0 / 120 (0.0%) — requires experimental clues', 8.8, PURPLE, 'middle', 'bold')

    # =========================================================================
    # c | Chemical-family bifurcation across the 251 omitted hidden species + ChemDraw exemplars
    # =========================================================================
    TOPC = 120
    F.letter(BX, TOPC, 'c')
    F.text(BX + 24, TOPC, 'Chemical-family bifurcation of the 251 omitted hidden intermediates', 12, INK, 'start', 'bold')

    fams = [
        ('Zwitterions / ylides / betaines', 12, 91.7, 66.7, 8.3),
        ('3-Membered halonium / onium', 12, 75.0, 58.3, 25.0),
        ('Tetrahedral carbonyl adducts', 15, 66.7, 33.3, 33.3),
        ('Neutral cyclic / tautomer', 70, 57.1, 25.7, 42.9),
        ('Oxocarbenium / iminium ions', 56, 50.0, 26.8, 50.0),
        ('Organometallic / main-group', 70, 37.1, 27.1, 60.0),
        ('Rearranging carbocations', 8, 37.5, 37.5, 62.5),
        ('Open-shell radical / SET', 8, 25.0, 25.0, 75.0),
    ]
    fx0, fw_bar = BX + 178, 174
    fy0 = TOPC + 28
    F.text(fx0, fy0 - 8, '← Inert (%)', 9.0, TEAL, 'start', 'bold')
    F.text(fx0 + fw_bar, fy0 - 8, 'Blind spot (%) →', 9.0, PURPLE, 'end', 'bold')

    for i, (flab, n_f, p_in, p_rec, p_bl) in enumerate(fams):
        y = fy0 + i * 36
        p_gap = max(0.0, 100.0 - p_in - p_bl)
        wi = fw_bar * (p_in / 100.0)
        wg = fw_bar * (p_gap / 100.0)
        wb = fw_bar * (p_bl / 100.0)
        wrec = fw_bar * (p_rec / 100.0)
        F.text(fx0 - 8, y + 11, f'{flab} ({n_f})', 9.0, INK, 'end', 'bold' if i in (0, 1, 5, 7) else 'normal')
        F.rect(fx0, y, wi - (0.5 if wg > 0.5 else 1.0), 16, TEAL_L, TEAL, 0.9, 2)
        if wrec > 1:
            F.rect(fx0, y, wrec, 16, TEAL, 'none', 0, 2)
        if wg > 0.5:
            F.rect(fx0 + wi, y, wg, 16, AMBER, 'none', 0, 0)
        F.rect(fx0 + wi + wg, y, wb, 16, PURPLE_L, PURPLE, 0.9, 2)
        F.text(fx0 + 4, y + 12, f'{p_in:.0f}%', 8.8, '#fff' if p_rec >= 25 else TEAL, 'start', 'bold')
        if p_bl >= 20:
            F.text(fx0 + fw_bar - 4, y + 12, f'{p_bl:.0f}%', 8.8, PURPLE, 'end', 'bold')

    # Right column of Panel c: ChemDraw exemplar structures for Inert vs Blind Spot regimes
    ex_x = BX + 362
    # Top card: Polar shortcut exemplars (Inert knowledge -> MechSight-Auto)
    F.rect(ex_x, fy0 - 4, 168, 138, '#F8FAFC', TEAL, 1.1, 6)
    F.text(ex_x + 84, fy0 + 12, 'Polar shortcuts (Inert → Auto)', 9.5, TEAL, 'middle', 'bold')
    mx_b, my_b, _, _ = place_mol(F, 'fam_bromonium', ex_x + 44, fy0 + 66, 0.78)
    halo(F, 'fam_bromonium', 3, mx_b, my_b, 0.78, AMBER, 7.5, 0.32)
    F.text(ex_x + 44, fy0 + 118, 'Halonium (75%)', 8.5, INK, 'middle')

    mx_y, my_y, _, _ = place_mol(F, 'fam_betaine', ex_x + 124, fy0 + 66, 0.74)
    halo(F, 'fam_betaine', 2, mx_y, my_y, 0.74, AMBER, 6.5, 0.30)
    halo(F, 'fam_betaine', 4, mx_y, my_y, 0.74, TEAL, 6.5, 0.28)
    F.text(ex_x + 124, fy0 + 118, 'Betaine (92%)', 8.5, INK, 'middle')

    # Bottom card: Open-shell / organometallic exemplars (Blind spots -> MechSight-Guided)
    by_card2 = fy0 + 146
    F.rect(ex_x, by_card2, 168, 138, '#FDF7F5', PURPLE, 1.1, 6)
    F.text(ex_x + 84, by_card2 + 16, 'Open-shell / coord. (Guided)', 9.5, PURPLE, 'middle', 'bold')
    mx_r, my_r, _, _ = place_mol(F, 'fam_radical', ex_x + 44, by_card2 + 72, 0.70)
    halo(F, 'fam_radical', 2, mx_r, my_r, 0.70, PURPLE, 7.5, 0.32)
    F.text(ex_x + 44, by_card2 + 124, 'Radical (75% blind)', 8.4, INK, 'middle')

    mx_p, my_p, _, _ = place_mol(F, 'fam_pd', ex_x + 126, by_card2 + 72, 0.72)
    halo(F, 'fam_pd', 1, mx_p, my_p, 0.72, PURPLE, 8.0, 0.28)
    F.text(ex_x + 126, by_card2 + 124, 'Pd(II) (60% blind)', 8.4, INK, 'middle')

    return F.save(OUT / 'figS3_falsified_and_tracking.svg')


def build_figS4():
    """Fig. S4: Heavy-atom bond-change (Δbonds) distribution and concerted pericyclic reaction audit
    with ChemDraw structures and electron-pushing arrows (§S8, Table S13).
    """
    F = Fig(1000, 380)

    # =========================================================================
    # a | Heavy-atom bond-change (Δbond) distribution across 1,483 reference steps
    # =========================================================================
    F.letter(0, 18, 'a')
    F.text(24, 18, 'Heavy-atom bond-change (Δbond) distribution across 1,483 reference steps', 12, INK, 'start', 'bold')

    X0, X1, Y0, Y1 = 54, 496, 56, 286
    Y = F.yaxis(X0, X1, Y0, Y1, 0, 42, (0, 10, 20, 30, 40), 'share of reference steps (%)', lx=30)
    dist = [
        ('Δbond = 0', '457 (±H+/SET)', 30.8, NAVY),
        ('Δbond = 1', '526 (1-bond)', 35.5, TEAL),
        ('Δbond = 2', '384 (2-bond/peri.)', 25.9, BLUE),
        ('Δbond = 3', '88 (flagged)', 5.9, AMBER),
        ('Δbond ≥ 4', '28 (flagged)', 1.9, RED),
    ]
    bw = 48
    for gi, (glab, gsub, pct, col) in enumerate(dist):
        gx = X0 + 14 + gi * 84
        F.bar(gx, Y1, pct, None, None, Y, bw, col, ci=False)
        F.text(gx + bw / 2, Y(pct) - 6, f'{pct:.1f}%', 10, col, 'middle', 'bold')
        F.text(gx + bw / 2, Y1 + 16, glab, 9.8, INK, 'middle', 'bold')
        F.text(gx + bw / 2, Y1 + 28, gsub, 8.8, MUTED, 'middle')

    # Threshold line between Δbonds=2 and Δbonds=3
    tx = X0 + 14 + 2 * 84 + bw + 18
    F.line(tx, Y0 - 6, tx, Y1, RED, 1.5, '4 3')
    F.text(tx - 8, Y0 + 6, '92.2% ≤ 2 (unflagged)', 9.5, TEAL, 'end', 'bold')
    F.text(tx + 8, Y0 + 6, '7.8% ≥ 3 (116 steps)', 9.5, RED, 'start', 'bold')

    # Inset breakdown of the 116 flagged reference steps
    ix, iy = tx + 8, Y0 + 22
    F.rect(ix, iy, 172, 92, '#FFF8F6', RED, 0.9, 5)
    F.text(ix + 8, iy + 15, '116 steps with Δbond ≥ 3:', 9.2, RED, 'start', 'bold')
    f116 = [
        ('TM/main-group coord.', '52.6% (61)'),
        ('3-Membered onium', '23.3% (27)'),
        ('Multi-bond frag.', '21.6% (25)'),
        ('Higher cycloadd.', '2.6% (3)'),
    ]
    for k, (kl, kv) in enumerate(f116):
        F.text(ix + 8, iy + 32 + k * 15, f'• {kl}: {kv}', 8.8, INK)

    # =========================================================================
    # b | Concerted pericyclic reaction audit with ChemDraw structures & curly arrows
    # =========================================================================
    BX = 528
    F.letter(BX, 18, 'b')
    F.text(BX + 24, 18, 'Concerted pericyclic audit: Δbond = 2 ≤ 2 (0.0% false flags)', 12, INK, 'start', 'bold')

    # Card 1: Diels-Alder [4+2] cycloaddition (n = 18)
    c1y = 36
    F.rect(BX + 12, c1y, 446, 152, '#F8FAFC', TEAL, 1.1, 6)
    F.text(BX + 24, c1y + 18, '1. Concerted [4+2] Diels–Alder cycloaddition (n = 18 reference steps)', 10.5, TEAL, 'start', 'bold')

    cy_da = c1y + 84
    dx, dy, dw, dh = place_mol(F, 'da_diene', BX + 74, cy_da, 1.0)
    ex, ey, ew, eh = place_mol(F, 'da_ene', BX + 136, cy_da + 4, 1.0)

    # 3 concerted curly arrows for Diels-Alder
    d_c0 = atom_xy(META, 'da_diene', 0, dx, dy)
    d_c3 = atom_xy(META, 'da_diene', 3, dx, dy)
    d_b01 = bond_mid(META, 'da_diene', 0, 1, dx, dy)
    d_b12 = bond_mid(META, 'da_diene', 1, 2, dx, dy)
    d_b23 = bond_mid(META, 'da_diene', 2, 3, dx, dy)
    e_c0 = atom_xy(META, 'da_ene', 0, ex, ey)
    e_c1 = atom_xy(META, 'da_ene', 1, ex, ey)
    e_b01 = bond_mid(META, 'da_ene', 0, 1, ex, ey)

    F.curly((d_b01[0] + 4, d_b01[1]), (e_c0[0] - 3, e_c0[1] + 2), bend=0.35, trim=(1, 2))
    F.curly((e_b01[0] - 3, e_b01[1] + 2), (d_c3[0] + 3, d_c3[1] - 2), bend=0.35, trim=(1, 2))
    F.curly((d_b23[0] - 2, d_b23[1] - 2), (d_b12[0] - 4, d_b12[1]), bend=0.55, trim=(1, 2))

    # Reaction arrow -> Adduct
    F.line(BX + 168, cy_da, BX + 216, cy_da, TEAL, 1.4, marker='aTeal')
    F.text(BX + 192, cy_da - 8, '[4+2]', 9.5, TEAL, 'middle', 'bold')

    # Highlight the 2 newly formed sigma bonds on adduct (atoms 2-3 and 6-7, leaving retained C4=C5 double bond untouched)
    ax, ay, aw, ah = place_mol(F, 'da_adduct', BX + 258, cy_da - 4, 0.95,
                               pre_bonds=[(2, 3, TEAL), (6, 7, TEAL)])

    # Right badges for Diels-Alder
    F.rect(BX + 306, c1y + 38, 140, 42, '#fff', TEAL, 1.0, 5)
    F.text(BX + 376, c1y + 55, '+2 σ formed, 0 broken', 9.5, INK, 'middle', 'bold')
    F.text(BX + 376, c1y + 71, 'Δbond = 2 (unflagged)', 9.8, TEAL, 'middle', 'bold')

    F.rect(BX + 306, c1y + 90, 140, 44, TEAL_L, TEAL, 1.0, 5)
    F.check(BX + 322, c1y + 106, TEAL, 4.5, 1.8)
    F.text(BX + 332, c1y + 110, '0 / 18 false flags', 9.5, TEAL, 'start', 'bold')
    F.text(BX + 332, c1y + 125, '18 / 18 concerted kept', 9.2, INK, 'start')

    # Card 2: [3,3]-Sigmatropic Claisen / Cope rearrangement (n = 6)
    c2y = 204
    F.rect(BX + 12, c2y, 446, 152, '#F8FAFC', BLUE_D, 1.1, 6)
    F.text(BX + 24, c2y + 18, '2. Concerted [3,3]-sigmatropic Claisen / Cope shift (n = 6 reference steps)', 10.5, BLUE_D, 'start', 'bold')

    cy_cl = c2y + 86
    sx, sy, sw, sh = place_mol(F, 'claisen_sub', BX + 98, cy_cl, 1.0)
    # 3 concerted curly arrows around the 6-membered Claisen transition state
    c_b01 = bond_mid(META, 'claisen_sub', 0, 1, sx, sy)
    c_b12 = bond_mid(META, 'claisen_sub', 1, 2, sx, sy)
    c_b23 = bond_mid(META, 'claisen_sub', 2, 3, sx, sy)
    c_b34 = bond_mid(META, 'claisen_sub', 3, 4, sx, sy)
    c_b45 = bond_mid(META, 'claisen_sub', 4, 5, sx, sy)
    c_a0 = atom_xy(META, 'claisen_sub', 0, sx, sy)
    c_a5 = atom_xy(META, 'claisen_sub', 5, sx, sy)

    # Dashed nascent C0-C5 bond
    F.line(c_a0[0], c_a0[1], c_a5[0], c_a5[1], TEAL, 1.1, '2.5 2')
    F.curly(c_b45, (c_a0[0] + 2, c_a0[1] - 1), bend=0.45, trim=(1, 2))
    F.curly(c_b01, c_b12, bend=0.55, trim=(1, 2))
    F.curly(c_b23, c_b34, bend=0.55, trim=(1, 2))

    # Reaction arrow -> Rearranged γ,δ-unsaturated aldehyde
    F.line(BX + 150, cy_cl, BX + 202, cy_cl, BLUE_D, 1.4, marker='aInk')
    F.text(BX + 176, cy_cl - 8, '[3,3]', 9.5, BLUE_D, 'middle', 'bold')

    # Highlight the newly formed C-C sigma bond (atoms 2-3)
    px, py, pw, ph = place_mol(F, 'claisen_prod', BX + 248, cy_cl, 1.0,
                               pre_bonds=[(2, 3, TEAL)])

    # Right badges for Claisen/Cope
    F.rect(BX + 306, c2y + 38, 140, 42, '#fff', BLUE_D, 1.0, 5)
    F.text(BX + 376, c2y + 55, '+1 σ formed, 1 broken', 9.5, INK, 'middle', 'bold')
    F.text(BX + 376, c2y + 71, 'Δbond = 2 (unflagged)', 9.8, BLUE_D, 'middle', 'bold')

    F.rect(BX + 306, c2y + 90, 140, 44, BLUE_L, BLUE_D, 1.0, 5)
    F.check(BX + 322, c2y + 106, BLUE_D, 4.5, 1.8)
    F.text(BX + 332, c2y + 110, '0 / 6 false flags', 9.5, BLUE_D, 'start', 'bold')
    F.text(BX + 332, c2y + 125, '6 / 6 concerted kept', 9.2, INK, 'start')

    return F.save(OUT / 'figS4_pericyclic_and_ablation.svg')


def build_figS5():
    """Fig. S5: 2026 Post-cutoff literature extraction & audit, controlled HRMS blind-spot rescue,
    and pathway-length restoration vs F1 gain (§S3 & §S9, Tables S2 & S14).
    """
    F = Fig(1000, 430)

    # =========================================================================
    # a | 2026 Literature multimodal extraction pipeline (left) & clue/gold audits (right)
    # =========================================================================
    F.letter(0, 18, 'a')
    F.text(24, 18, '2026 literature curation pipeline (N = 436 papers) and extraction quality audits', 12, INK, 'start', 'bold')

    steps_pipe = [
        ('1. 2026 OA PDFs', 'N = 436 articles', 'Jan–Nov 2026', '#F8FAFC', NAVY),
        ('2. Multimodal extract', 'Schemes + clues', 'Conclusions stripped', BLUE_L, BLUE_D),
        ('3. RDKit verification', 'Valence & element', 'MCS skeleton check', TEAL_L, TEAL),
        ('4. Curated corpus', '285 post-cutoff (1,571)', '436 total (2,315)', PURPLE_L, PURPLE),
    ]
    for i, (tit, l1, l2, bg, fg) in enumerate(steps_pipe):
        px = 18 + i * 134
        F.rect(px, 34, 122, 64, bg, fg, 1.1, 6)
        F.text(px + 61, 52, tit, 9.8, fg, 'middle', 'bold')
        F.text(px + 61, 68, l1, 9.2, INK, 'middle', 'bold')
        F.text(px + 61, 82, l2, 8.8, MUTED, 'middle')
        if i < 3:
            F.line(px + 124, 66, px + 132, 66, MUTED, 1.2, marker='aMuted')

    # Right of Panel a: Two horizontal stacked audit bars (4,388 clue lines & 50-paper manual audit)
    ax0, aw0 = 568, 414
    # Bar 1: 4,388 clue statements
    F.text(ax0, 44, 'Clue modality audit (4,388 statements across N = 436 papers):', 9.5, INK, 'start', 'bold')
    w_wet = aw0 * 0.914
    w_dft = aw0 * 0.069
    w_oth = aw0 * 0.017
    F.rect(ax0, 50, w_wet - 1, 16, BLUE_D, 'none', 0, 3)
    F.rect(ax0 + w_wet, 50, w_dft - 1, 16, AMBER, 'none', 0, 0)
    F.rect(ax0 + w_wet + w_dft, 50, w_oth, 16, RED, 'none', 0, 2)
    F.text(ax0 + 8, 62, '91.4% wet-lab experimental (4,009)', 9.2, '#fff', 'start', 'bold')
    F.text(ax0 + aw0, 44, '6.9% DFT · 0.6% concl.', 9.0, MUTED, 'end')

    # Bar 2: 50-paper manual audit (269 intermediates, 488 clues)
    F.text(ax0, 82, '50-Paper PDF audit (269 intermediates · 98.0% conclusion-free clues):', 9.5, INK, 'start', 'bold')
    w_ex = aw0 * 0.743
    w_va = aw0 * 0.212
    w_er = aw0 * 0.045
    F.rect(ax0, 88, w_ex - 1, 16, TEAL, 'none', 0, 3)
    F.rect(ax0 + w_ex, 88, w_va - 1, 16, '#26A69A', 'none', 0, 0)
    F.rect(ax0 + w_ex + w_va, 88, w_er, 16, RED, 'none', 0, 2)
    F.text(ax0 + 8, 100, '74.3% exact + 21.2% ±H+/ligand = 95.5% valid', 9.2, '#fff', 'start', 'bold')
    F.text(ax0 + aw0, 82, '4.5% error', 9.0, RED, 'end', 'bold')

    # =========================================================================
    # b | Wet-lab-only & held-out robustness across the 2026 literature corpus
    # =========================================================================
    TOPB = 140
    F.line(14, TOPB - 18, 986, TOPB - 18, GRID, 0.8)
    F.letter(0, TOPB, 'b')
    F.text(24, TOPB, 'Wet-lab-only and held-out robustness across 2026 literature', 12, INK, 'start', 'bold')

    X0, X1, Y0, Y1 = 52, 540, TOPB + 42, TOPB + 236
    Y = F.yaxis(X0, X1, Y0, Y1, 0, 75, (0, 25, 50, 75), 'intermediate recall (%)', lx=28)

    sets_b = [
        ('All test\n(n = 271)', 46.7, 56.7, '+10.0 pp'),
        ('DFT removed\n(n = 271)', 45.0, 55.7, '+10.7 pp'),
        ('Zero-DFT papers\n(n = 105)', 43.0, 54.4, '+11.4 pp'),
        ('Post-cutoff, no DFT\n(n = 167)', 47.7, 59.9, '+12.2 pp'),
    ]
    bw = 34
    for gi, (glab, v_dc, v_gd, dlab) in enumerate(sets_b):
        gx = X0 + 26 + gi * 118
        F.bar(gx, Y1, v_dc, None, None, Y, bw, '#E6B8A8', PURPLE, 1.1, ci=False)
        F.text(gx + bw / 2, Y(v_dc) - 5, f'{v_dc:.1f}', 9.0, INK, 'middle', 'bold')
        bx2 = gx + bw + 8
        F.bar(bx2, Y1, v_gd, None, None, Y, bw, PURPLE, 'none', 0, ci=False)
        F.text(bx2 + bw / 2, Y(v_gd) - 5, f'{v_gd:.1f}', 9.0, PURPLE, 'middle', 'bold')
        # Gain badge above pair
        y_top = Y(v_gd) - 21
        F.text(gx + bw + 4, y_top, dlab, 8.8, PURPLE, 'middle', 'bold')
        l1, l2 = glab.split('\n')
        F.text(gx + bw + 4, Y1 + 15, l1, 8.8, INK, 'middle', 'bold')
        F.text(gx + bw + 4, Y1 + 27, l2, 8.4, MUTED, 'middle')

    # Compact top legend for Panel b
    F.rect(X0, TOPB + 14, 10, 9, '#E6B8A8', PURPLE, 1.0, 2)
    F.text(X0 + 14, TOPB + 22, 'Direct + Clues', 9.2, INK, 'start')
    F.rect(X0 + 112, TOPB + 14, 10, 9, PURPLE, 'none', 0, 2)
    F.text(X0 + 126, TOPB + 22, 'MechSight-Guided', 9.2, PURPLE, 'start', 'bold')

    # =========================================================================
    # c | Pathway-length restoration & F1 score on 2026 Test Set (n = 271)
    # =========================================================================
    BX = 572
    F.letter(BX, TOPB, 'c')
    F.text(BX + 24, TOPB, 'Pathway-length restoration & F1 gain (n = 271 papers)', 12, INK, 'start', 'bold')

    # Left subchart: Mean intermediates per pathway vs Reference 5.15
    X0c1, X1c1, Y0c1, Y1c1 = BX + 40, BX + 224, TOPB + 42, TOPB + 236
    Yc1 = F.yaxis(X0c1, X1c1, Y0c1, Y1c1, 0, 6.5, (0, 2, 4, 6), 'mean intermediates / paper', lx=26)
    # Reference horizontal line at 5.15
    y_ref = Yc1(5.15)
    F.line(X0c1, y_ref, X1c1, y_ref, NAVY, 1.3, '4 3')
    F.text(X1c1 - 32, y_ref - 6, 'Author ref. = 5.15', 9.0, NAVY, 'end', 'bold')

    len_arms = [
        ('Pool vote', 2.45, '#FAEAE4', PURPLE),
        ('Direct', 3.82, GRAY, None),
        ('D+Clues', 4.06, '#E6B8A8', PURPLE),
        ('Auto', 4.43, TEAL, None),
        ('Guided', 5.34, PURPLE, None),
    ]
    for i, (lab, v, col, edge) in enumerate(len_arms):
        bx = X0c1 + 8 + i * 35
        F.bar(bx, Y1c1, v, None, None, Yc1, 24, col, edge or 'none', 1.1 if edge else 0, ci=False)
        F.text(bx + 12, Yc1(v) - 5 if v < 4.8 else Yc1(v) + 15, f'{v:.2f}', 8.8,
               '#fff' if lab == 'Guided' else INK, 'middle', 'bold')
        F.text(bx + 12, Y1c1 + 15, lab, 8.8, PURPLE if lab == 'Guided' else INK, 'middle', 'bold' if lab == 'Guided' else 'normal')

    # Right subchart: F1 score (%) on 2026 Confirmation (n = 271) and Wet-lab-only nodft (n = 271)
    X0c2, X1c2, Y0c2, Y1c2 = BX + 266, 988, TOPB + 42, TOPB + 236
    Yc2 = F.yaxis(X0c2, X1c2, Y0c2, Y1c2, 0, 55, (0, 15, 30, 45), 'intermediate F1 score (%)', lx=26)
    f1_arms = [
        ('Direct', 37.3, GRAY, None),
        ('Auto', 38.7, TEAL, None),
        ('D+Clues', 40.8, '#E6B8A8', PURPLE),
        ('Guided', 44.6, PURPLE, None),
    ]
    xs_f1 = []
    for i, (lab, v, col, edge) in enumerate(f1_arms):
        bx = X0c2 + 12 + i * 36
        xs_f1.append(bx + 13)
        F.bar(bx, Y1c2, v, None, None, Yc2, 26, col, edge or 'none', 1.1 if edge else 0, ci=False)
        F.text(bx + 13, Y1c2 - 6, f'{v:.1f}', 9.0, '#fff' if lab in ('Direct', 'Auto', 'Guided') else INK, 'middle', 'bold')
        F.text(bx + 13, Y1c2 + 15, lab, 9.0, PURPLE if lab == 'Guided' else INK, 'middle', 'bold' if lab == 'Guided' else 'normal')

    top_f1 = Yc2(48.2) - 6
    F.path(f'M{xs_f1[2]},{top_f1 + 5} L{xs_f1[2]},{top_f1} L{xs_f1[3]},{top_f1} L{xs_f1[3]},{top_f1 + 5}', PURPLE, 1.2)
    F.text((xs_f1[2] + xs_f1[3]) / 2, top_f1 - 4, '+3.9 pp F1', 9.5, PURPLE, 'middle', 'bold')
    F.text((X0c1 + X1c2) / 2, Y1c1 + 32, 'Restores reference pathway length without step-padding (wet-lab-only F1: +4.0 pp)', 9.2, MUTED, 'middle', 'bold')

    return F.save(OUT / 'figS5_blindspot_and_lit_pipeline.svg')


if __name__ == '__main__':
    for fn in (build_figS1, build_figS2, build_figS3, build_figS4, build_figS5):
        p = fn()
        print('generated', p)
