"""Shared drawing kit for the Angew figures: one palette, one type scale, SVG primitives.

Units: viewBox 1000 wide = 175 mm  ->  1 unit = 0.496 pt.  Text 11-13 units = 5.5-6.5 pt, panel letters 21 units.
Colour semantics (identical in every figure):
  GRAY   Direct             CTRL1-3  think-more controls (CoT, self-refine, self-consistency)
  TEAL   MechSight-Auto     PURPLE   MechSight-Guided (experimental evidence)
  AMBER  hidden intermediate / +hidden GT        BLUE  +visible GT        RED  gap / error
  PLUM   specialist / template models           NAVY  general LLM runs (Fig. 1)
"""
import json
import math
import re
from pathlib import Path

INK, MUTED, FAINT, GRID = '#182026', '#545E6B', '#8E99A6', '#E4E7EC'
# palette v3 (2026-11-07): Nature/Science Dual-Star (Royal Cobalt + Terracotta Coral + Imperial Ochre + Warm Slate)
GRAY = '#6E7785'
CTRL1, CTRL2, CTRL3 = '#D3DBE4', '#B5C2D1', '#94A6BA'
TEAL, TEAL_L = '#0F5C8C', '#E8F2F8'
PURPLE, PURPLE_L = '#C84B31', '#FCEEE9'
AMBER, AMBER_L = '#D9822B', '#FEF5E7'
BLUE, BLUE_L, BLUE_D = '#5A8EC8', '#EDF4FB', '#346399'
RED = '#B8293D'
PLUM, PLUM_L = '#8F4061', '#C88EA4'
NAVY, SLATE = '#1F3A56', '#5F7894'
EPUSH = '#B8293D'          # electron-pushing arrows
FONT = "Arial, Helvetica, sans-serif"


class Fig:
    def __init__(self, W=1000, H=640):
        self.W, self.H, self.out = W, H, []
        self.add(f'<defs>{self._mk("aInk", INK, 8)}{self._mk("aTeal", TEAL, 9)}{self._mk("aGray", GRAY, 8)}'
                 f'{self._mk("aMuted", MUTED, 6)}{self._mk("aPurple", PURPLE, 9)}{self._mk("aRed", RED, 9)}{self._mk("aAmber", AMBER, 8)}'
                 f'<marker id="aE" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0.6 L10,5 L0,9.4 L2.4,5 Z" fill="{EPUSH}"/></marker>'
                 f'<marker id="aEh" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0.6 L10,5 L2.4,5 Z" fill="{EPUSH}"/></marker>'
                 f'<filter id="glow" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="5"/></filter>'
                 f'<linearGradient id="gainFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{TEAL}" stop-opacity="0.22"/>'
                 f'<stop offset="1" stop-color="{TEAL}" stop-opacity="0.08"/></linearGradient></defs>')

    @staticmethod
    def _mk(id_, color, size):
        return (f'<marker id="{id_}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="{size}" markerHeight="{size}" '
                f'orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M0,0.8 L10,5 L0,9.2 L2.6,5 Z" fill="{color}"/></marker>')

    def add(self, s):
        self.out.append(s)

    # ---- primitives
    def text(self, x, y, s, size=12, fill=INK, anchor='start', weight='normal', style='normal', rot=None):
        s_esc = re.sub(r'&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)', '&amp;', str(s))
        tr = f' transform="rotate({rot} {x:.1f} {y:.1f})"' if rot is not None else ''
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
                 f'font-weight="{weight}" font-style="{style}"{tr}>{s_esc}</text>')

    def line(self, x1, y1, x2, y2, stroke=INK, w=1, dash=None, cap='round', marker=None, op=1):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        m = f' marker-end="url(#{marker})"' if marker else ''
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}" '
                 f'stroke-linecap="{cap}" opacity="{op}"{d}{m}/>')

    def path(self, d, stroke=INK, w=1, fill='none', dash=None, marker=None, op=1):
        ds = f' stroke-dasharray="{dash}"' if dash else ''
        m = f' marker-end="url(#{marker})"' if marker else ''
        self.add(f'<path d="{d}" stroke="{stroke}" stroke-width="{w}" fill="{fill}" stroke-linecap="round" '
                 f'stroke-linejoin="round" opacity="{op}"{ds}{m}/>')

    def circle(self, x, y, r, fill='none', stroke='none', w=1, op=1):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" opacity="{op}"/>')

    def rect(self, x, y, w, h, fill='none', stroke='none', sw=1, rx=0, dash=None, op=1):
        ds = f' stroke-dasharray="{dash}"' if dash else ''
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}" opacity="{op}"{ds}/>')

    def glow(self, x, y, w, h, color=AMBER, rx=9, op=0.28):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{color}" opacity="{op}" filter="url(#glow)"/>')

    def diamond(self, x, y, r, fill, stroke, w=1.2):
        self.add(f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" transform="rotate(45 {x:.1f} {y:.1f})" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{w}"/>')

    def tri(self, x, y, r, fill, stroke, w=1.2):
        self.add(f'<path d="M{x:.1f},{y - r:.1f} L{x + r * 0.95:.1f},{y + r * 0.7:.1f} L{x - r * 0.95:.1f},{y + r * 0.7:.1f} Z" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{w}" stroke-linejoin="round"/>')

    def star(self, x, y, r=5.5, fill=NAVY):
        pts = []
        for i in range(10):
            rr = r if i % 2 == 0 else r * 0.45
            a = math.pi / 2 + i * math.pi / 5
            pts.append(f'{x + rr * math.cos(a):.1f},{y - rr * math.sin(a):.1f}')
        self.add(f'<polygon points="{" ".join(pts)}" fill="{fill}"/>')

    def check(self, x, y, color=TEAL, sz=5, w=1.8):
        self.path(f'M{x - sz:.1f},{y:.1f} L{x - sz * 0.25:.1f},{y + sz * 0.7:.1f} L{x + sz:.1f},{y - sz * 0.8:.1f}', color, w)

    def cross(self, x, y, color=RED, sz=4.5, w=1.7):
        self.line(x - sz, y - sz, x + sz, y + sz, color, w); self.line(x - sz, y + sz, x + sz, y - sz, color, w)

    def badge(self, x, y, n, color, r=8.5):
        self.circle(x, y, r, fill=color)
        self.text(x, y + 4.2, str(n), 11.5, '#fff', 'middle', 'bold')

    def letter(self, x, y, s):
        self.text(x, y, s, 16.5, INK, 'start', 'bold')

    def pips(self, x, y, k):
        if k == 0:
            self.rect(x - 5, y, 10, 10, '#fff', FAINT, 1, 2, '2 1.5'); return
        for i in range(k):
            self.rect(x - k * 6 + i * 12 + 1, y, 10, 10, AMBER_L, AMBER, 1, 2)

    def bar(self, x, y0, v, lo, hi, Y, w, fill, stroke='none', sw=0, ci=True, rx=2.5):
        self.rect(x, Y(v), w, Y(0) - Y(v), fill, stroke, sw, rx)
        if ci and lo is not None:
            self.line(x + w / 2, Y(hi), x + w / 2, Y(lo), INK, 0.8, cap='butt', op=0.45)
            self.line(x + w / 2 - 3.5, Y(hi), x + w / 2 + 3.5, Y(hi), INK, 0.8, op=0.45)

    def yaxis(self, x0, x1, y0, y1, vmin, vmax, ticks, label=None, lx=34, fmt='{:.0f}'):
        for t in ticks:
            yy = y1 - (t - vmin) / (vmax - vmin) * (y1 - y0)
            self.line(x0, yy, x1, yy, GRID, 0.8, cap='butt')
            self.text(x0 - 6, yy + 4, fmt.format(t), 11, MUTED, 'end')
        if label:
            self.add(f'<text transform="translate({x0 - lx},{(y0 + y1) / 2}) rotate(-90)" font-size="11.5" fill="{MUTED}" '
                     f'text-anchor="middle">{label}</text>')
        self.line(x0, y1, x1, y1, INK, 0.9, cap='butt')
        return lambda v: y1 - (v - vmin) / (vmax - vmin) * (y1 - y0)

    def mol(self, mols_dir, name, x, y, meta=None, s=1.0):
        meta = meta or json.loads((Path(mols_dir) / 'meta.json').read_text())
        m = meta[name]
        body = (Path(mols_dir) / f'{name}.svg').read_text()
        body = re.sub(r"width='[\d.]+px' height='[\d.]+px'", f"x='{x:.1f}' y='{y:.1f}' width='{m['w'] * s:.1f}' height='{m['h'] * s:.1f}'", body, count=1)
        self.add(body)
        return m['w'] * s, m['h'] * s

    def curly(self, p0, p1, bend=0.45, half=False, col=None, w=0.85, trim=(2.5, 3.5)):
        """Electron-pushing arrow p0 -> p1 (quadratic curve). bend > 0 bulges to the left of the travel direction.
        half=True draws a fish-hook (one electron)."""
        (x0, y0), (x1, y1) = p0, p1
        L = math.hypot(x1 - x0, y1 - y0) or 1.0
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        x0, y0 = x0 + ux * trim[0], y0 + uy * trim[0]
        x1, y1 = x1 - ux * trim[1], y1 - uy * trim[1]
        cx, cy = (x0 + x1) / 2 + uy * bend * L, (y0 + y1) / 2 - ux * bend * L
        col = col or EPUSH
        mk = 'aEh' if half else 'aE'
        self.add(f'<path d="M{x0:.1f},{y0:.1f} Q{cx:.1f},{cy:.1f} {x1:.1f},{y1:.1f}" fill="none" stroke="{col}" '
                 f'stroke-width="{w}" stroke-linecap="round" marker-end="url(#{mk})"/>')

    def pill(self, x, y, s, w, color=NAVY, fill='#fff', size=10):
        self.rect(x - w / 2, y - 10, w, 16, fill, color, 0.8, 8)
        self.text(x, y + 2.5, s, size, color, 'middle')

    def save(self, path, width_mm=175):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width_mm}mm" '
               f'height="{width_mm * self.H / self.W:.2f}mm" viewBox="0 0 {self.W} {self.H}" font-family="{FONT}">'
               f'<rect width="{self.W}" height="{self.H}" fill="#fff"/>' + '\n'.join(self.out) + '</svg>')
        Path(path).write_text(svg)
        return len(svg)


def mol_meta(mols_dir):
    return json.loads((Path(mols_dir) / 'meta.json').read_text())


def atom_xy(meta, name, idx, x, y, s=1.0):
    """Figure coordinates of atom `idx` of structure `name` placed at (x, y) with scale s."""
    ax, ay = meta[name]['atoms'][str(idx)]
    return x + ax * s, y + ay * s


def bond_mid(meta, name, i, j, x, y, s=1.0):
    """Midpoint of bond i-j of a placed structure."""
    (ax, ay), (bx, by) = atom_xy(meta, name, i, x, y, s), atom_xy(meta, name, j, x, y, s)
    return (ax + bx) / 2, (ay + by) / 2
