"""Export a mechanism as one publication figure: panels of structures, arrows between them.

The screen viewer shows one step at a time; a paper wants the whole path at once.
Both come from the same renderer, so what a reviewer sees is exactly what the
viewer showed — this module only lays the frames out, crops them, and swaps the
page's CSS variables for print colours.

    python -m develop.viz.export_figure --case 1 --path pred --out imine.svg
    python -m develop.viz.export_figure --case 1 --path gt --mono --no-labels --columns 3
    python develop/viz/export_figure.py --payload develop/inspector/payload.js --list

Every panel of a path is cropped identically and drawn at one scale, so a
structure sits in the same place in every panel and the eye only has to follow
the arrows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

PRINT = {
    "ink": "#111111",
    "form": "#0F7A5F",
    "break": "#B3372B",
    "shift": "#2B3F9E",
    "accent_soft": "#EDF0FA",
    "break_soft": "#F7E2DE",
    "label": "#7A7A7A",
    "paper": "#FFFFFF",
}

MONO = dict(PRINT, form="#111111", **{"break": "#111111"}, shift="#111111",
            accent_soft="#F2F2F2", break_soft="#E8E8E8", label="#666666")

PRINT_TOKENS = (
    ("#FF00FF", "__ACCENT_SOFT__"),
    ("#ff00ff", "__ACCENT_SOFT__"),
    ("#FF0000", "__BREAK_SOFT__"),
    ("#ff0000", "__BREAK_SOFT__"),
    ("#000000", "__INK__"),
)


def _style(palette: Dict[str, str], caption_size: float) -> str:
    return f"""
    .mol-layer path {{ stroke-linecap: round; }}
    .ghost {{ stroke: {palette['form']}; stroke-width: 1.6; stroke-dasharray: 2.5 4.5; fill: none; }}
    .lp {{ fill: {palette['ink']}; }}
    .arrow {{ fill: none; stroke-width: 2.0; stroke-linecap: round; }}
    .arrow.form {{ stroke: {palette['form']}; }}
    .arrow.break {{ stroke: {palette['break']}; }}
    .arrow.shift {{ stroke: {palette['shift']}; }}
    .head.form {{ fill: {palette['form']}; }}
    .head.break {{ fill: {palette['break']}; }}
    .head.shift {{ fill: {palette['shift']}; }}
    .charge {{ font: 600 13px Helvetica, Arial, sans-serif; fill: {palette['ink']};
               text-anchor: middle; dominant-baseline: central; }}
    .pgl {{ font: 400 9.5px Helvetica, Arial, sans-serif; font-variant-numeric: tabular-nums; fill: {palette['label']};
            text-anchor: middle; dominant-baseline: central; }}
    .caption {{ font: 400 {caption_size}px Helvetica, Arial, sans-serif; fill: {palette['ink']};
                text-anchor: middle; }}
    .step-arrow {{ stroke: {palette['ink']}; stroke-width: 1.3; fill: none; }}
    .step-head {{ fill: {palette['ink']}; }}
    """


def _paint(svg: str, palette: Dict[str, str]) -> str:
    return (
        svg.replace("__INK__", palette["ink"])
        .replace("__ACCENT_SOFT__", palette["accent_soft"])
        .replace("__BREAK_SOFT__", palette["break_soft"])
    )


def crop_box(coords, world, pad: float) -> Tuple[float, float, float, float]:
    """One crop for every panel, so a structure sits in the same place throughout.

    A printed panel shows the whole molecule, not the reaction centre — the
    reader needs to see what did *not* change as much as what did.
    """
    gx, gy, gs = world
    xs = [gx + c[0] * gs for c in coords.values()]
    ys = [gy + c[1] * gs for c in coords.values()]
    return (min(xs) - pad, min(ys) - pad, max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad)


def build_figure(
    case: Dict[str, Any],
    path: str,
    *,
    palette: Dict[str, str],
    columns: int,
    show_labels: bool,
    show_ghost: bool,
    highlight: bool,
    panel_width: float,
    captions: bool,
) -> str:
    import render_rdkit as R

    coords = case["coords"]
    canvas = tuple(case["canvas"])
    world = R.global_transform(coords)
    frames = case[path]

    drawn = []
    for i, frame in enumerate(frames):
        nxt = frames[i + 1]["state"] if i + 1 < len(frames) else None
        drawn.append(
            R.render_frame(
                frame,
                coords,
                canvas,
                world,
                next_state=nxt,
                show_labels=show_labels,
                show_ghost=show_ghost,
                highlight=highlight,
                tokens=PRINT_TOKENS,
            )
        )
    cx, cy, cw, ch = crop_box(coords, world, pad=30.0)
    scale = panel_width / cw
    panel_h = ch * scale
    gap = 58.0                                   # room for the reaction arrow
    caption_h = 26.0 if captions else 8.0

    columns = max(1, min(columns, len(drawn)))
    rows = (len(drawn) + columns - 1) // columns
    width = columns * panel_width + (columns - 1) * gap + 24
    height = rows * (panel_h + caption_h) + (rows - 1) * 26 + 24

    parts: List[str] = []
    for i, out in enumerate(drawn):
        col, row = i % columns, i // columns
        x = 12 + col * (panel_width + gap)
        y = 12 + row * (panel_h + caption_h + 26)
        parts.append(
            f'<svg x="{x:.1f}" y="{y:.1f}" width="{panel_width:.1f}" height="{panel_h:.1f}" '
            f'viewBox="{cx:.1f} {cy:.1f} {cw:.1f} {ch:.1f}" overflow="hidden">{out["svg"]}</svg>'
        )
        if captions:
            last = i == len(drawn) - 1
            text = "product" if last else f"step {i + 1}"
            parts.append(
                f'<text class="caption" x="{x + panel_width / 2:.1f}" '
                f'y="{y + panel_h + 16:.1f}">{text}</text>'
            )
        if i < len(drawn) - 1 and col < columns - 1:
            ax = x + panel_width + 10
            ay = y + panel_h / 2
            parts.append(
                f'<path class="step-arrow" d="M {ax:.1f} {ay:.1f} L {ax + gap - 28:.1f} {ay:.1f}"/>'
                f'<polygon class="step-head" points="{ax + gap - 20:.1f},{ay:.1f} '
                f'{ax + gap - 29:.1f},{ay - 3.6:.1f} {ax + gap - 29:.1f},{ay + 3.6:.1f}"/>'
            )

    body = "".join(parts)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}">'
        f'<rect width="100%" height="100%" fill="{palette["paper"]}"/>'
        f"<style>{_style(palette, 11.0)}</style>{body}</svg>"
    )
    return _paint(svg, palette)


def load_cases(path: Path) -> List[Dict[str, Any]]:
    """Cases from either payload: the prototype's payload.json or the inspector's payload.js.

    Both carry the same per-case fields this module reads (coords, canvas, and
    gt/pred frames with state and transfers); the inspector wraps its JSON in a
    ``window.INSPECTOR_DATA = ...;`` assignment and names the list ``testcases``.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".js":
        text = text[text.index("=") + 1:].strip().rstrip(";")
    data = json.loads(text)
    return data.get("cases") or data.get("testcases") or []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--payload", default=str(HERE / "payload.json"))
    ap.add_argument("--case", default="0", help="index into payload cases, or a reaction id substring")
    ap.add_argument("--path", default="pred", choices=["gt", "pred"])
    ap.add_argument("--out", default=None, help="output .svg (default: <reaction>_<path>.svg here)")
    ap.add_argument("--columns", type=int, default=99, help="panels per row")
    ap.add_argument("--panel-width", type=float, default=300.0)
    ap.add_argument("--mono", action="store_true", help="all-black, for journals that charge for colour")
    ap.add_argument("--no-labels", dest="labels", action="store_false", help="drop the PGL atom labels")
    ap.add_argument("--no-ghost", dest="ghost", action="store_false", help="drop the forming-bond dashes")
    ap.add_argument("--highlight", action="store_true", help="tint the reaction centre (off in print by default)")
    ap.add_argument("--no-captions", dest="captions", action="store_false")
    ap.add_argument("--list", action="store_true", help="print the cases in the payload and exit")
    args = ap.parse_args()

    cases = load_cases(Path(args.payload))
    if args.list:
        for i, c in enumerate(cases):
            print(f"{i:>3}  GT {len(c['gt']) - 1:>2} 步  预测 {len(c['pred']) - 1:>2} 步  {c['reaction_id']}")
        return
    if args.case.isdigit():
        case = cases[int(args.case)]
    else:
        case = next(c for c in cases if args.case in c["reaction_id"])

    figure = build_figure(
        case,
        args.path,
        palette=MONO if args.mono else PRINT,
        columns=args.columns,
        show_labels=args.labels,
        show_ghost=args.ghost,
        highlight=args.highlight,
        panel_width=args.panel_width,
        captions=args.captions,
    )
    name = case["reaction_id"].split("::")[1]
    out = Path(args.out) if args.out else HERE / f"{name}_{args.path}.svg"
    out.write_text(figure, encoding="utf-8")
    print(f"wrote {out}  ({len(case[args.path])} panels)")


if __name__ == "__main__":
    main()
