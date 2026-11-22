"""P0: extract evidence cards (schema.json) from a paper and its SI with an LLM.

Input: papers/<paper_id>/main.pdf and (optionally) papers/<paper_id>/si.pdf.
Pages are converted with pdftotext -layout. Pages whose text mentions a
mechanistic keyword are kept with one neighbour on each side, grouped into
chunks of <= CHUNK_PAGES pages, and sent to the model one chunk per call.
Cards from all chunks are merged: one extra model call per paper groups the
cards that report the same experiment (main text vs SI, neighbouring chunks;
exact type/perturbation/bin grouping if that call fails), and the SI copy is
kept because it carries the numbers. Bins are normalised ('suppressed (<10%)'
-> 'suppressed') and unknown experiment types are mapped back to the schema.

Output: OUT/<model>/<paper_id>.json (merged cards, raw per-chunk cards, page-selection stats);
the raw calls go to OUT/<model>/trace.jsonl.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'agent_v2'))
import common as C
from common import ROOT

OUT = ROOT / 'outputs' / 'evidence_p0_20261101'
PAPERS = HERE / 'papers'
CHUNK_PAGES = 4
KEYWORDS = re.compile(
    r'mechanis|TEMPO|BHT|radical|deuter|\bKIE\b|kH/kD|isotope|18O|13C-label|light on|light off|on/off|quantum yield|'
    r'Stern.?Volmer|quench|cyclic voltam|\bCV\b|EPR|Hammett|reaction order|rate law|initial rate|kinetic|crossover|'
    r'radical clock|cyclopropyl|nonlinear effect|control experiment|intermediate|competition experiment|'
    r'in the absence of|without (light|catalyst|photocatalyst|base)', re.I)

SYSTEM = 'You are an expert physical organic chemist who curates experimental evidence from papers.'
TASK = """Below are pages from a chemistry paper ({file}). Extract every MECHANISTIC experiment reported on these pages as an evidence card: control experiments (a component omitted or replaced), radical trapping, radical clocks, light on/off, quantum yields, kinetic isotope effects, isotope labelling, crossover, Hammett studies, kinetic orders, nonlinear effects, Stern-Volmer quenching, cyclic voltammetry, spectroscopic detection of intermediates, stoichiometric reactions of proposed intermediates, stereochemical probes, competition experiments.

Do NOT extract: substrate-scope entries, optimisation tables (unless an entry is an explicit control: no light, no catalyst, no base, under air ...), characterisation data, or DFT/computational results.

Rules:
- Keep OBSERVATION and INTERPRETATION apart. outcome_raw states only what was measured (numbers with units, species detected, yes/no) with no interpretive words such as "suggests", "indicates", "consistent with", "radical pathway", "rate-determining". The authors' conclusion goes into author_interpretation (null if they give none on these pages).
- perturbation says what changed relative to the standard conditions; observable says what was measured.
- outcome_bin uses these bins (null if none applies):
{bins}
- location.page is the page number shown in the [page N] marker; location.label is the Scheme/Table/Figure/entry label if given.
- quote is a verbatim snippet (<= 30 words) from the pages that supports outcome_raw.
- If the same experiment is reported twice on these pages, give it once.

Answer with a JSON array of cards (possibly empty) inside <cards>...</cards>. Card fields: experiment_type (one of {types}), perturbation, observable, outcome_raw, outcome_bin, location {{file, page, label}}, author_interpretation, quote.

PAGES:
{pages}"""


def pdf_pages(path):
    text = subprocess.run(['pdftotext', '-layout', str(path), '-'], capture_output=True, text=True, check=True).stdout
    return [p for p in text.split('\f')]


def select_pages(pages):
    hits = {i for i, p in enumerate(pages) if KEYWORDS.search(p)}
    keep = sorted({j for i in hits for j in (i - 1, i, i + 1) if 0 <= j < len(pages) and pages[j].strip()})
    return keep, len(hits)


def chunks(keep):
    out, cur = [], []
    for i in keep:
        if cur and (i != cur[-1] + 1 or len(cur) == CHUNK_PAGES):
            out.append(cur)
            cur = []
        cur.append(i)
    return out + ([cur] if cur else [])


SCHEMA = json.loads((HERE / 'schema.json').read_text())
TYPES = SCHEMA['properties']['experiment_type']['enum']

DEDUP = """Below are evidence cards extracted chunk by chunk from the main text and the SI of one chemistry paper. The same experiment is often reported twice (briefly in the main text, in detail in the SI) or in two neighbouring chunks, with different wording.

Group the cards that report the SAME experiment: same perturbation of the same reaction and the same observable. Do not group different conditions, different substrates, different trapping agents, or different observables of one run (e.g. yield suppression vs. adduct detection by TEMPO are two cards).

Answer with a JSON array of groups, each group a list of card ids, inside <groups>...</groups>. Every card id appears in exactly one group; singletons are groups of one.

CARDS:
{cards}"""


def json_block(answer, tag):
    m = re.search(rf'<{tag}>(.*?)</{tag}>', answer, re.S) or re.search(r'```(?:json)?\s*(\[.*?\])\s*```', answer, re.S)
    if not m:
        return None
    body = re.sub(r'^```(?:json)?|```$', '', m.group(1).strip()).strip()
    if not body:
        return []
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def canonical_bin(value):
    """'suppressed (<10%)' -> 'suppressed'; {'radical_trap_adduct': 'adduct_detected'} -> 'adduct_detected'."""
    if isinstance(value, dict):
        value = '; '.join(str(v) for v in value.values())
    elif isinstance(value, list):
        value = '; '.join(str(v) for v in value)
    if value is None or str(value).strip().lower() in {'', 'null', 'none', 'n/a'}:
        return None
    return re.sub(r'\s*\([^)]*\)', '', str(value)).strip()


def parse_cards(answer):
    cards = json_block(answer, 'cards')
    if not isinstance(cards, list):
        return None
    cards = [c for c in cards if isinstance(c, dict) and c.get('experiment_type')]
    for c in cards:  # models sometimes return a field as an object/list; keep it as compact text
        for key in ('perturbation', 'observable', 'outcome_raw', 'author_interpretation', 'quote'):
            if c.get(key) is not None and not isinstance(c[key], str):
                c[key] = json.dumps(c[key], ensure_ascii=False)
        c['outcome_bin'] = canonical_bin(c.get('outcome_bin'))
        loc = c['location'] = c.get('location') if isinstance(c.get('location'), dict) else {}
        page = re.search(r'\d+', str(loc.get('page') or ''))
        loc['page'] = int(page.group()) if page else None
        if c['experiment_type'] not in TYPES:
            c['type_raw'] = c['experiment_type']
            c['experiment_type'] = 'radical_trap' if c['experiment_type'].startswith('radical_trap') else 'other'
    return cards


def norm(text):
    return ' '.join(re.findall(r'[a-z0-9]+', (text or '').lower()))


def exact_groups(cards):
    groups = {}
    for i, c in enumerate(cards):
        groups.setdefault((c['experiment_type'], norm(c.get('perturbation')), norm(c.get('outcome_bin'))), []).append(i)
    return list(groups.values())


def merge(cards, model, paper_id):
    """Group duplicates with one model call (exact-key grouping as fallback); keep the SI copy, else the longest."""
    listing = '\n'.join(f"R{i}: [{c['experiment_type']}] ({(c.get('location') or {}).get('file')} p{(c.get('location') or {}).get('page')}) "
                        f"perturbation: {c.get('perturbation')} | observable: {c.get('observable')} | outcome: {c.get('outcome_raw')}"
                        for i, c in enumerate(cards))
    groups, how = None, 'exact'
    if len(cards) > 1:
        got = json_block(model.complete([{'role': 'system', 'content': SYSTEM},
                                         {'role': 'user', 'content': DEDUP.format(cards=listing)}], phase=f'{paper_id}:dedup'), 'groups')
        try:
            got = [[int(str(x).lstrip('R')) for x in g] for g in got]
            if sorted(i for g in got for i in g) == list(range(len(cards))):
                groups, how = got, 'llm'
        except (TypeError, ValueError):
            pass
    groups = groups or exact_groups(cards)
    out = []
    for g in groups:
        members = [cards[i] for i in g]
        keep = max(members, key=lambda c: ((c.get('location') or {}).get('file') == 'si', len(c.get('outcome_raw') or '')))
        keep = dict(keep, also_reported=[m.get('location') for m in members if m is not keep])
        out.append(keep)
    out.sort(key=lambda c: ((c.get('location') or {}).get('file') != 'main', (c.get('location') or {}).get('page') or 0))
    for i, c in enumerate(out):
        c['id'] = f'E{i + 1:02d}'
    return out, how


def extract(paper_id, model_name, base_url):
    bins = '\n'.join(f'  {k}: {" | ".join(v)}' for k, v in SCHEMA['bins'].items())
    types = ', '.join(TYPES)
    out_dir = OUT / model_name
    model = C.Model(model_name, out_dir / 'trace.jsonl', base_url=base_url, max_tokens=16384, temperature=0.0)
    cards, stats, failed = [], {}, []
    for file in ('main', 'si'):
        path = PAPERS / paper_id / f'{file}.pdf'
        if not path.exists():
            continue
        pages = pdf_pages(path)
        keep, n_hits = select_pages(pages)
        stats[file] = {'pages': len(pages), 'keyword_pages': n_hits, 'sent_pages': len(keep)}
        for group in chunks(keep):
            text = '\n'.join(f'[page {i + 1}]\n{pages[i]}' for i in group)
            prompt = TASK.format(file='main text' if file == 'main' else 'Supporting Information', bins=bins,
                                 types=types, pages=text)
            for attempt in range(3):  # small models occasionally emit broken JSON; ask again
                answer = model.complete([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],
                                        phase=f'{paper_id}:{file}:{group[0] + 1}-{group[-1] + 1}:try{attempt}')
                got = parse_cards(answer)
                if got is not None:
                    break
            if got is None:
                failed.append(f'{file}:{group[0] + 1}-{group[-1] + 1}')
                continue
            for c in got:
                c['location'] = {**(c.get('location') or {}), 'file': file}
            cards += got
    merged, how = merge(cards, model, paper_id)
    result = {'paper_id': paper_id, 'model': model_name, 'page_stats': stats, 'failed_chunks': failed,
              'n_raw_cards': len(cards), 'merge': how, 'cards': merged, 'raw_cards': cards, 'tokens': model.tokens}
    (out_dir / f'{paper_id}.json').write_text(json.dumps(result, indent=1, ensure_ascii=False))
    return paper_id, len(result['cards']), failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--papers', nargs='*', help='paper ids under papers/ (default: all)')
    ap.add_argument('--models', nargs='+', default=[C.MODELS[0]])
    ap.add_argument('--base-url', default=C.DEFAULT_BASE_URL)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--pages-only', action='store_true', help='report page selection without calling the model')
    cli = ap.parse_args()
    ids = cli.papers or sorted(p.name for p in PAPERS.iterdir() if p.is_dir())
    if cli.pages_only:
        for pid in ids:
            for file in ('main', 'si'):
                path = PAPERS / pid / f'{file}.pdf'
                if path.exists():
                    pages = pdf_pages(path)
                    keep, n_hits = select_pages(pages)
                    print(f'{pid} {file}: {len(pages)} pages, {n_hits} keyword pages, {len(keep)} sent, '
                          f'{len(chunks(keep))} chunks')
        return
    jobs = [(pid, m) for m in cli.models for pid in ids if not (OUT / m / f'{pid}.json').exists()]
    with concurrent.futures.ThreadPoolExecutor(cli.workers) as pool:
        for fut in concurrent.futures.as_completed([pool.submit(extract, pid, m, cli.base_url) for pid, m in jobs]):
            pid, n, failed = fut.result()
            print(f'{pid}: {n} cards' + (f', unparsed chunks {failed}' if failed else ''), flush=True)


if __name__ == '__main__':
    main()
