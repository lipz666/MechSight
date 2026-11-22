"""P0 scoring: LLM evidence cards vs hand-annotated cards.

Hand cards: annotations/<paper_id>.json = {"paper_id": ..., "cards": [...]} in
the schema.json format. LLM cards: the extract.py output.

Matching is greedy one-to-one on token Jaccard of perturbation + observable,
with a bonus for the same experiment_type; pairs below MATCH_MIN stay
unmatched. Every pair is written to OUT/<model>/matches.jsonl for review, so
the matching itself can be audited before trusting the numbers.

Reported per model (pooled over papers, plus per paper):
  card recall / precision      matched / hand cards, matched / LLM cards
  type accuracy                same experiment_type among matched pairs
  bin accuracy                 same outcome_bin among matched pairs whose hand bin is not null
  leakage                      LLM outcome_raw containing interpretive wording
  quote found                  LLM quote present in the PDF text (whitespace/case-normalised)
  hideable per paper           cards with a non-null bin (hand cards where annotated, else LLM cards)
  strong per paper             hideable cards of the STRONG types (no controls, quenching, spectroscopy, CV)
  bins by type                 pooled bin counts per type; the majority share is what a base-rate predictor scores
P0 gates (doc/evidence_driven_mechanism_20261101.md): >=5 hideable/paper,
recall >= 0.85, bin accuracy >= 0.85, leakage <= 0.10.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

from extract import OUT, PAPERS, HERE, canonical_bin, pdf_pages

MATCH_MIN = 0.3
STRONG = {'kie', 'radical_trap', 'radical_clock', 'light_on_off', 'quantum_yield', 'crossover', 'isotope_label', 'hammett',
          'kinetic_order', 'nonlinear_effect', 'stoichiometric_intermediate', 'stereochemical_probe', 'competition'}
TYPE_BONUS = 0.3
INTERPRETIVE = re.compile(
    r'suggest|indicat|consistent with|support|impl(y|ies)|rule[sd]? out|excluded|likely|presumabl|plausibl|'
    r'radical (pathway|mechanism|process)|rate.(determining|limiting)|involv|via |mechanism', re.I)


def tokens(card):
    return set(re.findall(r'[a-z0-9]+', f"{card.get('perturbation', '')} {card.get('observable', '')}".lower()))


def similarity(a, b):
    ta, tb = tokens(a), tokens(b)
    jac = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    return jac + (TYPE_BONUS if a.get('experiment_type') == b.get('experiment_type') else 0.0)


def match(hand, llm):
    pairs = sorted(((similarity(h, l), i, j) for i, h in enumerate(hand) for j, l in enumerate(llm)), reverse=True)
    used_h, used_l, out = set(), set(), []
    for s, i, j in pairs:
        if s < MATCH_MIN or i in used_h or j in used_l:
            continue
        used_h.add(i)
        used_l.add(j)
        out.append((i, j, s))
    return out


def squash(text):
    return re.sub(r'\s+', ' ', (text or '').lower()).strip()


def pdf_text(pid):
    return squash(' '.join(' '.join(pdf_pages(p)) for p in (PAPERS / pid).glob('*.pdf')))


def binned(cards, types=None):
    return sum(1 for c in cards if c.get('outcome_bin') and (types is None or c['experiment_type'] in types))


def quote_found(quote, text):
    """Quote counts as found when >= 60% of its word 4-grams occur in the PDF text (pdftotext breaks lines/hyphens)."""
    words = squash(quote).split()
    grams = [' '.join(words[i:i + 4]) for i in range(max(1, len(words) - 3))]
    return bool(words) and sum(g in text for g in grams) >= 0.6 * len(grams)


def score(model):
    rows, log = [], []
    for path in sorted((OUT / model).glob('*.json')):
        pid = path.stem
        llm = json.loads(path.read_text())['cards']
        hand_path = HERE / 'annotations' / f'{pid}.json'
        hand = json.loads(hand_path.read_text())['cards'] if hand_path.exists() else None
        text = pdf_text(pid)
        row = {'paper': pid, 'llm_cards': len(llm),
               'leak': sum(bool(INTERPRETIVE.search(c.get('outcome_raw') or '')) for c in llm),
               'quote_found': sum(quote_found(c.get('quote'), text) for c in llm),
               'hideable': binned(hand if hand is not None else llm),
               'hideable_strong': binned(hand if hand is not None else llm, STRONG), 'annotated': hand is not None,
               'cards': llm}
        if hand is not None:
            pairs = match(hand, llm)
            row.update(hand_cards=len(hand), matched=len(pairs),
                       type_ok=sum(hand[i]['experiment_type'] == llm[j]['experiment_type'] for i, j, _ in pairs),
                       bin_n=sum(bool(hand[i].get('outcome_bin')) for i, j, _ in pairs),
                       bin_ok=sum(bool(hand[i].get('outcome_bin')) and squash(canonical_bin(hand[i]['outcome_bin'])) == squash(llm[j].get('outcome_bin'))
                                  for i, j, _ in pairs))
            hit_h, hit_l = {i for i, _, _ in pairs}, {j for _, j, _ in pairs}
            log += [{'paper': pid, 'sim': round(s, 2), 'hand': hand[i], 'llm': llm[j]} for i, j, s in pairs]
            log += [{'paper': pid, 'missed': h} for i, h in enumerate(hand) if i not in hit_h]
            log += [{'paper': pid, 'extra': l} for j, l in enumerate(llm) if j not in hit_l]
        rows.append(row)
    with (OUT / model / 'matches.jsonl').open('w') as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return rows


def ratio(a, b):
    return f'{a / b:.2f} ({a}/{b})' if b else 'n/a'


def report(model, rows):
    ann = [r for r in rows if r['annotated']]
    tot = lambda key, rs=ann: sum(r[key] for r in rs)
    llm_total = tot('llm_cards', rows)
    print(f'== {model}: {len(rows)} papers, {len(ann)} annotated')
    print(f"  card recall      {ratio(tot('matched'), tot('hand_cards'))}")
    print(f"  card precision   {ratio(tot('matched'), tot('llm_cards'))}")
    print(f"  type accuracy    {ratio(tot('type_ok'), tot('matched'))}")
    print(f"  bin accuracy     {ratio(tot('bin_ok'), tot('bin_n'))}")
    print(f"  leakage          {ratio(tot('leak', rows), llm_total)}")
    print(f"  quote found      {ratio(tot('quote_found', rows), llm_total)}")
    for key, label in (('hideable', 'hideable/paper  '), ('hideable_strong', 'strong/paper    ')):
        hide = [r[key] for r in rows]
        if hide:
            print(f'  {label} median {statistics.median(hide)}, min {min(hide)}, '
                  f'papers >=5: {sum(h >= 5 for h in hide)}/{len(hide)}')
    print('  outcome bins by type (all LLM cards; majority share = base-rate predictor accuracy):')
    by_type = collections.defaultdict(collections.Counter)
    for r in rows:
        for c in r['cards']:
            if c.get('outcome_bin'):
                by_type[c['experiment_type']][c['outcome_bin'].lower()] += 1
    for t, cnt in sorted(by_type.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(cnt.values())
        print(f"    {t:28s} n={n:3d} majority {cnt.most_common(1)[0][1] / n:.0%}  {dict(cnt.most_common(4))}")
    for r in rows:
        extra = f", recall {ratio(r['matched'], r['hand_cards'])}, bins {ratio(r['bin_ok'], r['bin_n'])}" if r['annotated'] else ''
        print(f"    {r['paper']}: {r['llm_cards']} LLM cards, {r['hideable']} hideable, {r['hideable_strong']} strong{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+')
    cli = ap.parse_args()
    models = cli.models or (sorted(p.name for p in OUT.iterdir() if p.is_dir()) if OUT.exists() else [])
    for m in models:
        report(m, score(m))


if __name__ == '__main__':
    main()
