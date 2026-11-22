"""Batch 2 of the Guided v7.1 confirmation set: more 2026 open-access mechanistic papers (fixed rule, 2026-11-06).

Rule (written before any batch-1 or batch-2 result was seen):
  Europe PMC, OPEN_ACCESS, IN_PMC, not a review, first published 2026-01-01..2026-11-05,
  abstract mentions mechanistic/mechanism AND catalysis/photoredox/radical/photocatalysis/electrochemistry,
  journal in the list below (lit150 list + common organic/organometallic journals);
  title exclusions as lit150 (heterogeneous, materials, enzyme, battery, ...);
  not already in experiments/lit150/papers, evidence_p0 or the lit172/batch-1 manifests;
  full-text XML has >= 3 mechanistic-experiment categories (lit150 CATS);
  ranked by number of categories (desc), then PMCID; the first TARGET whose main PDF is in the PMC OA bucket.
PDFs go to experiments/lit_more/papers/<PMCID>/main.pdf (data only: read as bytes and sent to argon-build).
Run with HTTPS_PROXY set if needed.
"""
import json
import re
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / 'lit150'))
import search as S  # noqa: E402
import select_download as D  # noqa: E402

PAPERS = HERE / 'papers'
TARGET = 260
MIN_CATS = 3
JOURNALS = S.JOURNALS + ['ACS Catal', 'Adv Sci (Weinh)', 'CCS Chem', 'Org Biomol Chem', 'RSC Adv', 'Molecules',
                         'iScience', 'Natl Sci Rev', 'Chem Asian J', 'Eur J Org Chem', 'Adv Synth Catal',
                         'ChemCatChem', 'Inorg Chem', 'Organometallics', 'Dalton Trans', 'ACS Omega',
                         'Chempluschem', 'ChemSusChem', 'ACS Org Inorg Au', 'Synthesis (Stuttg)', 'Chem Eur J',
                         'Research (Wash D C)', 'Sci China Chem', 'Chin J Chem', 'Exploration (Beijing)']
QUERY = ('(ABSTRACT:mechanistic OR ABSTRACT:mechanism) AND (ABSTRACT:catalys* OR ABSTRACT:photoredox OR '
         'ABSTRACT:radical OR ABSTRACT:photocataly* OR ABSTRACT:electrochem*) AND OPEN_ACCESS:y AND IN_PMC:y '
         'AND NOT PUB_TYPE:"review" AND FIRST_PDATE:[2026-01-01 TO 2026-11-05] AND ('
         + ' OR '.join(f'JOURNAL:"{j}"' for j in dict.fromkeys(JOURNALS)) + ')')


def excluded_ids():
    out = {p.name for p in (EXP / 'lit150' / 'papers').iterdir()}
    old = EXP / 'evidence_p0' / 'papers'
    if old.exists():
        out |= {p.name for p in old.iterdir()}
    o = EXP.parent / 'outputs' / 'pilots_v4_20261103'
    out |= set(json.loads((o / 'lit172_v7_real_20261105' / 'MANIFEST.json').read_text())['ids'])
    for b in json.loads((o / 'litnew_confirm_g71' / 'MANIFEST.json').read_text())['batches']:
        out |= set(b['ids'])
    return out


def search():
    out, cursor = [], '*'
    while True:
        q = urllib.parse.urlencode({'query': QUERY, 'format': 'json', 'pageSize': 1000, 'resultType': 'lite',
                                    'cursorMark': cursor})
        d = json.loads(D.get(f'{S.API}?{q}'))
        for r in d['resultList']['result']:
            if r.get('pmcid') and not S.EXCLUDE.search(r.get('title', '')):
                out.append({'pmcid': r['pmcid'], 'doi': r.get('doi'), 'date': r.get('firstPublicationDate'),
                            'journal': r.get('journalTitle'), 'title': r.get('title')})
        nxt = d.get('nextCursorMark')
        if not nxt or nxt == cursor or not d['resultList']['result']:
            break
        cursor = nxt
    print('hits', d.get('hitCount'), 'kept after title filter', len(out), flush=True)
    return out


def fetch(pmcid):
    D.PAPERS, D.OLD = PAPERS, HERE / '_none'
    return D.fetch_pdf(pmcid)


def main():
    HERE.mkdir(exist_ok=True)
    cands = [c for c in search() if c['pmcid'] not in excluded_ids()]
    (HERE / 'candidates.json').write_text(json.dumps(cands, indent=1, ensure_ascii=False))
    cache = HERE / 'cats_cache.json'
    cats = {k: v for k, v in (json.loads(cache.read_text()).items() if cache.exists() else []) if v is not None}
    with ThreadPoolExecutor(8) as ex:
        cats.update(dict(ex.map(D.cats_of, [c['pmcid'] for c in cands if c['pmcid'] not in cats])))
    cache.write_text(json.dumps(cats))
    rows = [{**c, 'cats': cats.get(c['pmcid']), 'n_cats': len(cats.get(c['pmcid']) or [])} for c in cands]
    strong = sorted((r for r in rows if r['n_cats'] >= MIN_CATS), key=lambda r: (-r['n_cats'], r['pmcid']))
    print(f'{len(rows)} new candidates, {sum(r["cats"] is None for r in rows)} without XML, '
          f'{len(strong)} with >= {MIN_CATS} categories', flush=True)
    pick, status = [], {}
    with ThreadPoolExecutor(6) as ex:
        for i in range(0, len(strong), 12):  # rank order, stop once TARGET PDFs are in hand
            chunk = strong[i:i + 12]
            for r, (_, st) in zip(chunk, ex.map(fetch, [r['pmcid'] for r in chunk])):
                status[r['pmcid']] = st
                if st in ('ok', 'cached') and len(pick) < TARGET:
                    pick.append({**r, 'pdf': st})
            if len(pick) >= TARGET:
                break
    (HERE / 'selected.json').write_text(json.dumps(pick, indent=1, ensure_ascii=False))
    from collections import Counter
    print(Counter(status.values()), 'picked', len(pick))


if __name__ == '__main__':
    main()
