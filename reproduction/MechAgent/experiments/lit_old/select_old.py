"""Select the old exposure strata: full-text mechanistic-experiment categories (lit150 rules, >= 3), OpenAlex
citation cross-check, bio terms re-checked on the full text's title/abstract, main PDF from the PMC OA bucket.

Stratum (decided on min(EuropePMC, OpenAlex) for high and max(...) for low, i.e. both sources must agree):
  high  >= 100 citations        low  <= 15 citations        (mid kept as reserve, not downloaded)
Writes selected_old.json and papers/<PMCID>/main.pdf. Generic User-Agent, no contact data.
"""
import json
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'lit150'))
import select_download as SD  # noqa: E402  (CATS, cats_of, get, bucket fetch)

SD.PAPERS = HERE / 'papers'
SD.OLD = HERE / '_none'
MIN_CATS = 3
UA = {'User-Agent': 'mech-origin-study/1.0'}


def openalex(doi):
    if not doi:
        return doi, None
    try:
        req = urllib.request.Request(f'https://api.openalex.org/works/doi:{doi}', headers=UA)
        return doi, json.loads(urllib.request.urlopen(req, timeout=60).read()).get('cited_by_count')
    except Exception:
        return doi, None


def band(r):
    lo = min(c for c in (r['cited_by'], r['openalex']) if c is not None)
    hi = max(c for c in (r['cited_by'], r['openalex']) if c is not None)
    return 'high' if lo >= 100 else 'low' if hi <= 15 else 'mid'


def main():
    cands = json.loads((HERE / 'candidates_old.json').read_text())
    cache = HERE / 'cats_cache_old.json'
    cats = json.loads(cache.read_text()) if cache.exists() else {}
    with ThreadPoolExecutor(6) as ex:
        cats.update(dict(ex.map(SD.cats_of, [c['pmcid'] for c in cands if cats.get(c['pmcid']) is None])))
    cache.write_text(json.dumps(cats))
    oa_cache = HERE / 'openalex_cache.json'
    oa = json.loads(oa_cache.read_text()) if oa_cache.exists() else {}
    with ThreadPoolExecutor(6) as ex:
        oa.update(dict(ex.map(openalex, [c['doi'] for c in cands if c.get('doi') and c['doi'] not in oa])))
    oa_cache.write_text(json.dumps(oa))
    rows = []
    for c in cands:
        r = dict(c, cats=cats.get(c['pmcid']), n_cats=len(cats.get(c['pmcid']) or []), openalex=oa.get(c.get('doi')))
        r['stratum'] = band(r)
        rows.append(r)
    strong = [r for r in rows if r['n_cats'] >= MIN_CATS]
    print('candidates', len(rows), 'without full text', sum(r['cats'] is None for r in rows),
          'with >=3 categories', len(strong), Counter(r['stratum'] for r in strong))
    pick = [r for r in strong if r['stratum'] in ('high', 'low')]
    with ThreadPoolExecutor(4) as ex:
        status = dict(ex.map(SD.fetch_pdf, [r['pmcid'] for r in pick]))
    for r in pick:
        r['pdf'] = status[r['pmcid']]
    (HERE / 'selected_old.json').write_text(json.dumps(pick, indent=1, ensure_ascii=False))
    (HERE / 'reserve_mid.json').write_text(json.dumps([r for r in strong if r['stratum'] == 'mid'], indent=1,
                                                      ensure_ascii=False))
    ok = [r for r in pick if r['pdf'] in ('ok', 'cached')]
    print('pdf status', Counter(status.values()), 'usable by stratum', Counter(r['stratum'] for r in ok))


if __name__ == '__main__':
    main()
