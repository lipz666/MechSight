"""Literature-memory scaling study, step 1: candidate pool.

Europe PMC, the lit_old query (mechanism words + catalysis/radical words, same journal list, bio NOT clause) widened
to 2000-2023 and to non-OA papers (abstract only). OpenAlex citation counts (total and up to 2023 from
counts_by_year) in batches of 50 DOIs. Bio and materials titles/abstracts dropped with the lit_old regexes.
Generic User-Agent, no contact data.

python search_scale.py            -> pool.json
"""
import importlib.util
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('so', HERE.parent / 'lit_old' / 'search_old.py')
SO = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SO)
UA = {'User-Agent': 'mech-origin-study/1.0'}
API = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?'
QUERY = (SO.MECH + ' AND (ABSTRACT:catalys* OR ABSTRACT:photoredox OR ABSTRACT:radical OR ABSTRACT:photocataly*) '
         'AND FIRST_PDATE:[2000-01-01 TO 2023-12-31] AND (' + ' OR '.join(f'JOURNAL:"{j}"' for j in SO.JOURNALS)
         + f') NOT ({SO.BIO_Q})')


def get(url, tries=6):
    for a in range(tries):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180).read())
        except Exception:  # noqa: BLE001
            time.sleep(5 * (a + 1))
    raise RuntimeError(url[:120])


def europepmc():
    out, cursor = [], '*'
    while True:
        p = {'query': QUERY, 'format': 'json', 'pageSize': 1000, 'resultType': 'core', 'cursorMark': cursor}
        r = get(API + urllib.parse.urlencode(p))
        res = r['resultList']['result']
        for x in res:
            auth = [a.get('lastName') or (a.get('fullName') or '').split()[-1]
                    for a in (x.get('authorList') or {}).get('author', []) if a.get('lastName') or a.get('fullName')]
            out.append({'pmid': x.get('pmid'), 'pmcid': x.get('pmcid'), 'doi': (x.get('doi') or '').lower() or None,
                        'year': int(x.get('pubYear') or 0), 'title': x.get('title') or '',
                        'journal': ((x.get('journalInfo') or {}).get('journal') or {}).get('medlineAbbreviation')
                        or ((x.get('journalInfo') or {}).get('journal') or {}).get('title'),
                        'abstract': x.get('abstractText') or '', 'authorString': x.get('authorString'),
                        'authors': auth, 'epmc_cites': x.get('citedByCount') or 0,
                        'oa': x.get('isOpenAccess') == 'Y', 'in_pmc': x.get('inPMC') == 'Y',
                        'keywords': ' '.join((x.get('keywordList') or {}).get('keyword', []))})
        print('europepmc', len(out), '/', r['hitCount'], flush=True)
        nxt = r.get('nextCursorMark')
        if not res or not nxt or nxt == cursor:
            return out
        cursor = nxt


def openalex_batch(dois):
    f = '|'.join('https://doi.org/' + d for d in dois)
    url = ('https://api.openalex.org/works?' + urllib.parse.urlencode(
        {'filter': f'doi:{f}', 'per-page': 50, 'select': 'doi,cited_by_count,counts_by_year,publication_year'}))
    out = {}
    try:
        for w in get(url)['results']:
            d = (w.get('doi') or '').lower().removeprefix('https://doi.org/')
            to2023 = w.get('cited_by_count', 0) - sum(c['cited_by_count'] for c in w.get('counts_by_year') or []
                                                      if c['year'] > 2023)
            out[d] = {'oa_cites': w.get('cited_by_count'), 'oa_cites_to2023': max(to2023, 0)}
    except RuntimeError:
        pass
    time.sleep(0.5)
    return out


def main():
    raw = HERE / 'epmc_raw.json'
    if raw.exists():
        pool = json.loads(raw.read_text())
    else:
        pool = europepmc()
        raw.write_text(json.dumps(pool))
    seen, uniq = set(), []
    for p in pool:
        key = p['doi'] or p['pmid']
        if key and key not in seen:
            seen.add(key)
            uniq.append(p)
    bio = SO.BIO
    excl = SO.EXCLUDE
    keep, why = [], Counter()
    for p in uniq:
        text = ' '.join([p['title'], p['abstract'], p['keywords']])
        if bio.search(text):
            why['bio'] += 1
        elif excl.search(p['title']):
            why['materials/other'] += 1
        elif not p['doi']:
            why['no doi'] += 1
        else:
            keep.append(p)
    cache = HERE / 'openalex_scale.json'
    oa = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [p['doi'] for p in keep if p['doi'] not in oa]
    batches = [todo[i:i + 50] for i in range(0, len(todo), 50)]
    with ThreadPoolExecutor(3) as ex:
        for i, r in enumerate(ex.map(openalex_batch, batches)):
            oa.update(r)
            if i % 20 == 0:
                cache.write_text(json.dumps(oa))
                print('openalex', len(oa), flush=True)
    cache.write_text(json.dumps(oa))
    for p in keep:
        p.update(oa.get(p['doi']) or {'oa_cites': None, 'oa_cites_to2023': None})
    (HERE / 'pool.json').write_text(json.dumps(keep, ensure_ascii=False))
    print(json.dumps({'europepmc': len(pool), 'unique': len(uniq), 'dropped': why, 'kept': len(keep),
                      'with_openalex': sum(p['oa_cites'] is not None for p in keep),
                      'in_pmc': sum(p['in_pmc'] for p in keep)}, indent=1))


if __name__ == '__main__':
    main()
