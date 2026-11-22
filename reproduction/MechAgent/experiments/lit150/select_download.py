"""Rank candidates by mechanistic-experiment categories in the full text, then fetch main PDFs.

Same idea as evidence_p0: Europe PMC full-text XML -> keyword count of mechanistic experiment
categories; papers with >= MIN_CATS categories are kept (the original 30 always included). The
main article PDF comes from the PMC OA S3 bucket (latest version) into papers/<PMCID>/main.pdf.
Run with HTTPS_PROXY set to the system proxy.
"""
import csv
import json
import re
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPERS = HERE / 'papers'
OLD = HERE.parent / 'evidence_p0'
BUCKET = 'https://pmc-oa-opendata.s3.amazonaws.com'
MIN_CATS, TARGET = 3, 300  # 4 gave 175 PDFs -> 115 usable extractions; 3 added to reach ~150
CATS = {
    'kie': r'kinetic isotope effect|\bKIE\b|k_?H\s*/\s*k_?D',
    'radical_trap': r'TEMPO|BHT|radical scaveng|radical trap|1,1-diphenylethylene',
    'radical_clock': r'radical clock|cyclopropyl.{0,40}ring[- ]opening|ring-opening of cyclopropane',
    'light_on_off': r'light on[/-]off|on/off experiment|light/dark',
    'quantum_yield': r'quantum yield',
    'stern_volmer': r'Stern[–-]Volmer|fluorescence quenching|luminescence quenching',
    'isotope_label': r'deuterium[- ]label|\bD-label|18O[- ]label|13C[- ]label|isotop\w* label|deuterated',
    'crossover': r'crossover experiment',
    'hammett': r'Hammett',
    'kinetic_order': r'kinetic (?:studies|analysis|profile)|reaction order|rate law|first-order dependence',
    'hrms_trap': r'HRMS.{0,80}(?:adduct|intermediate|detected)|detected by HRMS|ESI-MS.{0,60}intermediate',
    'epr': r'\bEPR\b|electron paramagnetic',
    'cv': r'cyclic voltammetry|\bCV\b measurements',
    'isolation': r'intermediate (?:was|were) isolated|isolated intermediate|X-ray.{0,60}intermediate',
    'in_situ_nmr': r'in situ NMR|NMR monitoring|31P NMR.{0,60}intermediate',
    'control': r'control experiments?',
    'dft': r'\bDFT\b|density functional',
}


def get(url, timeout=120, tries=6):
    import time
    for i in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * 2 ** i)


def cats_of(pmcid):
    try:
        xml = get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML').decode('utf-8', 'ignore')
    except Exception:
        return pmcid, None
    text = re.sub(r'<[^>]+>', ' ', xml)
    return pmcid, sorted(k for k, rx in CATS.items() if re.search(rx, text, re.I))


def fetch_pdf(pmcid):
    dest = PAPERS / pmcid / 'main.pdf'
    old = OLD / 'papers' / pmcid / 'main.pdf'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return pmcid, 'cached'
    if old.exists():
        shutil.copyfile(old, dest)
        return pmcid, 'copied'
    try:
        xml = get(f'{BUCKET}/?list-type=2&prefix={pmcid}.', 60).decode()
        keys = re.findall(r'<Key>([^<]+)</Key>', xml)
        if not keys:
            return pmcid, 'not in bucket'
        v = max(int(k.split('/')[0].split('.')[1]) for k in keys)
        key = f'{pmcid}.{v}/{pmcid}.{v}.pdf'
        if key not in keys:
            return pmcid, 'no main pdf'
        tmp = dest.with_suffix('.part')
        tmp.write_bytes(get(f'{BUCKET}/{key}', 300))
        tmp.rename(dest)
        return pmcid, 'ok'
    except Exception as e:
        return pmcid, f'error {type(e).__name__}'


def main():
    cands = json.loads((HERE / 'candidates.json').read_text())
    old_ids = [r['paper_id'] for r in csv.DictReader((OLD / 'papers.csv').open())]
    meta = {c['pmcid']: c for c in cands}
    ids = sorted(set(meta) | set(old_ids))
    cache = HERE / 'cats_cache.json'
    cats = {k: v for k, v in (json.loads(cache.read_text()).items() if cache.exists() else []) if v is not None}
    with ThreadPoolExecutor(4) as ex:
        cats.update(dict(ex.map(cats_of, [i for i in ids if i not in cats])))
    cache.write_text(json.dumps(cats))
    rows = [{'pmcid': p, **meta.get(p, {}), 'cats': cats.get(p), 'n_cats': len(cats.get(p) or []),
             'in_original_30': p in old_ids} for p in ids]
    min_cats = MIN_CATS
    while min_cats > 2 and sum(r['in_original_30'] or r['n_cats'] >= min_cats for r in rows) < 170:
        min_cats -= 1
    strong = [r for r in rows if r['in_original_30'] or r['n_cats'] >= min_cats]
    strong.sort(key=lambda r: (not r['in_original_30'], -r['n_cats'], r['pmcid']))
    pick = strong[:TARGET]
    print(f'{len(rows)} candidates, {sum(r["cats"] is None for r in rows)} without XML, '
          f'{len(strong)} with >= {min_cats} categories (or original), picking {len(pick)}')
    with ThreadPoolExecutor(4) as ex:
        status = dict(ex.map(fetch_pdf, [r['pmcid'] for r in pick]))
    for r in pick:
        r['pdf'] = status[r['pmcid']]
    (HERE / 'selected.json').write_text(json.dumps(pick, indent=1, ensure_ascii=False))
    from collections import Counter
    print(Counter(status.values()))


if __name__ == '__main__':
    sys.exit(main())
