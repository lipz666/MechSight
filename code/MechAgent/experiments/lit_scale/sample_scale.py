"""Literature-memory scaling study, step 2: stratified sample and PDFs.

Exposure c = OpenAlex citations accrued up to 2023 (counts_by_year; closest to training-time exposure).
Strata: log10(1 + c) in bins of 0.25 (0-0.25, ..., last bin open) x four publication periods
(2000-2007, 2008-2013, 2014-2019, 2020-2023). Per citation bin up to PER_BIN candidates, spread as evenly as
possible over the periods; inside a cell papers are drawn at random (seed 20261107), regardless of OA status
(OA is recorded as a covariate). Sparse bins are taken whole.
PDFs: PMC OA bucket for sampled papers that are in PMC (lit150 select_download.fetch_pdf).
Calibration: 150 random PDF papers are extracted from the abstract as well.

python sample_scale.py           -> sample.json, papers/<PMCID>/main.pdf
"""
import json
import math
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'lit150'))
import select_download as SD  # noqa: E402

SD.PAPERS = HERE / 'papers'
SD.OLD = HERE / '_none'
SEED, PER_BIN, WIDTH, N_CAL = 20261107, 110, 0.25, 150
PERIODS = [(2000, 2007), (2008, 2013), (2014, 2019), (2020, 2023)]


def period(y):
    return next(i for i, (a, b) in enumerate(PERIODS) if a <= y <= b)


def main():
    pool = [p for p in json.loads((HERE / 'pool.json').read_text())
            if p.get('oa_cites_to2023') is not None and 2000 <= p['year'] <= 2023]
    for p in pool:
        p['id'] = p['pmcid'] or f"PMID{p['pmid']}" if (p['pmcid'] or p['pmid']) else p['doi'].replace('/', '_')
        p['logc'] = math.log10(1 + p['oa_cites_to2023'])
        p['bin'] = int(p['logc'] / WIDTH)
    top = max(p['bin'] for p in pool)
    # merge the sparse top bins until the open last bin has >= 30 papers
    last = top
    while sum(p['bin'] >= last for p in pool) < 30 and last > 0:
        last -= 1
    for p in pool:
        p['bin'] = min(p['bin'], last)
    rng = random.Random(SEED)
    cells = defaultdict(list)
    for p in pool:
        cells[(p['bin'], period(p['year']))].append(p)
    for v in cells.values():
        rng.shuffle(v)
    sample = []
    for b in range(last + 1):
        quota = PER_BIN
        lists = [cells[(b, k)] for k in range(4)]
        take = [0] * 4
        while quota and any(take[k] < len(lists[k]) for k in range(4)):
            for k in range(4):
                if quota and take[k] < len(lists[k]):
                    take[k] += 1
                    quota -= 1
        for k in range(4):
            sample += lists[k][:take[k]]
    ids = [p['pmcid'] for p in sample if p['in_pmc'] and p['pmcid']]
    with ThreadPoolExecutor(8) as ex:
        status = dict(ex.map(SD.fetch_pdf, ids))
    for p in sample:
        p['source'] = 'pdf' if status.get(p['pmcid']) in ('ok', 'cached') else 'abstract'
        p.pop('abstract_len', None)
    pdfs = sorted(p['id'] for p in sample if p['source'] == 'pdf')
    cal = random.Random(SEED + 1).sample(pdfs, min(N_CAL, len(pdfs)))
    (HERE / 'sample.json').write_text(json.dumps({
        'rule': __doc__, 'seed': SEED, 'bins': {'width': WIDTH, 'last_open_bin': last},
        'papers': sample, 'calibration': cal}, ensure_ascii=False))
    print(json.dumps({'pool': len(pool), 'sample': len(sample), 'pdf': len(pdfs),
                      'by_bin': sorted(Counter(p['bin'] for p in sample).items()),
                      'by_period': sorted(Counter(period(p['year']) for p in sample).items()),
                      'max_cites_to2023': max(p['oa_cites_to2023'] for p in sample)}, indent=1))


if __name__ == '__main__':
    main()
