"""Layer-1 sample, wave 2 (decided 2026-11-07 after wave-1 extraction showed ~37 % eligibility, before any R1 call).

Adds, per citation bin, up to EXTRA further papers not yet sampled (bins >= 10: every remaining paper), with the same
period balancing and seed scheme as wave 1 (seed + 2). PDFs for in-PMC papers. Appends to sample.json with wave = 2.

python sample_wave2.py
"""
import json
import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sample_scale as S

HERE = Path(__file__).resolve().parent
EXTRA = 120


def main():
    sm = json.loads((HERE / 'sample.json').read_text())
    if any(p.get('wave') == 2 for p in sm['papers']):
        print('wave 2 already sampled')
        return
    last = sm['bins']['last_open_bin']
    have = {p['doi'] for p in sm['papers']}
    pool = [p for p in json.loads((HERE / 'pool.json').read_text())
            if p.get('oa_cites_to2023') is not None and 2000 <= p['year'] <= 2023 and p['doi'] not in have]
    for p in pool:
        p['id'] = p['pmcid'] or (f"PMID{p['pmid']}" if p['pmid'] else p['doi'].replace('/', '_'))
        p['logc'] = math.log10(1 + p['oa_cites_to2023'])
        p['bin'] = min(int(p['logc'] / S.WIDTH), last)
    rng = random.Random(S.SEED + 2)
    cells = defaultdict(list)
    for p in pool:
        cells[(p['bin'], S.period(p['year']))].append(p)
    for v in cells.values():
        rng.shuffle(v)
    add = []
    for b in range(last + 1):
        quota = 10 ** 6 if b >= 10 else EXTRA
        lists = [cells[(b, k)] for k in range(4)]
        take = [0] * 4
        while quota and any(take[k] < len(lists[k]) for k in range(4)):
            for k in range(4):
                if quota and take[k] < len(lists[k]):
                    take[k] += 1
                    quota -= 1
        for k in range(4):
            add += lists[k][:take[k]]
    ids = [p['pmcid'] for p in add if p['in_pmc'] and p['pmcid']]
    with ThreadPoolExecutor(8) as ex:
        status = dict(ex.map(S.SD.fetch_pdf, ids))
    for p in add:
        p['source'] = 'pdf' if status.get(p['pmcid']) in ('ok', 'cached') else 'abstract'
        p['wave'] = 2
    sm['papers'] += add
    sm['wave2_rule'] = __doc__
    (HERE / 'sample.json').write_text(json.dumps(sm, ensure_ascii=False))
    print(json.dumps({'added': len(add), 'total': len(sm['papers']),
                      'by_bin': sorted(Counter(p['bin'] for p in add).items())}))


if __name__ == '__main__':
    main()
