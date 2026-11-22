"""Layer 2 (ability) sample: PDF papers spread over the same citation bins as layer 1.

Per citation bin up to PER_BIN candidates from the in-PMC part of pool.json, in this order: papers of the
lit_old gold set (their argon caches are reused, no new calls), layer-1 PDF papers, then random others (seed).
PDFs from the PMC OA bucket; lit_old PDFs and argon caches (extract2/repair2) are copied into the lit_scale tree.

python tier2_select.py      -> tier2.json
"""
import json
import math
import random
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'lit150'))
sys.path.insert(0, str(HERE.parent / 'pilots_v4'))
import select_download as SD  # noqa: E402
import common4 as K  # noqa: E402

SD.PAPERS = HERE / 'papers'
SD.OLD = HERE / '_none'
PER_BIN, SEED = 60, 20261108
OLD_GOLD = K.OUT / 'lit_old_argon'
NEW_GOLD = K.OUT / 'lit_scale_gold'


def main():
    sample = json.loads((HERE / 'sample.json').read_text())
    last = sample['bins']['last_open_bin']
    tier1_pdf = {p['id'] for p in sample['papers'] if p['source'] == 'pdf'}
    old = {f.stem for f in (OLD_GOLD / 'gold2').glob('*.json')}
    pool = [p for p in json.loads((HERE / 'pool.json').read_text())
            if p['in_pmc'] and p['pmcid'] and p.get('oa_cites_to2023') is not None]
    for p in pool:
        p['id'] = p['pmcid']
        p['bin'] = min(int(math.log10(1 + p['oa_cites_to2023']) / 0.25), last)
    rng = random.Random(SEED)
    rng.shuffle(pool)
    pool.sort(key=lambda p: (p['id'] not in old, p['id'] not in tier1_pdf))
    chosen, per = [], Counter()
    for p in pool:
        if per[p['bin']] < PER_BIN:
            chosen.append(p)
            per[p['bin']] += 1
    for p in chosen:
        if p['id'] in old:
            src = HERE.parent / 'lit_old' / 'papers' / p['id'] / 'main.pdf'
            dst = HERE / 'papers' / p['id'] / 'main.pdf'
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)
            for tag in ('extract2', 'repair2'):
                s, d = OLD_GOLD / tag / f"{p['id']}.json", NEW_GOLD / tag / f"{p['id']}.json"
                if s.exists() and not d.exists():
                    d.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(s, d)
    with ThreadPoolExecutor(8) as ex:
        status = dict(ex.map(SD.fetch_pdf, [p['id'] for p in chosen]))
    keep = [p for p in chosen if status.get(p['id']) in ('ok', 'cached')]
    for p in keep:
        p.pop('abstract', None)
    (HERE / 'tier2.json').write_text(json.dumps({'rule': __doc__, 'papers': keep}, ensure_ascii=False))
    print(json.dumps({'chosen': len(chosen), 'with_pdf': len(keep), 'reused_old_gold': sum(p['id'] in old for p in keep),
                      'by_bin': sorted(Counter(p['bin'] for p in keep).items())}, indent=1))


if __name__ == '__main__':
    main()
