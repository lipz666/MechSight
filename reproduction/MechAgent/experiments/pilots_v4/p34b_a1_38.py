"""P34b: the pilot winner A1 (v7.1 + insertion of the left-out >=3/7 consensus species) on gemini-3.8-flash, dev set.

Reads the cached 3.8 v7.1 revisions in lit172g/G71_rev and writes lit172g/A1_place (one call per paper with left-out
consensus species). Run with PILOT_MODEL=gemini-3.8-flash.
"""
import json
import os

import common4 as K
import p28_lit172_v7 as G
import p29_gold_rebuild as GOLD
import p30_lit172_gold as P
import p31_guided71 as T
import p34_pilot36 as X

assert X.MODEL == 'gemini-3.8-flash', X.MODEL
X.PDS = 'lit172g'
RUN = K.OUT / 'p34b_a1_38_20261106'


def main():
    T.configure('dev')
    G.RUN, G.STATE = RUN, RUN / 'status.json'
    RUN.mkdir(parents=True, exist_ok=True)
    cs = P.cases()
    P.CS = cs
    lst = [c for c in cs.values() if c['clues'] and any(K.species_keys(s) for s in c['author'])]
    assert os.environ.get('FALLBACK_API_KEY')
    G.phase('a1_38', X.job, [('A1', c) for c in lst], 200)
    rec = P.records()
    rows = {}
    for c in lst:
        cid = c['id']
        refs = GOLD.substrate_refs(rec[cid]['record'])
        g = K.cached('lit172g/G71_rev', cid)
        a1 = K.cached('lit172g/A1_place', cid)
        arms = {'Direct': K.parse_mech(K.cached('lit172g/base_default', cid)),
                'Dclues': K.parse_mech(K.cached('lit172g/O_llm', cid)),
                'SC15': K.parse_mech(K.cached('lit172g/SC15_rev', cid)),
                'G71': K.parse_mech(g), 'A0': X.union(c, g), 'A1': K.parse_mech(a1) if a1 else K.parse_mech(g)}
        for a, m in arms.items():
            rows.setdefault(a, []).append((cid, T.score(m, c, refs)))
    out = {}
    for split in ('tune', 'check', 'all'):
        sub = {a: [r for r in rs if split == 'all' or X.tune(r[0]) == (split == 'tune')] for a, rs in rows.items()}
        out[split] = {'arms': {a: K.summarize(r, ['recall', 'precision', 'f1', 'n_pred']) for a, r in sub.items()},
                      'diffs': {f'{h}-{l}': {k: K.paired(sub[h], sub[l], k) for k in ('recall', 'precision', 'f1')}
                                for h, l in [('A1', 'Dclues'), ('A1', 'SC15'), ('A1', 'G71'), ('A1', 'Direct'),
                                             ('G71', 'Dclues')]}}
    (RUN / 'SUMMARY_P34B.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
