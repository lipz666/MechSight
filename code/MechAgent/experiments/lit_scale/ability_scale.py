"""Literature-memory scaling study, layer 2: mechanism ability vs exposure on the same papers.

Gold: the unchanged p29 argon-build pipeline on eligible PDF papers of the sample (scope rule of build_gold_old:
largest all-organic product fragment >= 6 heavy atoms). Arms (gemini-3.8-flash, official prompt with products,
T = 1.0, 4 samples each): A (reaction only) and C (reaction + citation, "write the mechanism this paper proposed").
Ability = strict recall of gold substrate intermediates. Exponent fitted exactly like the memory curve
(binomial over gold species x samples, p = A (1+c)^b), so memory and ability exponents are directly comparable.

python ability_scale.py --phase gold --workers 100
python ability_scale.py --phase run --workers 200
python ability_scale.py --phase analyze
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pilots_v4'))
sys.path.insert(0, str(HERE.parent / 'lit_old'))
import common4 as K  # noqa: E402
import p29_gold_rebuild as P29  # noqa: E402
import p30_lit172_gold as P30  # noqa: E402
import p33_m1_m4 as P33  # noqa: E402
import analyze_scale as AS  # noqa: E402
import r1_scale as R  # noqa: E402

P29.PAPERS = HERE / 'papers'
P29.OUTD = K.OUT / 'lit_scale_gold'
NS, SEED = 4, 20261107


def candidates():
    """Layer-2 PDF papers (tier2_select.py); eligibility is decided by the gold pipeline itself."""
    return json.loads((HERE / 'tier2.json').read_text())['papers']


def organic_size(smis):
    """Same rule as lit_old/build_gold_old.py (not imported: that module re-points P29 at lit_old)."""
    from rdkit import Chem
    org = set('C N O S P F Cl Br I B Si Se'.split()) | {'H'}
    best = 0
    for s in smis or []:
        m = Chem.MolFromSmiles(s or '')
        for fr in (Chem.GetMolFrags(m, asMols=True) if m else []):
            syms = {a.GetSymbol() for a in fr.GetAtoms()}
            if 'C' in syms and syms <= org:
                best = max(best, fr.GetNumHeavyAtoms())
    return best


def cases():
    ps = {p['id']: p for p in candidates()}
    P30.DS, P30.IDS, P30.GOLD.OUTD = 'litscale', list(ps), P29.OUTD
    cs = {}
    for cid, c in P30.cases().items():
        rec = json.loads((P29.OUTD / 'gold2' / f'{cid}.json').read_text())
        if organic_size(rec['record']['model_reaction'].get('products')) >= 6:
            p = ps[cid]
            cs[cid] = dict(c, meta={'authorString': p['authorString'], 'title': p['title'], 'journal': p['journal'],
                                    'year': p['year']}, c=p['oa_cites_to2023'], year=p['year'])
    return cs


MODEL = 'flash36'  # gemini-3.6-flash on the 117 gateway (switched from 3.8-flash on 2026-11-07, same as layer 1)


def tag(arm, s):
    return f'{arm}36_s{s}'


def answer(arm, s, cid):
    f = K.OUT / 'lit_scale' / tag(arm, s) / f'{cid}.json'
    return json.loads(f.read_text())['answer'] if f.exists() else None


def call36(arm, s, cid, prompt):
    """gemini-3.6-flash on the 117 gateway; the official prompt answers with a bare JSON list (no [ANSWER] tag),
    so an answer is accepted when it parses as a mechanism."""
    import os
    import time
    from openai import OpenAI
    dest = K.OUT / 'lit_scale' / tag(arm, s) / f'{cid}.json'
    if dest.exists():
        return json.loads(dest.read_text())['answer']
    dest.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=os.environ['FALLBACK_API_KEY'], base_url=R.BASE, timeout=900, max_retries=0)
    for attempt in range(6):
        try:
            with R._SEM[MODEL]:
                r = client.chat.completions.create(model=R.MODELS[MODEL], temperature=1.0, max_tokens=65536,
                                                   messages=[{'role': 'system', 'content': K.SYSTEM},
                                                             {'role': 'user', 'content': prompt}])
            ans = r.choices[0].message.content or ''
            if not ans.strip():
                raise RuntimeError('empty answer')
            dest.write_text(json.dumps({'id': cid, 'tag': tag(arm, s), 'model': R.MODELS[MODEL],
                                        'returned_model': r.model, 'temperature': 1.0, 'prompt': prompt,
                                        'answer': ans}, ensure_ascii=False, indent=1))
            return ans
        except Exception:  # noqa: BLE001
            time.sleep(min(60, 5 * 2 ** attempt))
    return None


def job(arm, c, s):
    p = K.official_prompt(c)
    if arm == 'C':
        p = p.replace('\n\n## Input', P33.CITE.format(cite=P33.citation(c['meta'])), 1)
    return 'ok' if call36(arm, s, c['id'], p) else 'error'


def analyze():
    cs = cases()
    rows = []
    for cid, c in cs.items():
        ks = [k for k in (K.species_keys(s) for s in c['author']) if k]
        if not ks:
            continue
        r = {'id': cid, 'c': c['c'], 'year': c['year'], 'G': len(ks)}
        for arm in ('A', 'C'):
            hits = []
            for s in range(1, NS + 1):
                m = K.parse_mech(answer(arm, s, cid))
                ps = P30.V.keys_of(m)
                hits.append(sum(bool(k & ps) for k in ks))
            r[arm] = hits
        rows.append(r)
    rep = {'papers': len(rows)}
    for arm in ('A', 'C'):
        d = {'c': np.array([r['c'] for r in rows], float), 'k': np.array([sum(r[arm]) for r in rows], float),
             'n': np.array([r['G'] * NS for r in rows], float), 'y': np.array([r['year'] - 2012 for r in rows], float)}
        d['x'] = np.log1p(d['c'])
        t, _ = AS.fit('power', d)
        rng = random.Random(SEED)
        bs = []
        for _ in range(1000):
            s = np.array([rng.randrange(len(rows)) for _ in rows])
            bs.append(AS.fit('power', AS.sub(d, s), t)[0][1])
        rep[f'{arm}_recall'] = round(100 * float(d['k'].sum() / d['n'].sum()), 2)
        rep[f'{arm}_exponent'] = [round(float(t[1]), 3)] + [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])]
    diff = [(sum(r['C']) - sum(r['A'])) / (r['G'] * NS) for r in rows]
    rng = random.Random(SEED)
    bs = [np.mean([diff[rng.randrange(len(diff))] for _ in diff]) for _ in range(2000)]
    rep['memory_index_C_minus_A'] = [round(100 * float(np.mean(diff)), 2)] + [round(100 * float(x), 2)
                                                                            for x in np.percentile(bs, [2.5, 97.5])]
    x = np.log10(1 + np.array([r['c'] for r in rows]))
    X = np.c_[np.ones(len(rows)), x, np.array([r['year'] for r in rows]) - 2012]
    sl = []
    for _ in range(1000):
        s = [rng.randrange(len(rows)) for _ in rows]
        sl.append(np.linalg.lstsq(X[s], np.array(diff)[s], rcond=None)[0][1])
    rep['memory_index_slope_per_10x'] = [round(100 * float(np.linalg.lstsq(X, np.array(diff), rcond=None)[0][1]), 2)] + [
        round(100 * float(v), 2) for v in np.percentile(sl, [2.5, 97.5])]
    (K.OUT / 'lit_scale' / 'SUMMARY_SCALE_ABILITY.json').write_text(json.dumps({'summary': rep, 'rows': rows},
                                                                              indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['gold', 'run', 'analyze'], required=True)
    ap.add_argument('--workers', type=int, default=100)
    cli = ap.parse_args()
    if cli.phase == 'gold':
        K.run_pool(P29.job, [(p['id'],) for p in candidates()], cli.workers)
        print(json.dumps({'candidates': len(candidates()), 'usable_in_scope': len(cases())}))
    elif cli.phase == 'run':
        cs = list(cases().values())
        jobs = [(arm, c, s) for arm in ('A', 'C') for c in cs for s in range(1, NS + 1)]
        for _ in range(3):
            K.run_pool(job, jobs, cli.workers)
            jobs = [j for j in jobs if answer(j[0], j[2], j[1]['id']) is None]
            if not jobs:
                break
    else:
        analyze()


if __name__ == '__main__':
    main()
