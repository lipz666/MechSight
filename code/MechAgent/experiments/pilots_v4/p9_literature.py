"""P9: real literature cases for the application (experimental clue -> MechSight -> mechanism).

Papers: the 30 open-access 2026 papers of evidence_p0 (after the model's training data; also a
contamination-free check). Phase 'extract' (one gemini-3.8-flash call per paper, main text via
pdftotext): the model reaction as SMILES, the authors' proposed key intermediates (SMILES), and
the clues their mechanistic experiments revealed (detected/trapped/isolated species, label
positions, HRMS signals) - observations only, without the authors' interpretation.
Phase 'run': per reaction, A0 (official prompt, no clue), O_llm (prompt + clues) and O_ms
(candidate pathways from 4 samples + the clues; eliminate and repair). Metric: recall of the
authors' key intermediates (sigma-graph keys); the extraction is spot-checked by hand.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess

import common4 as K
import p8_observations as P8

PAPERS = K.HERE.parent / 'evidence_p0' / 'papers'
EXTRACT = ('Below is the main text of a 2026 organic chemistry paper.\n\n{text}\n\nExtract, for the model reaction whose '
           'mechanism the authors study:\n'
           '1. "reactants": SMILES of the substrates and reagents/catalysts (list), "products": SMILES of the main product and '
           'stoichiometric by-products (list), "conditions": short text (catalyst, solvent, light, temperature).\n'
           '2. "author_intermediates": the key intermediates of the authors\' proposed mechanism, in order, as SMILES (radicals '
           'with explicit radical centres, e.g. [CH2]C), each with a short name.\n'
           '3. "clues": what the mechanistic experiments OBSERVED that reveals species not in the overall equation - species '
           'detected (HRMS m/z, NMR, EPR), trapped adducts, isolated intermediates, positions of isotope labels. State only '
           'the observation (what was seen, with numbers), never the authors\' interpretation.\n'
           'Use only what the paper states; write null when it is not stated. Answer with one JSON object between '
           '[ANSWER] and [/ANSWER].')


def text_of(pid):
    out = subprocess.run(['pdftotext', '-layout', str(PAPERS / pid / 'main.pdf'), '-'], capture_output=True, text=True)
    return out.stdout[:120000]


def extracted(pid):
    ans = K.cached('lit/extract', pid)
    m = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', ans or '', re.S)
    body = m.group(1) if m else (ans or '')
    lo, hi = body.find('{'), body.rfind('}')
    try:
        return json.loads(body[lo:hi + 1])
    except Exception:
        return None


def case_of(pid):
    e = extracted(pid)
    if not e or not e.get('reactants') or not e.get('products') or not e.get('author_intermediates'):
        return None
    clues = e.get('clues') or []
    clues = [c if isinstance(c, str) else json.dumps(c, ensure_ascii=False) for c in (clues if isinstance(clues, list) else [clues])]
    ints = [x.get('smiles') if isinstance(x, dict) else x for x in e['author_intermediates']]
    return {'id': pid, 'src': 'lit', 'reactants': e['reactants'], 'products': e['products'],
            'conditions': e.get('conditions') or '', 'author': [s for s in ints if isinstance(s, str) and s],
            'clues': '\n'.join(f'- {c}' for c in clues if c)}


def job_extract(pid):
    return 'ok' if K.call('lit/extract', pid, EXTRACT.format(text=text_of(pid))) else 'error'


def job_run(case):
    base = K.official_prompt(case)
    K.call('lit/A0', case['id'], base)
    for i in range(1, 4):
        K.call(f'lit/samp{i}', case['id'], base, temperature=0.7)
    if case['clues']:
        K.call('lit/O_llm', case['id'], base.replace('\n\n## Input', P8.OBS.format(obs=case['clues']), 1))
        pool = [m for i in range(1, 4) if (m := K.parse_mech(K.cached(f'lit/samp{i}', case['id'])))]
        a0 = K.parse_mech(K.cached('lit/A0', case['id']))
        pool = ([a0] if a0 else []) + pool
        if len(pool) >= 2:
            K.call('lit/O_ms', case['id'], P8.MS.format(reactants='.'.join(case['reactants']),
                                                        products='.'.join(case['products']), conditions=case['conditions'],
                                                        obs=case['clues'], cands=_render(pool)))
    return 'ok'


def _render(pool):
    import p3_selector as P3
    return P3.render(pool)


def analyze(pids):
    rows = {}
    for pid in pids:
        c = case_of(pid)
        if not c:
            continue
        akeys = [k for s in c['author'] if (k := K.species_keys(s))]
        for arm in ('A0', 'O_llm', 'O_ms'):
            m = K.parse_mech(K.cached(f'lit/{arm}', pid))
            if m is None:
                continue
            pk = {k for st in m for k in K.species_keys(st.get('intermediate_smiles') or '')}
            rec = sum(bool(k & pk) for k in akeys) / len(akeys) if akeys else 0.0
            rows.setdefault(arm, []).append((pid, {'author_intermediate_recall': rec}))
    rep = {a: K.summarize(r, ['author_intermediate_recall']) for a, r in rows.items()}
    for a in ('O_llm', 'O_ms'):
        if a in rows and 'A0' in rows:
            rep[f'{a}_minus_A0'] = K.paired(rows[a], rows['A0'], 'author_intermediate_recall')
    (K.OUT / 'SUMMARY_P9.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['extract', 'run', 'analyze'], default='extract')
    ap.add_argument('--workers', type=int, default=30)
    cli = ap.parse_args()
    pids = sorted(p.name for p in PAPERS.iterdir() if (p / 'main.pdf').exists())
    if cli.phase == 'extract':
        K.run_pool(job_extract, [(p,) for p in pids], cli.workers)
        ok = [p for p in pids if case_of(p)]
        print(json.dumps({'papers': len(pids), 'usable_cases': len(ok),
                          'with_clues': sum(bool(case_of(p)['clues']) for p in ok)}))
    elif cli.phase == 'run':
        K.run_pool(job_run, [(c,) for p in pids if (c := case_of(p))], cli.workers)
        analyze(pids)
    else:
        analyze(pids)
