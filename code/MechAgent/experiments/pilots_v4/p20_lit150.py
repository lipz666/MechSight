"""P20: expanded 2026 literature set (~150 papers, experiments/lit150).

Same design as P9/P16 on the larger set:
  extract   gemini-3.5-flash-lite on the vps gateway, the unchanged P9 EXTRACT prompt on pdftotext
            of the main article: model reaction (SMILES), authors' key intermediates, observed clues
  pool      base_default (official prompt), samp1..4 (T=0.7), enum3 (P2)        [gemini-3.8-flash]
  v5        Mech-Agent v5 rounds 1-2 (frozen p11 code; own stepmap request/cache files)
  clues     O_llm (official prompt + clues), V5_obs (P16 revision with the clues)
Metric: recall of the authors' key intermediates and all of them present (sigma-graph keys).
Data set name 'lit150'; outputs under OUT/lit150/<tag>/<PMCID>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import common4 as K
import p2_enumerate as P2
import p6_baselines as P6
import p8_observations as P8
import p9_literature as P9

DS = 'lit150'
PAPERS = K.HERE.parent / 'lit150' / 'papers'
LITE = {'base_url': 'https://LLM-GATEWAY-2.example/v1', 'model': 'gemini-3.5-flash-lite', 'key_env': 'OPENAI_API_KEY'}


def pids():
    sel = json.loads((K.HERE.parent / 'lit150' / 'selected.json').read_text())
    return [r['pmcid'] for r in sel if (PAPERS / r['pmcid'] / 'main.pdf').exists()]


def text_of(pid):
    out = subprocess.run(['pdftotext', '-layout', str(PAPERS / pid / 'main.pdf'), '-'], capture_output=True, text=True)
    return out.stdout[:120000]


def lite_call(tag, cid, prompt):
    dest = K.OUT / DS / tag / f'{cid}.json'
    if dest.exists():
        return json.loads(dest.read_text())['answer']
    from openai import OpenAI
    client = OpenAI(api_key=os.environ[LITE['key_env']], base_url=LITE['base_url'], timeout=600, max_retries=0)
    for attempt in range(6):
        try:
            r = client.chat.completions.create(model=LITE['model'], temperature=0.0,
                                               messages=[{'role': 'user', 'content': prompt}])
            ans = r.choices[0].message.content or ''
            if ans.strip():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps({'id': cid, 'tag': tag, 'model': LITE['model'], 'endpoint': LITE['base_url'],
                                            'returned_model': r.model, 'answer': ans}, ensure_ascii=False, indent=1))
                return ans
        except Exception:
            pass
        time.sleep(3 * 2 ** attempt)
    return None


def _parse(ans):
    """Tolerant JSON parse of a flash-lite extraction: one object, several objects (merged), trailing commas."""
    m = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', ans or '', re.S)
    body = re.sub(r'```(?:json)?', '', m.group(1) if m else (ans or ''))
    lo, hi = body.find('{'), body.rfind('}')
    if lo < 0 or hi <= lo:
        return None
    body = re.sub(r',\s*([}\]])', r'\1', body[lo:hi + 1])
    for text in (body, '[' + body + ']'):
        try:
            out = json.loads(text)
        except Exception:
            continue
        if isinstance(out, list):
            merged = {}
            for d in out:
                if isinstance(d, dict):
                    merged.update(d)
            out = merged
        return out if isinstance(out, dict) else None
    return None


def extracted(pid):
    for tag in ('extract', 'extract_retry'):
        e = _parse(K.cached(f'{DS}/{tag}', pid))
        if e and e.get('reactants') and e.get('products') and e.get('author_intermediates') and _valid(e):
            return e
    return _parse(K.cached(f'{DS}/extract', pid))


def _fix(smi):
    return smi.replace('\u2022', '').replace('•', '') if isinstance(smi, str) else smi


def _valid(e):
    from rdkit import Chem
    ints = [x.get('smiles') if isinstance(x, dict) else x for x in e.get('author_intermediates') or []]
    return any(isinstance(s, str) and Chem.MolFromSmiles(_fix(s)) is not None for s in ints)


def case_of(pid):
    """P9.case_of on the lit150 extraction, plus a check that every SMILES parses."""
    from rdkit import Chem
    e = extracted(pid)
    if not e or not e.get('reactants') or not e.get('products') or not e.get('author_intermediates'):
        return None
    clues = e.get('clues') or []
    clues = [c if isinstance(c, str) else json.dumps(c, ensure_ascii=False) for c in (clues if isinstance(clues, list) else [clues])]
    ints = [_fix(x.get('smiles') if isinstance(x, dict) else x) for x in e['author_intermediates']]
    lst = lambda v: v if isinstance(v, list) else [v]  # noqa: E731
    reac = [_fix(s) for s in lst(e['reactants']) if isinstance(s, str) and Chem.MolFromSmiles(_fix(s)) is not None]
    prod = [_fix(s) for s in lst(e['products']) if isinstance(s, str) and Chem.MolFromSmiles(_fix(s)) is not None]
    author = [s for s in ints if isinstance(s, str) and s and K.species_keys(s)]
    if not reac or not prod or not author:
        return None
    return {'id': pid, 'src': 'lit', 'reactants': reac, 'products': prod, 'conditions': e.get('conditions') or '',
            'author': author, 'clues': '\n'.join(f'- {c}' for c in clues if c and c != 'null')}


def cases():
    return {p: c for p in pids() if (c := case_of(p))}


RETRY_NOTE = ('\n\nFormat reminder: answer with exactly ONE JSON object containing all keys; every structure must be '
              'a valid RDKit SMILES (radical centres as bracket atoms such as [CH2] or [O], never a dot symbol); list '
              'the key intermediates of the proposed mechanism even if the paper draws them only in a scheme.')


def job_extract(pid):
    return 'ok' if lite_call('extract', pid, P9.EXTRACT.format(text=text_of(pid))) else 'error'


def job_extract_retry(pid):
    """Second flash-lite pass (same prompt + format reminder) only for papers the first pass left unusable."""
    if case_of(pid):
        return 'usable'
    return 'ok' if lite_call('extract_retry', pid, P9.EXTRACT.format(text=text_of(pid)) + RETRY_NOTE) else 'error'


def job_pool(arm, case):
    if arm == 'enum3':
        return P2.job(DS, case)
    return P6.job(DS, arm, case)


def job_clues(case):
    if not case['clues']:
        return 'noclues'
    import p11_mechagent as M
    import p16_supplement_g38 as P16
    cid = case['id']
    K.call(f'{DS}/O_llm', cid, K.official_prompt(case).replace('\n\n## Input', P8.OBS.format(obs=case['clues']), 1))
    cands = M.cand_list(DS, cid)
    if len(cands) < 2:
        return 'skip'
    comp = K.cached(f'{DS}/V5_comp1', cid)
    st = M.checked_states(case, comp) if comp else []
    states = '\n'.join(f'- ({loc}) {s}' for loc, s in st) or '- none'
    p = P16.V5_OBS_REVISE.format(cands=M.render(case, cands), states=states, obs=case['clues'], **M.fields(case))
    return 'ok' if K.call(f'{DS}/V5_obs', cid, p) else 'error'


def analyze():
    import p11_mechagent as M
    cs = cases()
    arms = ('Direct', 'Direct_clues', 'MechSight_Auto', 'MechSight_Guided')
    rows = {a: [] for a in arms}
    per = []
    for pid, c in sorted(cs.items()):
        akeys = [k for s in c['author'] if (k := K.species_keys(s))]
        ans = {'Direct': K.cached(f'{DS}/base_default', pid), 'MechSight_Auto': M.final(DS, pid, 2)}
        if c['clues']:
            ans['Direct_clues'] = K.cached(f'{DS}/O_llm', pid)
            ans['MechSight_Guided'] = K.cached(f'{DS}/V5_obs', pid)
        row = {'id': pid, 'n_author': len(akeys), 'has_clues': bool(c['clues'])}
        for a, t in ans.items():
            m = K.parse_mech(t)
            if m is None:
                continue
            pk = {k for st in m for k in K.species_keys(st.get('intermediate_smiles') or '')}
            hits = sum(bool(k & pk) for k in akeys)
            s = {'recall': hits / len(akeys), 'all_present': float(hits == len(akeys))}
            rows[a].append((pid, s))
            row[a] = s
        per.append(row)
    clue_ids = {r['id'] for r in per if r['has_clues'] and all(a in r for a in arms)}
    rep = {'n_papers': len(per), 'n_paired_with_clues': len(clue_ids)}
    rep['all_papers'] = {a: K.summarize(rows[a], ['recall', 'all_present']) for a in ('Direct', 'MechSight_Auto')}
    rep['all_papers']['Auto_minus_Direct'] = {k: K.paired(rows['MechSight_Auto'], rows['Direct'], k) for k in ('recall', 'all_present')}
    paired = {a: [(c, s) for c, s in rows[a] if c in clue_ids] for a in arms}
    rep['paired_with_clues'] = {a: K.summarize(r, ['recall', 'all_present']) for a, r in paired.items()}
    for hi, lo in (('MechSight_Auto', 'Direct'), ('Direct_clues', 'Direct'), ('MechSight_Guided', 'Direct'),
                   ('MechSight_Guided', 'Direct_clues'), ('MechSight_Guided', 'MechSight_Auto')):
        rep['paired_with_clues'][f'{hi}_minus_{lo}'] = {k: K.paired(paired[hi], paired[lo], k) for k in ('recall', 'all_present')}
    (K.OUT / 'SUMMARY_P20_LIT150.json').write_text(json.dumps({'summary': rep, 'per_paper': per}, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['extract', 'pool', 'r1', 'r2', 'clues', 'analyze'], required=True)
    ap.add_argument('--workers', type=int, default=100)
    cli = ap.parse_args()
    if cli.phase == 'extract':
        ids = pids()
        K.run_pool(job_extract, [(p,) for p in ids], cli.workers)
        K.run_pool(job_extract_retry, [(p,) for p in ids], cli.workers)
        cs = cases()
        print(json.dumps({'papers': len(ids), 'usable': len(cs), 'with_clues': sum(bool(c['clues']) for c in cs.values())}))
    elif cli.phase == 'pool':
        arms = ['base_default'] + [f'samp{i}' for i in range(1, 5)] + ['enum3']
        K.run_pool(job_pool, [(a, c) for a in arms for c in cases().values()], cli.workers)
    elif cli.phase in ('r1', 'r2'):
        import p11_mechagent as M
        fn = M.job_r1 if cli.phase == 'r1' else M.job_r2
        K.run_pool(fn, [(DS, c) for c in cases().values()], cli.workers)
    elif cli.phase == 'clues':
        K.run_pool(job_clues, [(c,) for c in cases().values()], cli.workers)
    else:
        analyze()
