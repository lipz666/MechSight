"""Resume the fixed 172-paper Auto v5 set, then real-observation Guided v7.

Literature adaptation of p25: four hypotheses, three votes, Auto-first revision,
checked completion states, one inclusion retry. No simulated HRMS/GT-derived clues.
Without an observed formula, hypotheses are atom/valence checked rather than
filtered against a formula derived from ground truth. Formula bookkeeping is
computed on the proposed structures and supplied to the votes as hypotheses.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading

import common4 as K
import p11_mechagent as M
import p20_lit150 as L
import p24_v6 as V
import p25_guided7 as G
import p8_observations as P8
import stepmap as SMAP

DS = 'lit150'
RUN = K.OUT / 'lit172_v7_real_20261105'
STATE = RUN / 'status.json'

HYP = (
    'Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\n'
    'Reaction conditions: {conditions}\n\nExperimental observations:\n{obs}\n\n'
    'Propose up to four distinct mechanistic intermediate structures supported by these observations '
    'on the way from the reactants to the products. Distinguish a trapped adduct from its precursor, '
    'and an inferred intermediate from a directly detected species. Respect any explicitly reported '
    'formula, isotope location or structure, but do not invent an observed formula or mass. '
    'Explain briefly how each proposal accounts for the observations. Give one valid SMILES per line '
    'between <states> and </states>. If no structure can reasonably be proposed, leave the block empty.'
)
VOTE = (
    'Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\n'
    'Reaction conditions: {conditions}\n\nExperimental observations:\n{obs}\n\n'
    'Mechanism proposed without observations:\n{path}\n\n'
    'Proposed intermediates (RDKit checked), proposal support, calculated formulas and possible '
    'formula bookkeeping decompositions (computed hypotheses, not experimental measurements):\n{isomers}\n\n'
    'For every structure, discuss how it forms from the preceding species, proceeds to the next '
    'species by elementary steps, and explains the actual observations. Distinguish inferred precursors '
    'from trapped adducts. Choose the best supported mechanistic intermediate. If none is justified, '
    'choose 0. End with <choice>k</choice>, k being its number or 0.'
)
REVISE = (
    'Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\n'
    'Reaction conditions: {conditions}\n\nExperimental observations:\n{obs}\n\n'
    'The hypothesis selected by three independent judgements is:\n{chosen}\n'
    'This is an inference to assess against the observations, not an additional experimental fact.\n\n'
    'Candidate mechanisms (candidate 1 is the Auto final path; flagged steps change several bonds):\n'
    '{cands}\n\nChecked intermediates proposed for flagged steps:\n{states}\n\n'
    'Write the final mechanism consistent with the actual observations. Incorporate the selected '
    'intermediate when chemically justified, preserve the other sound intermediates and make every '
    'step elementary. Do not treat a trapped adduct as a native intermediate unless justified. '
    'Return the final mechanism between [ANSWER] and [/ANSWER] as a JSON list of steps, each '
    '{{"step": n, "type": ..., "subtype": ..., "intermediate_smiles": ...}}.'
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    tmp.replace(path)


def status(phase, **extra):
    value = {'pid': os.getpid(), 'phase': phase, 'updated': datetime.now().astimezone().isoformat(), **extra}
    save(STATE, value)
    print(json.dumps(value), flush=True)


def predictive_case(case):
    # Author structures never enter generation, selection or revision.
    return {k: case[k] for k in ('id', 'reactants', 'products', 'conditions', 'clues')}


def obs(case):
    return case['clues']


def job_clue(case, k):
    tag = 'O_llm' if k == 0 else f'G7L_clue{k}'
    prompt = K.official_prompt(case).replace('\n\n## Input', P8.OBS.format(obs=obs(case)), 1)
    return 'ok' if K.call(f'{DS}/{tag}', case['id'], prompt, temperature=0.2 if k == 0 else 0.7) else 'error'


def clue_paths(cid):
    return [m for t in ('O_llm', 'G7L_clue1', 'G7L_clue2')
            if (m := K.parse_mech(K.cached(f'{DS}/{t}', cid)))]


def job_hyp(case, k):
    p = HYP.format(obs=obs(case), **M.fields(case))
    return 'ok' if K.call(f'{DS}/G7L_hyp{k}', case['id'], p, temperature=0.7) else 'error'


def proposals(case):
    support = Counter()
    for k in range(G.NHYP):
        for _, smi in M.checked_states(case, K.cached(f'{DS}/G7L_hyp{k}', case['id']) or ''):
            support[smi] += 1
    for path in clue_paths(case['id']):
        text = '<states>\n' + '\n'.join(dict.fromkeys(s['intermediate_smiles'] for s in path)) + '\n</states>'
        for _, smi in M.checked_states(case, text):
            support[smi] += 1
    return support


def job_vote(case, k):
    iso = proposals(case)
    if not iso:
        return 'no_proposals'
    lines = []
    for i, (s, n) in enumerate(iso.most_common(), 1):
        book = '; '.join(G.bookkeeping(case, s)) or 'no simple decomposition'
        lines.append(f'{i}. {s} [support={n}; calculated formula={V._formula(s)}]\n   {book}')
    auto = K.parse_mech(M.final(DS, case['id'], 2)) or []
    p = VOTE.format(obs=obs(case), path=json.dumps(auto), isomers='\n'.join(lines), **M.fields(case))
    return 'ok' if K.call(f'{DS}/G7L_vote{k}', case['id'], p, temperature=0.7) else 'error'


def choice(case):
    iso = proposals(case)
    ranked = [s for s, _ in iso.most_common()]
    votes = Counter()
    for k in range(G.NVOTE):
        match = re.search(r'<choice>\s*(\d+)\s*</choice>', K.cached(f'{DS}/G7L_vote{k}', case['id']) or '')
        if match and 0 <= int(match[1]) <= len(ranked):
            votes[ranked[int(match[1]) - 1] if int(match[1]) else None] += 1
    if not ranked or not votes:
        return None
    return max(votes, key=lambda s: (votes[s], iso[s] if s else 0))


def guided_final(cid):
    retry = K.cached(f'{DS}/G7L_retry', cid)
    return retry if retry is not None and K.parse_mech(retry) else K.cached(f'{DS}/G7L_rev', cid)


def job_revise(case):
    cid = case['id']
    ch = choice(case)
    auto = K.parse_mech(M.final(DS, cid, 2))
    cands = ([auto] if auto else []) + M.cand_list(DS, cid)[:4] + clue_paths(cid)
    states = [s for tag in ('V5_comp1', 'V5_comp2')
              for _, s in M.checked_states(case, K.cached(f'{DS}/{tag}', cid) or '')]
    p = REVISE.format(obs=obs(case), chosen=ch or 'none justified by the votes',
                      cands=M.render(case, cands), states='\n'.join(dict.fromkeys(states)) or '- none', **M.fields(case))
    ans = K.call(f'{DS}/G7L_rev', cid, p)
    if ans is None:
        return 'error'
    m = K.parse_mech(ans)
    if ch and m and not (K.species_keys(ch) & V.keys_of(m)):
        retry = ('\n\nPrevious answer:\n' + ans + '\n\nThe selected hypothesis ' + ch
                 + ' is absent. Reassess its elementary-step and observation consistency. Include it if justified; '
                 + 'otherwise explicitly explain the evidence-based reason for rejecting it before the final [ANSWER].')
        return 'retried' if K.call(f'{DS}/G7L_retry', cid, p + retry) else 'error'
    return 'ok'


def phase(name, fn, jobs, workers):
    pending = jobs
    for attempt in range(3):
        status(name, jobs=len(jobs), pending=len(pending), attempt=attempt + 1, workers=workers)
        failed, lock = [], threading.Lock()
        def work(*args):
            try:
                result = fn(*args)
            except Exception as exc:
                print(name, type(exc).__name__, flush=True)
                result = 'error'
            if result == 'error':
                with lock:
                    failed.append(args)
            return result
        K.run_pool(work, pending, workers)
        if not failed:
            return
        pending = failed
    raise RuntimeError(f'{name}: {len(pending)} jobs still failed; restart to resume their caches')


def mapping(cases):
    M.requests({DS: cases})
    cache = json.loads(SMAP.CACHE.read_text()) if SMAP.CACHE.exists() else {}
    req = {json.loads(line)['rxn'] for line in SMAP.REQ.read_text().splitlines() if line}
    todo = req - cache.keys()
    status('mapping', requested=len(req), pending=len(todo))
    if todo:
        mapper_python = K.ROOT.parent / 'external/env-map/bin/python'
        subprocess.run([str(mapper_python), str(K.HERE / 'stepmap.py')], check=True)
    M._cache = None


def analyze(cases):
    rows = {a: [] for a in ('Direct', 'MechSight_Auto', 'Direct_clues', 'MechSight_Guided_V7_literature')}
    per = []
    for cid, c in cases.items():
        keys = [K.species_keys(s) for s in c['author'] if K.species_keys(s)]
        answers = {'Direct': K.cached(f'{DS}/base_default', cid), 'MechSight_Auto': M.final(DS, cid, 2)}
        if c['clues']:
            answers.update(Direct_clues=K.cached(f'{DS}/O_llm', cid), MechSight_Guided_V7_literature=guided_final(cid))
        rec = {'id': cid, 'n_author': len(keys), 'has_clues': bool(c['clues'])}
        for arm, answer in answers.items():
            mech = K.parse_mech(answer)
            pred = V.keys_of(mech)
            hits = sum(bool(k & pred) for k in keys)
            scores = {'recall': hits / len(keys), 'all_present': float(hits == len(keys)),
                      'parsed': float(bool(mech)), 'available': float(answer is not None)}
            rows[arm].append((cid, scores))
            rec[arm] = scores
        per.append(rec)
    met = ['recall', 'all_present', 'parsed', 'available']
    paired_ids = {cid for cid, c in cases.items() if c['clues']}
    paired = {a: [(cid, sc) for cid, sc in r if cid in paired_ids] for a, r in rows.items()}
    rep = {'n_papers': len(cases), 'n_with_clues': len(paired_ids),
           'method': 'Guided v7 literature adaptation, real observations only; not frozen simulated-HRMS v7',
           'all_papers': {a: K.summarize(rows[a], met) for a in ('Direct', 'MechSight_Auto')},
           'paired_with_clues': {a: K.summarize(r, met) for a, r in paired.items()}}
    for hi, lo in [('MechSight_Auto', 'Direct'), ('MechSight_Guided_V7_literature', 'Direct'),
                   ('MechSight_Guided_V7_literature', 'Direct_clues'), ('MechSight_Guided_V7_literature', 'MechSight_Auto')]:
        rep['paired_with_clues'][f'{hi}_minus_{lo}'] = {k: K.paired(paired[hi], paired[lo], k) for k in met[:2]}
    save(RUN / 'SUMMARY_LIT172_V7_REAL.json', {'summary': rep, 'per_paper': per})
    print(json.dumps(rep, indent=1), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=300)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['STEPMAP_REQ'] = str(K.OUT / 'stepmap_lit150_requests.jsonl')
    os.environ['STEPMAP_CACHE'] = str(K.OUT / 'stepmap_lit150_cache.json')
    SMAP.REQ = Path(os.environ['STEPMAP_REQ'])
    SMAP.CACHE = Path(os.environ['STEPMAP_CACHE'])
    cases = L.cases()
    assert len(cases) == 172, f'Dataset changed: {len(cases)}'
    inputs = {cid: predictive_case(c) for cid, c in cases.items()}
    manifest = {'ids': sorted(cases), 'inputs_sha256': hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
                'model': 'gemini-3.8-flash', 'endpoint': 'http://LLM-GATEWAY-1.example:8080/v1', 'workers': cli.workers,
                'auto': 'existing frozen v5', 'guided': 'v7 literature adaptation with real observations',
                'hypotheses': G.NHYP, 'votes': G.NVOTE, 'n_with_clues': sum(bool(c['clues']) for c in cases.values()),
                'changes_from_p25': ['No GT-derived simulated HRMS or outcome-conditioned blind-case selection',
                    'No exact-formula filter when no formula is observed; keep RDKit valence/atom/endpoint checks',
                    'Compute formula bookkeeping on proposals before voting; decompositions are hypotheses',
                    'Votes may reject all hypotheses; revision distinguishes inferred species and observed adducts',
                    'One evidence-based inclusion retry; no observation-free paper counted in Guided'],
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    mf = RUN / 'MANIFEST.json'
    if mf.exists():
        old = json.loads(mf.read_text())
        assert old['inputs_sha256'] == manifest['inputs_sha256'], 'Prediction inputs changed'
        assert old['source_sha256'] == manifest['source_sha256'], 'Method changed; use a new run directory'
    else:
        save(mf, manifest)
    if cli.analyze_only:
        analyze(cases)
        return
    assert os.environ.get('FALLBACK_API_KEY'), 'FALLBACK_API_KEY required'
    assert os.environ.get('FAILOVER_CAP_VPS') == '0', 'Only the requested endpoint may be enabled'
    assert os.environ.get('FAILOVER_CAP_117') == str(cli.workers)
    cs = list(inputs.values())
    arms = ['base_default', 'samp1', 'samp2', 'samp3', 'samp4', 'enum3']
    phase('auto_pool', L.job_pool, [(a, c) for a in arms for c in cs], cli.workers)
    mapping(inputs)
    phase('auto_round1', M.job_r1, [(DS, c) for c in cs], cli.workers)
    mapping(inputs)
    phase('auto_round2', M.job_r2, [(DS, c) for c in cs], cli.workers)
    guided = [c for c in cs if c['clues']]
    phase('guided_clue_paths', job_clue, [(c, k) for c in guided for k in range(3)], cli.workers)
    phase('guided_hypotheses', job_hyp, [(c, k) for c in guided for k in range(G.NHYP)], cli.workers)
    phase('guided_votes', job_vote, [(c, k) for c in guided for k in range(G.NVOTE)], cli.workers)
    phase('guided_revision', job_revise, [(c,) for c in guided], cli.workers)
    status('analyzing')
    rep = analyze(cases)
    status('complete', papers=len(cases), guided_papers=len(guided), summary=str(RUN / 'SUMMARY_LIT172_V7_REAL.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Do not overwrite the status of another process when the run lock is held.
        if not isinstance(exc, BlockingIOError):
            status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
