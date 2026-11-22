"""P36: clue classification (all 436 papers) and gold/clue audit (stratified sample), gemini-4-argon-build.

classify  one text-only call per paper: every clue line is labelled experimental / computational / mixed /
          author_conclusion; mixed lines get an experimental-only rewrite. Gives (i) the clue set without
          computational content used by p37_nodft.py and (ii) the share of lines that still carry author conclusions.
audit     one call per sampled paper with the full PDF: each scored author intermediate is checked (correct structure,
          proposed by the authors), missing author intermediates are listed, and each clue line is checked for
          author conclusions. The output is a pre-audit for human confirmation, not a human audit.

python p36_clue_audit.py classify [--workers 100]
python p36_clue_audit.py audit [--n 50] [--workers 50]
python p36_clue_audit.py report
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import threading
import time

from openai import OpenAI

import common4 as K
import p29_gold_rebuild as GOLD
import p30_lit172_gold as P
import p31_guided71 as T

MODEL = 'gemini-4-argon-build'
BASE = 'http://LLM-GATEWAY-1.example:8080/v1'
OUTD = K.OUT / 'supp_clue_audit_20261106'
_SEM = threading.Semaphore(int(os.environ.get('ARGON_CAP', '100')))
SEED = 20261106
_DEV = (P.DS, P.IDS, P.RUN)

CLASSIFY = (
    'Below are observations extracted from a research article on a reaction mechanism, one per numbered line. They '
    'will be given to a model that must propose the mechanism, so they should contain experimental observations only.\n'
    'Label every line with exactly one of:\n'
    '- "experimental": wet-lab observations only (yields, control reactions, spectroscopy, HRMS, isotope labelling, '
    'KIE, radical trapping or clocks, kinetics, electrochemistry, X-ray structures, quenching, quantum yields);\n'
    '- "computational": only results of calculations (DFT, xTB, ab initio, computed energies, barriers, transition '
    'states, BDEs, orbital or spin-density analyses, computed spectra);\n'
    '- "mixed": both kinds in one line; also give "experimental_only", the line rewritten with every computational '
    'statement removed and nothing added;\n'
    '- "author_conclusion": the line states the authors\' mechanistic interpretation (e.g. "indicating a radical '
    'pathway", "suggesting that intermediate X is formed") rather than an observation, whether or not it also '
    'contains observations.\n\n'
    'Lines:\n{lines}\n\n'
    'Return JSON between [ANSWER] and [/ANSWER]: a list of objects {{"line": n, "label": ..., '
    '"experimental_only": ... (only for "mixed")}}, one per line, in order.'
)

AUDIT = (
    'You are auditing a machine-extracted record of the mechanism proposed in the attached article (read the full '
    'PDF, including schemes and figures).\n\n'
    'Model reaction (extracted): {reaction}\n\n'
    'Extracted author intermediates (substrate-derived, used for scoring):\n{ints}\n\n'
    'Extracted experimental observations given to the models, one per numbered line:\n{clues}\n\n'
    'Tasks:\n'
    '1. For each extracted intermediate: is it an intermediate the authors propose for this model reaction '
    '("proposed": true/false), and is the SMILES the correct structure, including charges and radical centres '
    '("structure": "correct" / "minor_error" (e.g. wrong charge or radical position, same skeleton) / "wrong")? '
    'Give a one-sentence reason for anything not correct.\n'
    '2. List substrate-derived intermediates the authors propose for this model reaction that are missing from the '
    'list (name and SMILES).\n'
    '3. For each observation line: does it state an author mechanistic conclusion rather than an observation '
    '("conclusion": true/false)?\n'
    'Return JSON between [ANSWER] and [/ANSWER]: {{"intermediates": [{{"index": i, "proposed": ..., "structure": ..., '
    '"reason": ...}}], "missing": [{{"name": ..., "smiles": ...}}], "clues": [{{"line": n, "conclusion": ...}}]}}'
)


def clue_lines(case):
    return [l.strip()[2:].strip() if l.strip().startswith('- ') else l.strip()
            for l in (case['clues'] or '').split('\n') if l.strip()]


def corpus():
    """[(set, cid, case, record)] for the 436 scored papers."""
    out = []
    for which in ('dev', 'test'):
        if which == 'dev':  # T.configure('dev') does not reset the dataset after a 'test' call
            P.DS, P.IDS, P.RUN = _DEV
        T.configure(which)
        cs = P.cases()
        rec = P.records()
        for c in cs.values():
            if c['clues'] and any(K.species_keys(s) for s in c['author']):
                out.append((which, c['id'], c, rec[c['id']]))
    return out


def call(tag, pid, text, pdf=False):
    dest = OUTD / tag / f'{pid}.json'
    if dest.exists():
        return json.loads(dest.read_text())['answer']
    trace = OUTD / tag / 'traces' / f'{pid}.jsonl'
    trace.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=os.environ['FALLBACK_API_KEY'], base_url=BASE, timeout=900, max_retries=0)
    content = [{'type': 'text', 'text': text}] + ([GOLD.pdf_part(pid)] if pdf else [])
    for attempt in range(6):
        t0 = time.monotonic()
        try:
            with _SEM:
                r = client.chat.completions.create(model=MODEL, temperature=0.2, max_tokens=32768,
                                                   messages=[{'role': 'user', 'content': content}])
            ans = r.choices[0].message.content or ''
            if parse(ans) is None:
                raise RuntimeError('no parsable [ANSWER] block')
            dest.write_text(json.dumps({'id': pid, 'tag': tag, 'model': MODEL, 'returned_model': r.model,
                                        'usage': r.usage.model_dump() if r.usage else None, 'prompt': text,
                                        'answer': ans}, ensure_ascii=False, indent=1))
            return ans
        except Exception as e:  # noqa: BLE001
            with trace.open('a') as f:
                f.write(json.dumps({'attempt': attempt, 'error': type(e).__name__, 'detail': str(e)[:300],
                                    'elapsed': round(time.monotonic() - t0, 1)}) + '\n')
            time.sleep(min(60, 5 * 2 ** attempt))
    return None


def parse(ans):
    m = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', ans or '', re.S)
    if not m:
        return None
    txt = re.sub(r'^```(?:json)?|```$', '', m.group(1).strip(), flags=re.M).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


def job_classify(which, cid, case, rec):
    lines = clue_lines(case)
    text = CLASSIFY.format(lines='\n'.join(f'{i + 1}. {l}' for i, l in enumerate(lines)))
    return 'ok' if call('classify', cid, text) else 'error'


def audit_ints(case, rec):
    """Scored author intermediates with their extracted names."""
    names = {}
    for it in rec['record'].get('intermediates') or []:
        k = K.species_keys(it.get('smiles') or '')
        if k:
            names.setdefault(frozenset(k), it.get('name') or it.get('label') or '')
    out = []
    for s in case['author']:
        k = K.species_keys(s)
        if k:
            out.append((s, names.get(frozenset(k), '')))
    return out


def job_audit(which, cid, case, rec):
    mr = rec['record'].get('model_reaction') or {}
    reaction = (f"{' + '.join(mr.get('reactants') or [])} -> {' + '.join(mr.get('products') or [])}; "
                f"catalysts/reagents: {', '.join(mr.get('catalysts') or [])}; conditions: {mr.get('conditions', '')}")
    ints = '\n'.join(f'{i + 1}. {name} | {s}' for i, (s, name) in enumerate(audit_ints(case, rec)))
    clues = '\n'.join(f'{i + 1}. {l}' for i, l in enumerate(clue_lines(case)))
    return 'ok' if call('audit', cid, AUDIT.format(reaction=reaction, ints=ints, clues=clues), pdf=True) else 'error'


def sample(items, n):
    rng = random.Random(SEED)
    dev = [x for x in items if x[0] == 'dev']
    test = [x for x in items if x[0] == 'test']
    k_dev = round(n * len(dev) / len(items))
    return rng.sample(dev, k_dev) + rng.sample(test, n - k_dev)


def nodft_clues(cid, case):
    """Clue text without computational content (None if the paper has no classification)."""
    lab = parse(json.loads((OUTD / 'classify' / f'{cid}.json').read_text())['answer']) \
        if (OUTD / 'classify' / f'{cid}.json').exists() else None
    if lab is None:
        return None
    lines = clue_lines(case)
    by = {int(x.get('line', 0)): x for x in lab if isinstance(x, dict)}
    keep = []
    for i, l in enumerate(lines, 1):
        x = by.get(i, {'label': 'experimental'})
        if x.get('label') == 'computational':
            continue
        if x.get('label') == 'mixed':
            l = (x.get('experimental_only') or '').strip()
            if not l:
                continue
        keep.append('- ' + l)
    return '\n'.join(keep)


def report():
    items = corpus()
    rows, leak = [], []
    for which, cid, case, rec in items:
        f = OUTD / 'classify' / f'{cid}.json'
        if not f.exists():
            continue
        lab = parse(json.loads(f.read_text())['answer']) or []
        labels = [x.get('label') for x in lab if isinstance(x, dict)]
        rows.append({'set': which, 'id': cid, 'lines': len(clue_lines(case)), 'labelled': len(labels),
                     **{k: labels.count(k) for k in ('experimental', 'computational', 'mixed', 'author_conclusion')}})
    tot = lambda k: sum(r[k] for r in rows)
    summary = {'papers_classified': len(rows), 'lines': tot('lines'),
               **{k: tot(k) for k in ('experimental', 'computational', 'mixed', 'author_conclusion')},
               'papers_with_computational_content': sum(1 for r in rows if r['computational'] + r['mixed'] > 0),
               'papers_with_author_conclusion_lines': sum(1 for r in rows if r['author_conclusion'] > 0)}
    au = []
    for which, cid, case, rec in items:
        f = OUTD / 'audit' / f'{cid}.json'
        if not f.exists():
            continue
        a = parse(json.loads(f.read_text())['answer']) or {}
        ints = a.get('intermediates') or []
        au.append({'set': which, 'id': cid, 'n_extracted': len(audit_ints(case, rec)),
                   'proposed': sum(1 for x in ints if x.get('proposed') is True),
                   'correct': sum(1 for x in ints if x.get('proposed') is True and x.get('structure') == 'correct'),
                   'minor': sum(1 for x in ints if x.get('structure') == 'minor_error'),
                   'wrong': sum(1 for x in ints if x.get('structure') == 'wrong' or x.get('proposed') is False),
                   'missing': len(a.get('missing') or []),
                   'clue_lines': len(a.get('clues') or []),
                   'clue_conclusions': sum(1 for x in a.get('clues') or [] if x.get('conclusion') is True),
                   'detail': a})
    if au:
        ext = sum(r['n_extracted'] for r in au)
        ok = sum(r['correct'] for r in au)
        miss = sum(r['missing'] for r in au)
        summary['audit'] = {'papers': len(au), 'extracted': ext, 'correct_and_proposed': ok,
                            'minor_error': sum(r['minor'] for r in au), 'wrong_or_not_proposed': sum(r['wrong'] for r in au),
                            'missing': miss, 'precision': round(ok / ext, 3) if ext else None,
                            'recall_vs_correct_plus_missing': round(ok / (ok + miss), 3) if ok + miss else None,
                            'clue_lines': sum(r['clue_lines'] for r in au),
                            'clue_lines_with_conclusion': sum(r['clue_conclusions'] for r in au)}
    (OUTD / 'SUMMARY_CLUE_AUDIT.json').write_text(json.dumps({'summary': summary, 'classify': rows, 'audit': au},
                                                              indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['classify', 'audit', 'report'])
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--workers', type=int, default=100)
    cli = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    if cli.mode == 'report':
        return report()
    assert os.environ.get('FALLBACK_API_KEY'), 'FALLBACK_API_KEY required'
    items = corpus()
    if cli.mode == 'classify':
        jobs = items
        fn = job_classify
    else:
        jobs = sample(items, cli.n)
        (OUTD / 'AUDIT_SAMPLE.json').write_text(json.dumps([[w, c] for w, c, _, _ in jobs], indent=1))
        fn = job_audit
    K.run_pool(lambda *a: fn(*a), jobs, cli.workers)
    report()


if __name__ == '__main__':
    main()
