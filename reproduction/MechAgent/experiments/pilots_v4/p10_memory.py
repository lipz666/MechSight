"""P10: a hidden-event memory (ctx engine) - do hidden events learned from other reactions transfer?

Phases:
  cards     one gemini-3.8-flash call per corpus mechanism (oMe-Template + Silver, Silver-dev
            removed; retrieval.corpus): up to 3 transferable hidden-event cards (trigger motif and
            conditions in words, transient species type, example SMILES)
  build     cards -> markdown memories of an isolated ctx instance (CTX_HOME=OUT/ctx_home,
            project 'mechcards'), then `ctx reindex` (EmbeddingGemma + keywords)
  describe  per test reaction, the model describes the reaction centre and conditions in words
            (ctx retrieves text; SMILES queries do not work)
  recall    ctx recall top 12 per description; std keeps the top 4, strict first drops cards whose
            source reaction name matches the model's own identification of the target (P1 identify)
  run       arms (T=0.2), pool = base_default + samp1..4 + the three P2 paths:
              M_gen_std     official prompt + recalled cards (no pool)
              M_ms_std      pool + recalled cards -> keep the applicable ones, eliminate, repair
              M_ms_strict   the same with same-name cards removed
              M_ms_random   the same with 4 random cards (relevance control)
              M_ms_none     pool, no cards, same eliminate/repair instruction (control)
Sets: T3_dev (47), J3 (147). Metric: all GT intermediates present (exact PGL ground truth).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess

import common4 as K
import p3_selector as P3
import retrieval as RT

CTX = os.path.expanduser('~/.ctx/bin/ctx')
CTX_HOME = K.OUT / 'ctx_home'
CARDS = ('Below is a curated stepwise mechanism of a named organic reaction.\n\nName: {name}\nReactants: {reactants}\n'
         'Products: {products}\nConditions: {conditions}\nSteps (subtype: intermediate):\n{steps}\n\n'
         'Identify the hidden events of this mechanism: short-lived intermediates that cannot be read from the overall '
         'equation (bonds that form and later break, ions, ylides, adducts, rearranging species, catalyst- or '
         'reagent-bound species). Ignore simple proton transfers. For each (at most 3, most important first) write a '
         'TRANSFERABLE card that would help a chemist recognise the same hidden event in a different reaction:\n'
         '  "species_type": short class name of the transient species\n'
         '  "trigger": the structural motif and conditions that lead to it, in words, generic (no specific substituents)\n'
         '  "role": what it does next and why it matters for the product\n'
         '  "example_smiles": its SMILES in this reaction\n'
         'Answer only with a JSON list between [ANSWER] and [/ANSWER] (an empty list if there is no hidden event).')
DESCRIBE = ('Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: {conditions}\n\n'
            'In two or three sentences of plain words (no SMILES), describe the reacting functional groups, the reagents '
            'and conditions, and the kind of transformation (bonds made and broken overall). Do not write a mechanism. '
            'Answer between <desc> and </desc>.')
MEMBLOCK = ('Mechanistic knowledge recalled from a memory of hidden events seen in OTHER reactions (each may or may not '
            'apply here):\n{cards}')
MS = ('Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: {conditions}\n\n'
      '{memory}Candidate mechanisms proposed for this reaction (intermediate SMILES after each step):\n\n{cands}\n\n'
      '{instruction} Return the final mechanism between [ANSWER] and [/ANSWER] as a JSON list of steps, each '
      '{{"step": n, "type": ..., "subtype": ..., "intermediate_smiles": ...}}.')
WITH_MEM = ('First decide which recalled hidden events genuinely apply to this reaction. Use the applicable ones to discard '
            'candidates that miss or contradict them, then write the final mechanism: keep a candidate or repair the best '
            'one so that it contains every applicable hidden intermediate (adapted to this substrate).')
NO_MEM = ('First judge each candidate: discard those with unsound elementary steps or missing intermediates. Then write the '
          'final mechanism: keep the most plausible candidate or repair it.')


def sets():
    return {'T3_dev': K.load_fukuyama('T3_dev'), 'J3': K.load_fukuyama('J3')}


# ------------------------------------------------------------------ cards and memory
def job_cards(c):
    steps = '\n'.join(f"  {i}. {s.get('subtype')}: {s.get('intermediate_smiles')}" for i, s in enumerate(c['gold'], 1))
    p = CARDS.format(name=c['name'], reactants='.'.join(c['reactants']), products='.'.join(c['products']),
                     conditions=c['conditions'] or 'not given', steps=steps)
    return 'ok' if K.call('cards', c['id'], p) else 'error'


def all_cards():
    out = []
    for c in RT.corpus():
        ans = K.cached('cards', c['id'])
        m = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', ans or '', re.S)
        body = m.group(1) if m else (ans or '')
        lo, hi = body.find('['), body.rfind(']')
        try:
            cards = json.loads(body[lo:hi + 1]) if lo >= 0 else []
        except Exception:
            cards = []
        for j, x in enumerate(cards if isinstance(cards, list) else []):
            if isinstance(x, dict) and x.get('species_type') and x.get('trigger'):
                out.append({'mid': f'mem_card{len(out):021d}', 'source': c['id'], 'name': c['name'], 'norm': c['norm'],
                            'text': f"Hidden event ({x['species_type']}), seen in {c['name']}. Trigger: {x['trigger']} "
                                    f"Role: {x.get('role', '')} Example intermediate: {x.get('example_smiles', '')}"})
    return out


def build():
    cards = all_cards()
    env = dict(os.environ, CTX_HOME=str(CTX_HOME))
    if not (CTX_HOME / 'config.yaml').exists():
        subprocess.run([CTX, 'init'], env=env, check=True, capture_output=True)
    mem = CTX_HOME / 'memory'
    for c in cards:
        title = f"{c['name']} | {c['text'].split(')')[0].replace('Hidden event (', '')}".replace('\n', ' ')[:120]
        (mem / f"{c['mid']}.md").write_text(
            f"---\nid: {c['mid']}\ntype: fact\ntitle: {json.dumps(title, ensure_ascii=False)}\nscope: mechcards\n"
            f"status: active\nsource: user\nconfidence: 0.85\ncreated_at: 2026-11-03T16:00:00+00:00\n"
            f"updated_at: 2026-11-03T16:00:00+00:00\n---\n{c['text']}\n")
    (K.OUT / 'cards_index.json').write_text(json.dumps(cards, ensure_ascii=False))
    r = subprocess.run([CTX, 'reindex'], env=env, capture_output=True, text=True)
    print(len(cards), 'cards written;', r.stdout.strip()[-200:], r.stderr.strip()[-200:])


def cards_by_id():
    return {c['mid']: c for c in json.loads((K.OUT / 'cards_index.json').read_text())}


# ------------------------------------------------------------------ describe / recall
def job_describe(ds, case):
    p = DESCRIBE.format(reactants='.'.join(case['reactants']), products='.'.join(case['products']),
                        conditions=case['conditions'])
    return 'ok' if K.call(f'{ds}/describe', case['id'], p) else 'error'


def recall(ds, case):
    dest = K.OUT / ds / 'recall' / f"{case['id']}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    desc = (K.tagged(K.cached(f'{ds}/describe', case['id']), 'desc') or [''])
    q = ' '.join(desc).strip()
    if not q:
        return None
    env = dict(os.environ, CTX_HOME=str(CTX_HOME))
    r = subprocess.run([CTX, 'recall', q, '--project', 'mechcards', '--limit', '12', '--json'], env=env,
                       capture_output=True, text=True)
    try:
        hits = [h['id'] for h in json.loads(r.stdout) if h['id'].startswith('mem_card')]
    except Exception:
        return None
    if not (K.OUT / 'cards_index.json').exists():
        return None  # memory not built yet: do not cache an empty recall
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({'query': q, 'hits': hits}))
    return {'query': q, 'hits': hits}


def chosen(ds, case, mode, cards):
    rec = recall(ds, case)
    if not rec:
        return []
    if mode == 'random':
        ids = sorted(cards)
        return [cards[i] for i in random.Random(case['id']).sample(ids, 4)]
    hits = [cards[h] for h in rec['hits'] if h in cards]
    if mode == 'strict':
        names = RT.identify(case, ds)
        hits = [c for c in hits if not any(RT._name_match(c['norm'], n) for n in names)]
    return hits[:4]


# ------------------------------------------------------------------ arms
def pool(ds, cid):
    p = P3.pool(ds, cid)
    return {k: v for k, v in p.items() if k not in ('A2_std', 'A3_std')}


def job_run(ds, case, cards):
    cid = case['id']
    fields = dict(reactants='.'.join(case['reactants']), products='.'.join(case['products']), conditions=case['conditions'])
    std = chosen(ds, case, 'std', cards)
    if std:
        block = MEMBLOCK.format(cards='\n'.join(f'- {c["text"]}' for c in std))
        K.call(f'{ds}/M_gen_std', cid, K.official_prompt(case).replace('\n\n## Input', '\n\n## ' + block + '\n\n## Input', 1))
    p = pool(ds, cid)
    if len(p) < 2:
        return 'skip'
    names = P3.order(cid, p)
    cands = P3.render([p[n] for n in names])
    for mode in ('std', 'strict', 'random'):
        sel = std if mode == 'std' else chosen(ds, case, mode, cards)
        mem = MEMBLOCK.format(cards='\n'.join(f'- {c["text"]}' for c in sel)) + '\n\n' if sel else ''
        K.call(f'{ds}/M_ms_{mode}', cid, MS.format(memory=mem, cands=cands, instruction=WITH_MEM if sel else NO_MEM, **fields))
    K.call(f'{ds}/M_ms_none', cid, MS.format(memory='', cands=cands, instruction=NO_MEM, **fields))
    return 'ok'


def analyze(all_sets):
    rep = {}
    arms = ['base_default', 'judge_pick', 'A3_std', 'M_gen_std', 'M_ms_none', 'M_ms_random', 'M_ms_std', 'M_ms_strict']
    for ds, cases in all_sets.items():
        rows = {a: [] for a in arms}
        for cid, case in cases.items():
            if K.cached(f'{ds}/M_ms_none', cid) is None or K.cached(f'{ds}/base_default', cid) is None:
                continue
            for a in arms:
                if a == 'judge_pick':
                    p = P3.pool(ds, cid)
                    ans = K.cached(f'{ds}/judge', cid)
                    m = re.search(r'<best>\s*(\d+)', ans or '')
                    nm = P3.order(cid, p)
                    pick = nm[int(m.group(1)) - 1] if m and 0 < int(m.group(1)) <= len(nm) else 'base_default'
                    rows[a].append((cid, K.score(case, p.get(pick))))
                    continue
                ans = K.cached(f'{ds}/{a}', cid)
                rows[a].append((cid, K.score(case, K.parse_mech(ans if ans is not None else K.cached(f'{ds}/base_default', cid)))))
        block = {a: K.summarize(r, ['all_present', 'in_order_frac']) for a, r in rows.items() if r}
        for a in arms[1:]:
            if rows[a]:
                block[f'{a}_minus_A0'] = K.paired(rows[a], rows['base_default'], 'all_present')
        for a in ('M_ms_std', 'M_ms_strict', 'M_ms_random'):
            block[f'{a}_minus_M_ms_none'] = K.paired(rows[a], rows['M_ms_none'], 'all_present')
        rep[ds] = block
    (K.OUT / 'SUMMARY_P10.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['cards', 'build', 'describe', 'run', 'analyze'], required=True)
    ap.add_argument('--workers', type=int, default=200)
    cli = ap.parse_args()
    s = sets()
    if cli.phase == 'cards':
        K.run_pool(job_cards, [(c,) for c in RT.corpus()], cli.workers)
        print(len(all_cards()), 'cards')
    elif cli.phase == 'build':
        build()
    elif cli.phase == 'describe':
        K.run_pool(job_describe, [(ds, c) for ds, cs in s.items() for c in cs.values()], cli.workers)
        n = sum(bool(recall(ds, c)) for ds, cs in s.items() for c in cs.values())
        print(n, 'recalled')
    elif cli.phase == 'run':
        cards = cards_by_id()
        K.run_pool(job_run, [(ds, c, cards) for ds, cs in s.items() for c in cs.values()], cli.workers)
        analyze(s)
    else:
        analyze(s)
