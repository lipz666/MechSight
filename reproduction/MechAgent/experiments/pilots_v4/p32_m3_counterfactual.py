"""P32 / M3: block the literature mechanism's key step k* and see whether the model still writes it.

Pre-registered in manuscript/机理能力来源_预注册_v1.md (M3). Phases:
  design   gemini-4-argon-build (text only) reads the gold record of each in-scope old paper and, when the mechanism
           has a key step k* that hinges on one site of one substrate, proposes a PRESERVING analogue (edit >= 3 bonds
           away from the site) and a BREAKING analogue (site loses the H it needs, or the group is deleted), with
           atom-map numbers carried over, a parent-only ghost SMARTS and a parent-blind judge question. One repair round.
  check    RDKit: maps, H counts, edit distance, MCS >= 80 % of parent heavy atoms, ghost SMARTS matches parent and
           not the breaking analogue, both analogues differ from the parent; PubChem InChIKey lookup (novelty).
  predict  gemini-3.8-flash, official prompt WITHOUT products (predict product + mechanism), parent / preserve /
           break, 2 samples at T = 0.7.
  judge    gemini-4-argon-build, blind (sees one molecule, the model mechanism, the generic k* question), 2 passes.
  analyze  prefilter: parent uses k* in >= 1 of 2 samples (judge pass a). Primary: k* usage break - preserve,
           parent-paired bootstrap. Ghost structures by program only.

python p32_m3_counterfactual.py --phase all --workers 100
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter

import numpy as np
from openai import OpenAI
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

import common4 as K
import p29_gold_rebuild as P29

RDLogger.DisableLog('rdApp.*')
LIT_OLD = K.HERE.parent / 'lit_old'
SCOPE = json.loads((LIT_OLD / 'gold_scope_old.json').read_text())
GOLD = K.OUT / 'lit_old_argon' / 'gold2'
RUN = K.OUT / 'm3_counterfactual_20261106'
ARGON = 'gemini-4-argon-build'
_SEM = threading.Semaphore(int(os.environ.get('ARGON_CAP', '100')))
VARIANTS = ('parent', 'preserve', 'break')
NSAMP, TEMP, SEED = 2, 0.7, 20261106
KTYPES = ['beta_H_elimination', '1,2_shift', 'enolization_alpha_deprotonation', 'cyclopropyl_ring_opening',
          'directed_CH_activation', 'intramolecular_trap_cyclization', 'other_site_specific']

DESIGN = '''You are designing a counterfactual test of whether a model reasons about a reaction mechanism or recites it.

Literature reaction (substrate atoms carry atom-map numbers):
reactants: {reactants}
catalysts/reagents: {catalysts}
products: {products}
conditions: {conditions}
Literature mechanism, intermediates in order (label: SMILES):
{intermediates}

Task.
1. Decide whether this mechanism contains ONE key step k* that is only possible because of a specific site on ONE
   reactant: e.g. beta-H elimination (needs a beta-H), a 1,2-shift (needs the migrating group), enolization /
   alpha-deprotonation (needs an alpha-H), cyclopropyl(carbinyl) radical ring opening (needs the cyclopropane),
   directing-group-assisted C-H activation/cyclometalation (needs the directing group or the ortho C-H), intramolecular
   trapping/cyclization (needs the tethered group). If none is clear-cut, answer {{"kstar": null}}.
2. Otherwise design two NEW analogues of that reactant (write the full reactant SMILES; keep the atom-map number on
   every atom that is unchanged; new atoms get no map number):
   - "preserve": ONE small edit (e.g. add/remove a methyl, swap Cl/F/OMe, change a para substituent) on an atom at
     least 3 bonds away from every site atom. k* must stay fully possible.
   - "break": the smallest edit that makes k* structurally impossible: either every H that k* needs on the site atoms
     is replaced (e.g. by methyl or F), or the group k* needs is deleted. Keep everything else.
3. "site_atoms": map numbers of the site atoms. "requirement": "H" (k* needs H on site_atoms; the break analogue
   removes all H from them) or "group" (k* needs the atoms in site_atoms; the break analogue deletes them).
4. "ghost_smarts": a SMARTS for a substructure of the PARENT reactant at the site that is absent from the break
   analogue (e.g. the C-H bearing carbon pattern or the deleted group), so that an intermediate showing it on the
   break analogue would be a remembered structure.
5. "judge_question": ONE yes/no question that asks whether a mechanism contains k* at that kind of site, phrased so
   it applies equally to the parent and both analogues and does not reveal the literature mechanism or which
   analogue it is (e.g. "Does the mechanism contain a beta-hydride elimination that forms a C=C bond in the chain
   derived from the alkyl halide?").
Answer with ONE JSON object between [ANSWER] and [/ANSWER]:
{{"kstar": {{"type": one of {ktypes}, "step_label": "...", "description": "...", "reactant_index": int,
  "site_atoms": [int], "requirement": "H"|"group", "preserve": "SMILES", "preserve_edit": "...", "break": "SMILES",
  "break_edit": "...", "ghost_smarts": "...", "judge_question": "..."}}}}'''

REPAIR = '''Your previous design failed these automatic checks:
{errors}
Previous answer:
{prev}
Fix the design (same rules) and answer again with ONE JSON object between [ANSWER] and [/ANSWER].'''

JUDGE = '''Below is a proposed mechanism for a reaction. Answer one question about it.

reactants and reagents: {reactants}
conditions: {conditions}
mechanism (steps in order):
{steps}

Question: {question}
Judge only what the steps actually show (read the SMILES). Answer with ONE JSON object between [ANSWER] and [/ANSWER]:
{{"answer": "yes" or "no", "step": the step number that shows it or null, "reason": "one sentence"}}'''


def argon(tag, key, text):
    dest = RUN / tag / f'{key}.json'
    if dest.exists():
        return json.loads(dest.read_text())['answer']
    dest.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=os.environ['FALLBACK_API_KEY'], base_url=P29.BASE, timeout=900, max_retries=0)
    for attempt in range(6):
        try:
            with _SEM:
                r = client.chat.completions.create(model=ARGON, temperature=0.2, max_tokens=16384,
                                                   messages=[{'role': 'user', 'content': text}])
            ans = r.choices[0].message.content or ''
            if '[ANSWER]' not in ans:
                raise RuntimeError('no [ANSWER] block')
            dest.write_text(json.dumps({'id': key, 'tag': tag, 'model': ARGON, 'returned_model': r.model,
                                        'prompt': text, 'answer': ans}, ensure_ascii=False, indent=1))
            return ans
        except Exception:  # noqa: BLE001
            time.sleep(min(60, 5 * 2 ** attempt))
    return None


# ---------------------------------------------------------------- inputs

def gold(pid):
    return json.loads((GOLD / f'{pid}.json').read_text())


def mapped_reactants(pid):
    mr = gold(pid)['record']['model_reaction']
    out, n = [], 0
    for s in P29._lst(mr.get('reactants')):
        m = P29._mol(s)
        if m is None:
            continue
        for a in m.GetAtoms():
            n += 1
            a.SetAtomMapNum(n)
        out.append(Chem.MolToSmiles(m))
    return out


def design_prompt(pid):
    g = gold(pid)
    mr = g['record']['model_reaction']
    ints = [f"{i.get('label')}: {i.get('smiles')}" for i in g['record'].get('intermediates', [])
            if i.get('role') in ('substrate_intermediate', 'trapped_adduct')]
    return DESIGN.format(reactants=json.dumps(mapped_reactants(pid)), catalysts=json.dumps(P29._lst(mr.get('catalysts'))),
                         products=json.dumps(P29._lst(mr.get('products'))), conditions=mr.get('conditions') or '',
                         intermediates='\n'.join(ints), ktypes=json.dumps(KTYPES))


# ---------------------------------------------------------------- checks

def _strip(m):
    m = Chem.Mol(m)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return m


def _sig(m):
    """Per map number: element, charge, total H, aromatic, sorted mapped neighbours, count of unmapped neighbours."""
    out = {}
    for a in m.GetAtoms():
        if a.GetAtomMapNum():
            nb = sorted(x.GetAtomMapNum() for x in a.GetNeighbors() if x.GetAtomMapNum())
            un = sum(1 for x in a.GetNeighbors() if not x.GetAtomMapNum())
            out[a.GetAtomMapNum()] = (a.GetSymbol(), a.GetFormalCharge(), a.GetTotalNumHs(), a.GetIsAromatic(),
                                      tuple(nb), un)
    return out


def _dist_from(parent, maps_a, maps_b):
    idx = {a.GetAtomMapNum(): a.GetIdx() for a in parent.GetAtoms()}
    dm = Chem.GetDistanceMatrix(parent)
    ds = [dm[idx[x]][idx[y]] for x in maps_a for y in maps_b if x in idx and y in idx]
    return min(ds) if ds else 99


def _mcs_frac(a, b):
    r = rdFMCS.FindMCS([_strip(a), _strip(b)], timeout=10, ringMatchesRingOnly=True,
                       atomCompare=rdFMCS.AtomCompare.CompareElements, bondCompare=rdFMCS.BondCompare.CompareOrder)
    return r.numAtoms / a.GetNumHeavyAtoms()


def check_design(pid, d):
    errs = []
    k = (d or {}).get('kstar')
    if not k:
        return ['no kstar']
    reacts = mapped_reactants(pid)
    i = k.get('reactant_index')
    if not isinstance(i, int) or not 0 <= i < len(reacts):
        return ['reactant_index out of range']
    par = Chem.MolFromSmiles(reacts[i])
    site = [x for x in (k.get('site_atoms') or []) if isinstance(x, int)]
    pmaps = {a.GetAtomMapNum() for a in par.GetAtoms()}
    if not site or not set(site) <= pmaps:
        return ['site_atoms are not map numbers of the chosen reactant']
    psig = _sig(par)
    for v in ('preserve', 'break'):
        m = Chem.MolFromSmiles(k.get(v) or '')
        if m is None:
            errs.append(f'{v} SMILES does not parse')
            continue
        if Chem.MolToSmiles(_strip(m)) == Chem.MolToSmiles(_strip(par)):
            errs.append(f'{v} is identical to the parent')
            continue
        asig = _sig(m)
        changed = {x for x in pmaps if asig.get(x) != psig[x]}
        if not changed:
            errs.append(f'{v}: no mapped atom changed; keep map numbers on unchanged atoms only')
        if _mcs_frac(par, m) < 0.8:
            errs.append(f'{v}: shares < 80 % of the parent heavy atoms')
        if v == 'preserve':
            if any(x not in asig or asig[x][2] != psig[x][2] for x in site):
                errs.append('preserve: a site atom is missing or its H count changed')
            if _dist_from(par, changed, site) < 3:
                errs.append('preserve: the edit is closer than 3 bonds to a site atom')
        else:
            if k.get('requirement') == 'H':
                if any(psig[x][2] == 0 for x in site):
                    errs.append('break: a site atom has no H in the parent')
                if any(x in asig and asig[x][2] > 0 for x in site):
                    errs.append('break: a site atom still carries H')
            elif k.get('requirement') == 'group':
                if any(x in asig for x in site):
                    errs.append('break: site (group) atoms still present')
            else:
                errs.append('requirement must be "H" or "group"')
            if _dist_from(par, changed, site) > 2:
                errs.append('break: an atom more than 2 bonds from the site was changed')
    q = Chem.MolFromSmarts(k.get('ghost_smarts') or '')
    b = Chem.MolFromSmiles(k.get('break') or '')
    if q is None:
        errs.append('ghost_smarts does not parse')
    elif b is not None:
        if not _strip(par).HasSubstructMatch(q):
            errs.append('ghost_smarts does not match the parent reactant')
        if _strip(b).HasSubstructMatch(q):
            errs.append('ghost_smarts also matches the break analogue')
    if not (k.get('judge_question') or '').strip():
        errs.append('judge_question missing')
    return errs


def job_design(pid):
    a = argon('design', pid, design_prompt(pid))
    d = P29.parse(a)
    errs = check_design(pid, d)
    if errs and errs != ['no kstar']:
        a2 = argon('design_repair', pid, design_prompt(pid) + '\n\n' + REPAIR.format(errors='\n'.join(errs), prev=a))
        d2 = P29.parse(a2)
        e2 = check_design(pid, d2)
        if len(e2) < len(errs):
            d, errs = d2, e2
    out = RUN / 'designs' / f'{pid}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'id': pid, 'design': d, 'errors': errs}, ensure_ascii=False, indent=1))
    return 'no kstar' if errs == ['no kstar'] else 'ok' if not errs else 'failed'


def pubchem_novel(smiles):
    key = Chem.MolToInchiKey(_strip(Chem.MolFromSmiles(smiles)))
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{key}/cids/JSON'
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'mech-origin-study/1.0'})
            urllib.request.urlopen(req, timeout=30).read()
            return False
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return True
            time.sleep(2 + 3 * attempt)
        except Exception:  # noqa: BLE001
            time.sleep(2 + 3 * attempt)
    return None


def accepted():
    out = {}
    for f in sorted((RUN / 'designs').glob('*.json')):
        r = json.loads(f.read_text())
        if not r['errors']:
            out[r['id']] = r['design']['kstar']
    return out


def novelty():
    path = RUN / 'pubchem.json'
    nov = json.loads(path.read_text()) if path.exists() else {}
    for pid, k in accepted().items():
        for v in ('preserve', 'break'):
            key = f'{pid}__{v}'
            if nov.get(key) is None:
                nov[key] = pubchem_novel(k[v])
                time.sleep(0.25)  # PubChem asks for <= 5 requests per second
    path.write_text(json.dumps(nov, indent=1))
    return nov


# ---------------------------------------------------------------- prediction and judging

def variant_inputs(pid, k, v):
    mr = gold(pid)['record']['model_reaction']
    reacts = [Chem.MolToSmiles(_strip(Chem.MolFromSmiles(s))) for s in mapped_reactants(pid)]
    if v != 'parent':
        reacts[k['reactant_index']] = Chem.MolToSmiles(_strip(Chem.MolFromSmiles(k[v])))
    cats = [s for s in P29._lst(mr.get('catalysts')) if P29._mol(s) is not None]
    return reacts + cats, mr.get('conditions') or ''


def predict_prompt(inputs, conditions):
    t = K.PROMPTS['default']
    t = t.replace('- Products (SMILES): { products_smiles }\n',
                  '- Products: NOT GIVEN. First decide the major product(s) under these conditions; the last step of '
                  'your mechanism must give the product.\n')
    t = t.replace('  "products_smiles": { products_smiles },\n', '')
    for token, value in {'{ reactants_smiles }': inputs, '{ conditions }': conditions}.items():
        t = t.replace(token, json.dumps(value))
    return t


def job_predict(pid, k, v, s):
    inputs, cond = variant_inputs(pid, k, v)
    ans = K.call(f'm3/pred_s{s}', f'{pid}__{v}', predict_prompt(inputs, cond), temperature=TEMP)
    return 'ok' if ans else 'error'


def job_judge(pid, k, v, s, p):
    mech = K.parse_mech(K.cached(f'm3/pred_s{s}', f'{pid}__{v}'))
    if not mech:
        return 'unparsed'
    inputs, cond = variant_inputs(pid, k, v)
    steps = '\n'.join(f"{i + 1}. {st.get('type')}/{st.get('subtype')}: {st.get('intermediate_smiles')}"
                      for i, st in enumerate(mech))
    a = argon(f'judge_{p}', f'{pid}__{v}__s{s}', JUDGE.format(reactants=json.dumps(inputs), conditions=cond,
                                                               steps=steps, question=k['judge_question']))
    return 'ok' if a else 'error'


def verdict(pid, v, s, p='a'):
    f = RUN / f'judge_{p}' / f'{pid}__{v}__s{s}.json'
    if not f.exists():
        return None
    d = P29.parse(json.loads(f.read_text())['answer']) or {}
    return {'yes': 1.0, 'no': 0.0}.get(str(d.get('answer')).strip().lower())


def ghost(pid, k, v, s):
    """1 if any intermediate is valence-invalid, has an element absent from all inputs, or shows the ghost SMARTS."""
    mech = K.parse_mech(K.cached(f'm3/pred_s{s}', f'{pid}__{v}'))
    if not mech:
        return None
    inputs, _ = variant_inputs(pid, k, v)
    elems = {a.GetSymbol() for s_ in inputs for a in Chem.MolFromSmiles(s_).GetAtoms()} | {'H'}
    q = Chem.MolFromSmarts(k['ghost_smarts'])
    flags = Counter()
    for st in mech:
        smi = st.get('intermediate_smiles') or ''
        m = Chem.MolFromSmiles(smi)
        if m is None:
            raw = Chem.MolFromSmiles(smi, sanitize=False)
            if raw is not None:
                try:
                    Chem.SanitizeMol(raw)
                except Chem.AtomValenceException:
                    flags['valence'] += 1
                except Exception:  # noqa: BLE001  (kekulization etc.: unparsable, not a valence ghost)
                    pass
            continue
        if {a.GetSymbol() for a in m.GetAtoms()} - elems:
            flags['element'] += 1
        if v == 'break' and m.HasSubstructMatch(q):
            flags['smarts'] += 1
    return flags


# ---------------------------------------------------------------- driver

def phase(name, fn, jobs, workers):
    print(name, len(jobs), flush=True)
    for _ in range(3):
        failed = []
        lock = threading.Lock()

        def work(*a):
            try:
                r = fn(*a)
            except Exception as e:  # noqa: BLE001
                print(name, type(e).__name__, str(e)[:120], flush=True)
                r = 'error'
            if r == 'error':
                with lock:
                    failed.append(a)
            return r
        K.run_pool(work, jobs, workers)
        if not failed:
            return
        jobs = failed


def boot_mean(xs, n=2000):
    rng = random.Random(SEED)
    xs = [x for x in xs if x is not None]
    bs = [np.mean([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(n)]
    return [round(100 * float(np.mean(xs)), 1), round(100 * float(np.percentile(bs, 2.5)), 1),
            round(100 * float(np.percentile(bs, 97.5)), 1), len(xs)]


def analyze():
    acc = accepted()
    nov = json.loads((RUN / 'pubchem.json').read_text()) if (RUN / 'pubchem.json').exists() else {}
    rows = []
    for pid, k in acc.items():
        r = {'id': pid, 'type': k['type'], 'requirement': k['requirement'],
             'novel_preserve': nov.get(f'{pid}__preserve'), 'novel_break': nov.get(f'{pid}__break')}
        for v in VARIANTS:
            u = [verdict(pid, v, s) for s in range(1, NSAMP + 1)]
            ub = [verdict(pid, v, s, 'b') for s in range(1, NSAMP + 1)]
            g = [ghost(pid, k, v, s) for s in range(1, NSAMP + 1)]
            r[v] = {'use': [x for x in u if x is not None], 'use_b': [x for x in ub if x is not None],
                    'parsed': sum(x is not None for x in g) / NSAMP,
                    'ghost_any': [float(bool(x)) for x in g if x is not None],
                    'ghost_smarts': [float(x['smarts'] > 0) for x in g if x is not None],
                    'ghost_flags': [dict(x) for x in g if x is not None]}
        r['prefilter'] = any(x == 1.0 for x in r['parent']['use'])
        rows.append(r)
    kept = [r for r in rows if r['prefilter'] and r['preserve']['use'] and r['break']['use']]
    mean = lambda xs: float(np.mean(xs)) if xs else None  # noqa: E731
    agree = [a == b for r in rows for v in VARIANTS for a, b in zip(r[v]['use'], r[v]['use_b'])]
    rep = {
        'designs': len(list((RUN / 'designs').glob('*.json'))), 'accepted': len(acc),
        'type_counts': Counter(r['type'] for r in rows), 'prefilter_pass': sum(r['prefilter'] for r in rows),
        'analyzed': len(kept),
        'novel_both': sum(bool(r['novel_preserve']) and bool(r['novel_break']) for r in kept),
        'judge_agreement': round(float(np.mean(agree)), 3) if agree else None,
        'kstar_use': {v: boot_mean([mean(r[v]['use']) for r in kept]) for v in VARIANTS},
        'PRIMARY_break_minus_preserve': boot_mean([mean(r['break']['use']) - mean(r['preserve']['use'])
                                                   for r in kept]),
        'ghost_any': {v: boot_mean([mean(r[v]['ghost_any']) for r in kept]) for v in VARIANTS},
        'ghost_smarts_break': boot_mean([mean(r['break']['ghost_smarts']) for r in kept]),
        'parsed': {v: round(float(np.mean([r[v]['parsed'] for r in kept])), 3) if kept else None for v in VARIANTS},
        'novel_only_break_minus_preserve': boot_mean([mean(r['break']['use']) - mean(r['preserve']['use'])
                                                      for r in kept if r['novel_preserve'] and r['novel_break']]),
    }
    rep['exploratory_by_requirement'] = {
        req: {'n': sum(r['requirement'] == req for r in kept),
              'kstar_use': {v: boot_mean([mean(r[v]['use']) for r in kept if r['requirement'] == req])
                            for v in VARIANTS},
              'break_minus_preserve': boot_mean([mean(r['break']['use']) - mean(r['preserve']['use'])
                                                 for r in kept if r['requirement'] == req])}
        for req in ('H', 'group') if any(r['requirement'] == req for r in kept)}
    p = rep['PRIMARY_break_minus_preserve']
    rep['PRIMARY_verdict'] = ('reasoning: break < preserve' if len(kept) and p[2] < 0 else
                              'not supported')
    rep['main_text_eligible'] = len(kept) >= 25
    (RUN / 'SUMMARY_M3.json').write_text(json.dumps({'summary': rep, 'rows': rows}, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='all', choices=['all', 'design', 'novelty', 'predict', 'judge', 'analyze'])
    ap.add_argument('--workers', type=int, default=100)
    ap.add_argument('--limit', type=int, default=0)
    cli = ap.parse_args()
    pids = [p['pmcid'] for p in SCOPE]
    if cli.limit:
        pids = pids[:cli.limit]
    if cli.phase in ('all', 'design'):
        phase('m3_design', job_design, [(p,) for p in pids], cli.workers)
        print(Counter(json.loads(f.read_text())['errors'] == [] for f in (RUN / 'designs').glob('*.json')))
    acc = {p: k for p, k in accepted().items() if p in pids}
    if cli.phase in ('all', 'novelty'):
        novelty()
    if cli.phase in ('all', 'predict'):
        phase('m3_predict', job_predict, [(p, k, v, s) for p, k in acc.items() for v in VARIANTS
                                          for s in range(1, NSAMP + 1)], cli.workers)
    if cli.phase in ('all', 'judge'):
        phase('m3_judge', job_judge, [(p, k, v, s, j) for p, k in acc.items() for v in VARIANTS
                                      for s in range(1, NSAMP + 1) for j in ('a', 'b')], cli.workers)
    analyze()


if __name__ == '__main__':
    main()
