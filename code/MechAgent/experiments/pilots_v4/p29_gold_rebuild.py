"""P29: rebuild the literature gold standard with gemini-4-argon-build reading the full PDF (schemes included).

Per paper: one extraction call (PDF + prompt) -> programmatic consistency checks -> at most one repair call
(PDF + previous JSON + list of failed checks). Outputs cached under OUT/lit_gold_argon/<tag>/<PMCID>.json;
the parsed, checked record goes to OUT/lit_gold_argon/gold/<PMCID>.json.

Gold for scoring = intermediates with role "substrate_intermediate" that pass every check.

python p29_gold_rebuild.py --ids PMC1,PMC2 --workers 100      (pilot)
python p29_gold_rebuild.py --all --workers 100
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import threading
import time

from openai import OpenAI
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

import common4 as K

RDLogger.DisableLog('rdApp.*')
PAPERS = K.HERE.parent / 'lit150' / 'papers'
PAPERS_MORE = K.HERE.parent / 'lit_more' / 'papers'  # batch-2 confirmation papers (2026-11-06)


def pdf_path(pid):
    f = PAPERS / pid / 'main.pdf'
    return f if f.exists() else PAPERS_MORE / pid / 'main.pdf'
OUTD = K.OUT / 'lit_gold_argon'
MODEL = 'gemini-4-argon-build'
BASE = 'http://LLM-GATEWAY-1.example:8080/v1'
_SEM = threading.Semaphore(int(os.environ.get('ARGON_CAP', '100')))
METALS = set('Li Na K Mg Ca Al Sn Ti Zr V Cr Mn Fe Co Ni Cu Zn Ru Rh Pd Ag Ir Pt Au Hg Sc Y La Ce Sm Mo W Re Os '
             'Ga In Bi Pb Cd Eu Yb Gd Nd'.split())

PROMPT = '''The attached PDF is a 2026 paper with mechanistic studies. Read the text AND the schemes/figures.

Build a machine-checkable record of the mechanism the authors propose for their model reaction.

1. "model_reaction": the specific reaction for which the mechanism is drawn (usually the model substrates, e.g. 1a + 2a -> 3a).
   - "reactants": SMILES of the substrates and stoichiometric reagents (each a separate list item).
   - "catalysts": SMILES of catalysts, photocatalysts, ligands and additives that are not consumed (separate list).
   - "products": SMILES of the main product and stoichiometric by-products.
   - "conditions": short text (catalyst loading, solvent, light, temperature, atmosphere).
   - "compound_labels": map from the paper's labels (1a, 2a, 3a, ...) to SMILES for the compounds used above.
2. "intermediates": every species of the authors' proposed mechanism (scheme of the catalytic cycle / proposed mechanism),
   in mechanistic order. For each:
   - "label": the paper's label (e.g. "I", "Int-A", "B", "7"); "name": short description.
   - "smiles": the FULL structure as the authors draw it, with every generic group (R, Ar, L, X) instantiated with the
     model substrates / actual ligand of the model reaction. Radicals with explicit radical centres in brackets
     (e.g. [CH2]C, [O]), charges explicit, metals with their drawn ligands. Never use text, abbreviations or "*".
   - "role": one of
       "substrate_intermediate"  - contains ANY atoms of the substrates/stoichiometric reagents and lies on the proposed
                                   pathway, INCLUDING metal/catalyst complexes that carry a substrate fragment
                                   (e.g. oxidative-addition complexes L-Pd(Ar)X, substrate-bound catalyst, acyl-metal);
       "catalyst_state"          - species made ONLY of catalyst / photocatalyst / ligand / additive atoms, with no
                                   substrate atoms at all, e.g. Pd(0)L2, PC*, PC.-, Cu(I)L;
       "trapped_adduct"          - a species formed by a trap (TEMPO, BHT, DMPO, ...) or a detected off-path adduct;
       "off_cycle"               - resting states, decomposition, side products.
   - "evidence": "detected" (observed: HRMS, NMR, EPR, X-ray, isolated) or "proposed".
   - "source": where it is drawn (e.g. "Scheme 4", "Fig. 5c").
3. "clues": what the mechanistic experiments OBSERVED (species detected with m/z or spectra, trapped adducts, isolated
   intermediates, isotope-label positions, KIE values, radical-clock / TEMPO results). Observations with numbers only,
   never the authors' interpretation.
4. Self-check before answering: every substrate_intermediate must be derivable from the reactants (its non-metal atoms
   come from the reactants/reagents; elements must be present in reactants or catalysts), must differ from the reactants
   and products, and must use the same substituents as the model substrates. Fix any inconsistency.

If the paper proposes no mechanism with drawn intermediates, return "intermediates": [].
Answer with ONE JSON object between [ANSWER] and [/ANSWER].'''

REPAIR = '''Your previous record for this paper:
{prev}

Programmatic checks failed:
{errors}

Re-read the PDF (text and schemes) and return the corrected full record in the same JSON format between [ANSWER] and
[/ANSWER]. Fix each failed item: correct the SMILES (valid RDKit SMILES, model substituents, elements from the model
reaction), or change its role if it is not a substrate intermediate, or drop it if the paper does not draw it.'''


def pdf_part(pid):
    b = base64.b64encode(pdf_path(pid).read_bytes()).decode()
    return {'type': 'file', 'file': {'file_data': 'data:application/pdf;base64,' + b, 'filename': 'main.pdf'}}


def call(tag, pid, text):
    dest = OUTD / tag / f'{pid}.json'
    if dest.exists():
        return json.loads(dest.read_text())['answer']
    trace = OUTD / tag / 'traces' / f'{pid}.jsonl'
    trace.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=os.environ['FALLBACK_API_KEY'], base_url=BASE, timeout=900, max_retries=0)
    for attempt in range(6):
        t0 = time.monotonic()
        try:
            with _SEM:
                r = client.chat.completions.create(
                    model=MODEL, temperature=0.2, max_tokens=32768,
                    messages=[{'role': 'user', 'content': [{'type': 'text', 'text': text}, pdf_part(pid)]}])
            ans = r.choices[0].message.content or ''
            if '[ANSWER]' not in ans:
                raise RuntimeError('no [ANSWER] block')
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
    body = re.sub(r'```(?:json)?', '', m.group(1) if m else '')
    lo, hi = body.find('{'), body.rfind('}')
    if lo < 0:
        return None
    try:
        return json.loads(re.sub(r',\s*([}\]])', r'\1', body[lo:hi + 1]))
    except Exception:  # noqa: BLE001
        return None


def _mol(s):
    return Chem.MolFromSmiles(s) if isinstance(s, str) and s.strip() else None


def _elems(m):
    return {a.GetSymbol() for a in m.GetAtoms()} - {'H'}


def _organic(m):
    return sum(a.GetSymbol() not in METALS for a in m.GetAtoms())


def _mcs_frac(m, refs, rings=False):
    best = 0
    for r in refs:
        try:
            res = rdFMCS.FindMCS([m, r], timeout=2, bondCompare=rdFMCS.BondCompare.CompareAny,
                                 atomCompare=rdFMCS.AtomCompare.CompareElements, ringMatchesRingOnly=rings,
                                 completeRingsOnly=rings)
            best = max(best, res.numAtoms)
        except Exception:  # noqa: BLE001
            pass
    return best / max(_organic(m), 1)


def _lst(v):
    return v if isinstance(v, list) else ([v] if v else [])


def substrate_refs(rec):
    """(substrate refs, catalyst refs): reactant/product molecules and catalyst/ligand molecules (>=3 heavy atoms)."""
    mr = rec.get('model_reaction') or {}
    big = lambda smis: [m for s in smis if (m := _mol(s)) and m.GetNumHeavyAtoms() >= 3]  # noqa: E731
    cats = []
    for s in _lst(mr.get('catalysts')):  # ligands are scored metal-free, like the fragments
        cats += _demetal(s)
    return big(_lst(mr.get('reactants')) + _lst(mr.get('products'))), cats


def _demetal(smiles):
    """Metal-free fragments (>=3 heavy atoms) of a species, as RDKit molecules."""
    out = []
    for part in (smiles or '').split('.'):
        m = _mol(part)
        if m is None:
            continue
        rw = Chem.RWMol(m)
        for i in sorted((a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() in METALS), reverse=True):
            rw.RemoveAtom(i)
        try:
            frags = Chem.MolToSmiles(rw.GetMol()).split('.')
        except Exception:  # noqa: BLE001
            continue
        for f in frags:
            fm = Chem.MolFromSmiles(f) or Chem.MolFromSmiles(f, sanitize=False)
            if fm is not None and fm.GetNumHeavyAtoms() >= 3:
                out.append(fm)
    return out


def _is_sub(fm, refs):
    """Fragment derives from the substrates: >=60% complete-ring skeleton match, and closer to them than to any ligand."""
    subs, cats = refs
    s = _mcs_frac(fm, subs, rings=True)
    return s >= 0.6 and s >= (_mcs_frac(fm, cats, rings=True) if cats else 0)


def substrate_fragments(smiles, refs):
    """Relaxed scoring key: delete metal atoms, keep fragments derived from the substrates (see _is_sub) that are not
    identical to a reactant/product (an unchanged coordinated substrate carries no information). Canonical SMILES."""
    ends = {Chem.MolToSmiles(r) for r in refs[0]}
    whole = _mol(smiles)
    if whole is not None and not any(a.GetSymbol() in METALS for a in whole.GetAtoms()):
        can = Chem.MolToSmiles(whole)  # metal-free species: the species itself is the key
        return [] if can in ends else [can]
    out = set()
    for fm in _demetal(smiles):
        if _is_sub(fm, refs):
            try:
                can = Chem.MolToSmiles(fm)
            except Exception:  # noqa: BLE001
                continue
            if can not in ends:
                out.add(can)
    return sorted(out)


def _has_substrate_part(smiles, refs):
    return any(_is_sub(fm, refs) for fm in _demetal(smiles))


def reclassify(rec):
    """catalyst_state species that carry a substrate fragment are substrate intermediates (definition in the prompt)."""
    refs = substrate_refs(rec)
    n = 0
    for x in rec.get('intermediates') or []:
        if isinstance(x, dict) and x.get('role') == 'catalyst_state' and _mol(x.get('smiles')) is not None:
            if _has_substrate_part(x['smiles'], refs):
                x['role'], x['role_auto'] = 'substrate_intermediate', 'catalyst_state->substrate_intermediate'
                n += 1
    return n


def check(rec):
    """List of human-readable failed checks (empty = consistent)."""
    errs = []
    if not isinstance(rec, dict):
        return ['answer is not a single JSON object']
    mr = rec.get('model_reaction') or {}
    lst = lambda v: v if isinstance(v, list) else ([v] if v else [])  # noqa: E731
    reac, cat, prod = lst(mr.get('reactants')), lst(mr.get('catalysts')), lst(mr.get('products'))
    rm, cm, pm = [], [], []
    for name, smis, out in (('reactant', reac, rm), ('catalyst', cat, cm), ('product', prod, pm)):
        for s in smis:
            m = _mol(s)
            (out.append(m) if m else errs.append(f'{name} SMILES does not parse: {s!r}'))
    if not rm:
        errs.append('no valid reactant SMILES')
    if not pm:
        errs.append('no valid product SMILES')
    pool = set().union(*[_elems(m) for m in rm + cm]) if rm + cm else set()
    if pm and rm:
        missing = set().union(*[_elems(m) for m in pm]) - pool - METALS
        if missing:
            errs.append(f'product elements {sorted(missing)} not present in reactants/catalysts (reaction unbalanced)')
    ends = {Chem.MolToSmiles(m) for m in rm + pm}
    refs = [m for m in rm + pm if m.GetNumHeavyAtoms() >= 3]
    srefs = substrate_refs(rec)
    sub = [x for x in rec.get('intermediates') or [] if isinstance(x, dict) and x.get('role') == 'substrate_intermediate']
    seen = set()
    for x in sub:
        s, lab = x.get('smiles'), x.get('label') or x.get('name')
        m = _mol(s)
        if m is None:
            errs.append(f'intermediate {lab}: SMILES does not parse: {s!r}')
            continue
        can = Chem.MolToSmiles(m)
        if can in seen:
            errs.append(f'intermediate {lab}: duplicate of an earlier intermediate')
        seen.add(can)
        if can in ends:
            errs.append(f'intermediate {lab}: identical to a reactant/product')
        if _organic(m) < 3:
            errs.append(f'intermediate {lab}: fewer than 3 non-metal heavy atoms (catalyst_state?)')
        extra = _elems(m) - pool - {'H'}
        if extra:
            errs.append(f'intermediate {lab}: elements {sorted(extra)} not in reactants/catalysts')
        metal = any(a.GetSymbol() in METALS for a in m.GetAtoms())
        if refs and not (_has_substrate_part(s, srefs) if metal else _mcs_frac(m, refs) >= 0.6):
            errs.append(f'intermediate {lab}: shares <60% of its skeleton with any reactant/product (substituents differ?)')
    if not sub:
        errs.append('no substrate_intermediate')
    return errs


def job(pid):
    rec = parse(call('extract2', pid, PROMPT))
    if isinstance(rec, dict):
        reclassify(rec)
    errs = check(rec)
    used = 'extract2'
    if errs and rec is not None:
        rec2 = parse(call('repair2', pid, PROMPT + '\n\n' + REPAIR.format(prev=json.dumps(rec, ensure_ascii=False),
                                                                           errors='\n'.join('- ' + e for e in errs))))
        if isinstance(rec2, dict):
            reclassify(rec2)
            errs2 = check(rec2)
            if len(errs2) <= len(errs):
                rec, errs, used = rec2, errs2, 'repair2'
    if not isinstance(rec, dict):
        return 'error'
    gold, relaxed = [], []
    refs = substrate_refs(rec)
    for x in rec.get('intermediates') or []:
        if isinstance(x, dict) and x.get('role') == 'substrate_intermediate':
            lab = x.get('label') or x.get('name')
            if not any(e.startswith(f'intermediate {lab}:') for e in errs) and _mol(x.get('smiles')):
                gold.append(x['smiles'])
                relaxed.append(substrate_fragments(x['smiles'], refs))
    out = {'id': pid, 'used': used, 'errors': errs, 'record': rec, 'gold_intermediates': gold,
           'gold_fragments': relaxed,
           'reaction_ok': not any(e.startswith(('no valid', 'product elements')) for e in errs)}
    (OUTD / 'gold2').mkdir(parents=True, exist_ok=True)
    (OUTD / 'gold2' / f'{pid}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return 'clean' if not errs else f'{len(errs)} issues'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', default='')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--workers', type=int, default=100)
    cli = ap.parse_args()
    assert os.environ.get('FALLBACK_API_KEY'), 'FALLBACK_API_KEY required'
    if cli.all:
        import p20_lit150 as L
        ids = L.pids()
    else:
        ids = [x for x in cli.ids.split(',') if x]
    K.run_pool(job, [(p,) for p in ids], cli.workers)
