"""Inert-knowledge probe: does the LLM know the hidden intermediates it omits?

Hidden (net-invisible) events are sigma bonds that are formed and later broken
within the GT mechanism (same definition as fukuyama_metrics._transient).

Per case, one model (gemini-3.8-flash) answers three conditions:
  E  executable mechanism (agent_v2.generate, PGL actions; same prompt as J3)
  S  spontaneous free-text mechanism with SMILES of every intermediate
  G  S plus a generic, case-independent cue to consider net-invisible
     intermediates
A hidden event counts as used in E if the predicted path forms and breaks the
same (H-anonymised) bond pair. It counts as recalled in S or G if any listed
SMILES matches a GT species that contains the transient bond. Matching uses a
sigma-graph key: heavy-atom connectivity plus H count per heavy atom; bond
orders and charges are ignored, so resonance forms match.

The pre-registered criteria are in REGISTRATION.json, written before any call.
The API key is read from OPENAI_API_KEY only.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'agent_v2'))
import agent_v2 as A  # noqa: E402
import common as C  # noqa: E402
from common import ROOT, chem, fm  # noqa: E402
from develop import metrics  # noqa: E402

# From 2026-11-03 all runs use the vps proxy (key in OPENAI_API_KEY); outputs up to then came from
# http://LLM-GATEWAY-1.example:8080/v1 gemini-3.8-flash, the same model per the user. Each output file records its endpoint.
BASE_URL = 'https://LLM-GATEWAY-2.example/v1'
MODEL = 'gemini-3.8-flash-high'
OUT = ROOT / 'outputs' / 'inert_knowledge_20261102'
MAX_TOKENS = 65536


def configure(model=None, base_url=None, max_tokens=None, pooled=False):
    """Point every condition at another model; its outputs go under models/<model>/.
    pooled: the model is the default one served under another name (another gateway), so its outputs
    join the default directories; each record still stores the model name it was produced with."""
    global MODEL, BASE_URL, OUT, MAX_TOKENS
    if model and model != MODEL:
        MODEL = model
        if not pooled:
            OUT = ROOT / 'outputs' / 'inert_knowledge_20261102' / 'models' / model
    BASE_URL = base_url or BASE_URL
    MAX_TOKENS = max_tokens or MAX_TOKENS

FREE_SYSTEM = 'You are an expert organic chemist who writes rigorous stepwise arrow-pushing mechanisms.'
FREE_TASK = ('Reactants and reagents (SMILES): {reactants}\n'
             'Products (SMILES): {products}\n'
             'Reaction conditions: {conditions}\n\n'
             'Propose the complete stepwise arrow-pushing mechanism from the reactants to the products. '
             'Describe each elementary step briefly.{cue}\n\n'
             'Finally, list the SMILES of every intermediate species that forms between the reactants and the '
             'products, in order of formation, one per line, inside <intermediates> and </intermediates>. Give '
             'explicit formal charges. Do not list the starting materials or the final products.')
CUE = ('\n\nBefore answering, consider carefully whether the mechanism passes through short-lived species that '
       'do not appear in the overall equation, for example bonds that form and later break again, rearrangements, '
       'ylides, onium ions, carbocations or carbanions, carbenes, ketenes, tetrahedral adducts, or neighbouring-group '
       'participation. Include every such species.')


# ------------------------------------------------------------------ sigma keys
def _key_from_graph(heavy, hcount, bonds, with_h=True):
    """Canonical SMILES of a heavy-atom graph with all bonds single and no charges."""
    from rdkit import Chem
    rw = Chem.RWMol()
    idx = {}
    for n in sorted(heavy):
        a = Chem.Atom(heavy[n])
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(int(hcount[n]) if with_h else 0)
        idx[n] = rw.AddAtom(a)
    for x, y in bonds:
        if x in idx and y in idx and rw.GetBondBetweenAtoms(idx[x], idx[y]) is None:
            rw.AddBond(idx[x], idx[y], Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return Chem.MolToSmiles(mol)


def state_component_keys(state, members):
    atoms = dict(chem.pgl.iter_atoms(state))
    bmap = chem.pgl.bond_map(state)
    heavy = {n: atoms[n].element for n in members if atoms[n].element != 'H'}
    hcount = {n: int(atoms[n].Hcount or 0) for n in heavy}
    for (x, y) in bmap:
        if x in heavy and atoms[y].element == 'H':
            hcount[x] += 1
        if y in heavy and atoms[x].element == 'H':
            hcount[y] += 1
    bonds = [(x, y) for (x, y) in bmap if x in heavy and y in heavy]
    return _key_from_graph(heavy, hcount, bonds), _key_from_graph(heavy, hcount, bonds, with_h=False)


def smiles_keys(smi):
    """Keys for every fragment of a model SMILES; empty list if unparsable."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return []
        mol.UpdatePropertyCache(strict=False)
    out = []
    for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
        frag.UpdatePropertyCache(strict=False)
        heavy, hcount = {}, {}
        for a in frag.GetAtoms():
            if a.GetSymbol() == 'H':
                continue
            heavy[a.GetIdx()] = a.GetSymbol()
            hcount[a.GetIdx()] = a.GetTotalNumHs(includeNeighbors=True)
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in frag.GetBonds()]
        if heavy:
            out.append((_key_from_graph(heavy, hcount, bonds), _key_from_graph(heavy, hcount, bonds, with_h=False)))
    return out


# ------------------------------------------------------------------ GT hidden species
def anon_pair(x, y):
    return tuple(sorted((fm._anon(x), fm._anon(y))))


def hidden_events(case):
    """{anonymised transient pair: {'heavy', 'window', 'keys', 'skel'}}.

    The hidden window of a transient pair runs from the first GT step that
    touches it to the step before the last one. The hidden species are the
    molecules in those states that contain a heavy atom of the pair and differ
    from every starting and final molecule. A pair formed and broken only
    within one step has an empty window: it is a concerted relay, not a
    hidden intermediate.
    """
    gold = fm.gold_steps_from_clean(case)
    per_step = [fm.sigma_events(s) for s in gold]
    transient = fm._transient(per_step)
    if not transient:
        return {}
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    states = C.gt_states(case)
    endpoint = {state_component_keys(s, c)[0] for s in (initial, states[-1]) for c in C._components(s)}
    events = {}
    for t in transient:
        touched = [i for i, ev in enumerate(per_step) if any((a, b) == t for _, a, b in ev)]
        window = range(touched[0], touched[-1])
        heavy_atoms = {lab for lab in t if lab != 'H*'}
        info = {'heavy': 'H*' not in t, 'window': len(window), 'keys': set(), 'skel': set()}
        for k in window:
            st = states[k]
            for comp in C._components(st):
                if not comp & heavy_atoms:
                    continue
                key, skel = state_component_keys(st, comp)
                if key and key not in endpoint:
                    info['keys'].add(key)
                    info['skel'].add(skel)
        events[t] = info
    return events


def endpoint_smiles(state):
    return '.'.join(C._molecule_smiles(state, c) for c in C._components(state))


# ------------------------------------------------------------------ jobs
def job(cond, case):
    dest = OUT / cond / f"{case['case_id']}.json"
    if dest.exists():
        return 'cached'
    model = C.Model(MODEL, OUT / cond / 'traces' / f"{case['case_id']}.jsonl", base_url=BASE_URL, max_tokens=MAX_TOKENS)
    try:
        if cond == 'E':
            actions, errors = A.generate(model, case, phase='E')
            rec = {'actions_pgl': actions, 'parse_errors': errors}
        else:
            initial = chem.parse_pgl_literal(case['initial_state_pgl'])
            target = chem.parse_pgl_literal(case['target_state_pgl'])
            prompt = FREE_TASK.format(reactants=endpoint_smiles(initial), products=endpoint_smiles(target),
                                      conditions=A.conditions_text(case), cue=CUE if cond == 'G' else '')
            answer = model.complete([{'role': 'system', 'content': FREE_SYSTEM}, {'role': 'user', 'content': prompt}], cond)
            m = re.search(r'<intermediates>(.*?)</intermediates>', answer, re.S)
            lines = [ln.strip().strip('`').strip() for ln in (m.group(1) if m else '').splitlines()]
            rec = {'prompt': prompt, 'intermediates': [ln for ln in lines if ln], 'tag_found': bool(m)}
    except Exception as exc:
        return f'error:{type(exc).__name__}'
    rec.update({'case_id': case['case_id'], 'cond': cond, 'model': MODEL, 'tokens': model.tokens, 'calls': len(model.events)})
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    return 'ok'


def run(cases, conds, workers):
    jobs = [(cond, c) for cond in conds for c in cases.values()]
    random.Random(7).shuffle(jobs)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(job, *j) for j in jobs]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            status = fut.result()
            if status.startswith('error') or done % 25 == 0:
                print(json.dumps({'done': done, 'of': len(jobs), 'status': status}), flush=True)


# ------------------------------------------------------------------ analysis
def load(cond, cid):
    f = OUT / cond / f'{cid}.json'
    return json.loads(f.read_text()) if f.exists() else None


def analyze(cases):
    rows = []
    per_case = {}
    for cid, case in cases.items():
        ev = hidden_events(case)
        recs = {c: load(c, cid) for c in ('E', 'S', 'G')}
        if not ev or any(r is None for r in recs.values()):
            continue
        sc = fm.score_case(case, recs['E']['actions_pgl'])
        used = set(map(tuple, sc['predicted_transient']))
        listed = {c: [k for smi in recs[c]['intermediates'] for k in smiles_keys(smi)] for c in ('S', 'G')}
        for t, info in ev.items():
            if not info['keys']:
                continue  # concerted relay or no distinct hidden species: not probeable
            row = {'case': cid, 'pair': '-'.join(t), 'heavy': info['heavy'], 'E': t in used}
            for c in ('S', 'G'):
                row[c] = any(k in info['keys'] for k, _ in listed[c])
                row[c + '_skel'] = any(s in info['skel'] for _, s in listed[c])
            rows.append(row)
        per_case[cid] = {'sigma_L1': bool(sc['sigma_L1']), 'n_hidden': len(ev)}
    (OUT / 'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')

    def block(sel, tag):
        rs = [r for r in rows if sel(r)]
        if not rs:
            return {'events': 0}
        ids = [r['case'] for r in rs]
        out = {'events': len(rs), 'cases': len(set(ids))}
        for k in ('E', 'S', 'G', 'S_skel', 'G_skel'):
            vals = [float(r[k]) for r in rs]
            lo, hi = metrics.clustered_bootstrap_ci(vals, ids, seed=20261102)
            out[f'recall_{k}'] = [round(statistics.mean(vals), 3), round(lo, 3), round(hi, 3)]
        omitted = [r for r in rs if not r['E']]
        out['omitted_in_E'] = len(omitted)
        if omitted:
            oid = [r['case'] for r in omitted]
            for name, f in (('known_S_or_G', lambda r: r['S'] or r['G']), ('known_G', lambda r: r['G']),
                            ('known_S', lambda r: r['S']), ('known_S_or_G_skel', lambda r: r['S_skel'] or r['G_skel'])):
                vals = [float(f(r)) for r in omitted]
                lo, hi = metrics.clustered_bootstrap_ci(vals, oid, seed=20261102)
                out[f'omitted_{name}'] = [round(statistics.mean(vals), 3), round(lo, 3), round(hi, 3)]
        diff = [float(r['G']) - float(r['S']) for r in rs]
        lo, hi = metrics.clustered_bootstrap_ci(diff, ids, seed=20261102)
        out['G_minus_S'] = [round(statistics.mean(diff), 3), round(lo, 3), round(hi, 3)]
        out['contingency_E_vs_SorG'] = dict(Counter(f"E{int(r['E'])}_K{int(r['S'] or r['G'])}" for r in rs))
        return out

    report = {'model': MODEL, 'cases_with_hidden_events': len(per_case),
              'all_events': block(lambda r: True, 'all'), 'heavy_atom_events': block(lambda r: r['heavy'], 'heavy'),
              'h_transfer_events': block(lambda r: not r['heavy'], 'h')}
    # E sanity: sigma_L1 on all cases with an E result, and visible/hidden recall for comparison with the main law
    allE = [fm.score_case(c, load('E', cid)['actions_pgl']) for cid, c in cases.items() if load('E', cid)]
    report['E_sigma_L1_all_cases'] = round(statistics.mean(float(s['sigma_L1']) for s in allE), 3) if allE else None
    report['E_cases'] = len(allE)
    tag_missing = {c: sum(1 for cid in cases if (r := load(c, cid)) and not r['tag_found']) for c in ('S', 'G')}
    report['free_text_missing_tag'] = tag_missing
    (OUT / 'SUMMARY.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=100)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--conds', nargs='+', default=['E', 'S', 'G'])
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--model')
    ap.add_argument('--base-url')
    ap.add_argument('--max-tokens', type=int)
    cli = ap.parse_args()
    configure(cli.model, cli.base_url, cli.max_tokens)
    sys.path.insert(0, str(HERE.parent / 'agent_v2'))
    import j3_dev
    cases = j3_dev.load_cases()
    if cli.limit:
        cases = dict(list(cases.items())[:cli.limit])
    if cli.selftest:
        n = hit = 0
        for case in cases.values():
            for t, info in hidden_events(case).items():
                n += 1
                hit += bool(info['keys'])
        print(json.dumps({'cases': len(cases), 'with_hidden': sum(bool(hidden_events(c)) for c in cases.values()),
                          'hidden_events': n, 'events_with_species_key': hit}))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if not cli.analyze_only:
        run(cases, cli.conds, cli.workers)
    analyze(cases)


if __name__ == '__main__':
    main()
