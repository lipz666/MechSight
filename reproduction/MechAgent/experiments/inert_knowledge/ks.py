"""Knowledge-source experiments for the inert-knowledge story.

KS-1 ID          The model identifies the transformation (named reaction or textbook class,
                 confidence, and whether it recognises the exact problem). Free text, SMILES input
                 as in S. Tests whether hidden intermediates the model knows are concentrated in
                 reactions it can name, i.e. whether the knowledge is indexed by reaction schema.
KS-2 E_selfname  E plus the model's own KS-1 identification as a hint (no GT information). Tests
                 whether querying the schema activates the knowledge inside the executable format.
KS-3 S_analog    S on a homologue: one CH3 added on a spectator carbon far from every GT bond
                 change, so the mechanism is unchanged. S_r2 (S resampled on the original) is the
                 noise reference. Tests whether the knowledge transfers beyond the memorised problem.
Registration: outputs/inert_knowledge_20261102/REGISTRATION_ks.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import random
import re
import statistics
import sys
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase2  # noqa: E402
import probe as P  # noqa: E402
from probe import A, C, chem, fm, metrics  # noqa: E402

ID_TASK = ('Reactants and reagents (SMILES): {reactants}\n'
           'Products (SMILES): {products}\n'
           'Reaction conditions: {conditions}\n\n'
           'Identify this transformation. Do not write a mechanism. Answer with one JSON object inside <id> and </id> '
           'with the keys: "named_reaction" (the name of the named reaction, or null if it is not a named reaction), '
           '"reaction_class" (the textbook mechanistic class in a few words), "confidence" (0 to 1, that the '
           'identification is right), and "seen_problem" (true if you recognise this exact problem, e.g. from a '
           'textbook or exam, otherwise false).')
NAME_HINT = 'Hint: this transformation has been identified as {name} ({cls}).'
ANALOG_LABEL = 'C99001'


# ------------------------------------------------------------------ KS-3 analog construction
def analog_case(case):
    """The case with one CH3 on a carbon >= 3 bonds from every atom any GT step touches, or None."""
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    touched = set()
    for s in case['gt_step_changes']:
        for key in ('bonds_formed', 'bonds_broken', 'bond_order_changes'):
            for item in s[key]:
                touched |= {item[0], item[1]}
        touched |= {c[0] for c in s['charge_changes']}
    bmap = chem.pgl.bond_map(initial)
    adj = {}
    for (x, y) in bmap:
        adj.setdefault(x, set()).add(y)
        adj.setdefault(y, set()).add(x)
    dist = {t: 0 for t in touched if t in adj}
    queue = deque(dist)
    while queue:
        n = queue.popleft()
        for m in adj.get(n, ()):
            if m not in dist:
                dist[m] = dist[n] + 1
                queue.append(m)
    atoms = dict(chem.pgl.iter_atoms(initial))
    eligible = sorted(n for n, a in atoms.items() if a.element == 'C' and int(a.Hcount or 0) >= 1
                      and not a.charge and dist.get(n, 0) >= 3)
    if not eligible:
        return None
    site = max(eligible, key=lambda n: (dist[n], n))

    def mutate(state_pgl):
        st = chem.parse_pgl_literal(state_pgl)
        for sp in st.species:
            for sub in sp.molecule.substructures:
                if site in sub.atomdict:
                    sub.atomdict[site].Hcount = int(sub.atomdict[site].Hcount or 0) - 1
                    sub.atomdict[ANALOG_LABEL] = chem.Atom('C', Hcount=3)
                    sub.bondlist.append(chem.Bond(site, ANALOG_LABEL, 'single'))
                    return repr(st)
        raise ValueError('site not found')

    new = copy.deepcopy(case)
    new['initial_state_pgl'] = mutate(case['initial_state_pgl'])
    new['target_state_pgl'] = mutate(case['target_state_pgl'])
    new['analog_site'] = site
    new['analog_site_distance'] = dist[site]
    return new


# ------------------------------------------------------------------ jobs
def free_text(cond, case, task):
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    target = chem.parse_pgl_literal(case['target_state_pgl'])
    fields = dict(reactants=P.endpoint_smiles(initial), products=P.endpoint_smiles(target), conditions=A.conditions_text(case))
    prompt = task.format(cue='', **fields)
    model = C.Model(P.MODEL, P.OUT / cond / 'traces' / f"{case['case_id']}.jsonl", base_url=P.BASE_URL, max_tokens=P.MAX_TOKENS)
    answer = model.complete([{'role': 'system', 'content': P.FREE_SYSTEM}, {'role': 'user', 'content': prompt}], cond)
    return prompt, answer, model


def parse_id(answer):
    """The JSON object inside <id></id>; falls back to the first {...} when the tags are missing or unclosed."""
    m = re.search(r'<id>(.*?)</id>', answer, re.S) or re.search(r'(\{.*?\})', answer, re.S)
    try:
        return json.loads(m.group(1)) if m else None
    except json.JSONDecodeError:
        return None


def identification(cid):
    rec = P.load('ID', cid) or {}
    return rec.get('id') or (parse_id(rec['answer']) if rec.get('answer') else None) or {}


def job(cond, case):
    dest = P.OUT / cond / f"{case['case_id']}.json"
    if dest.exists():
        return 'cached'
    try:
        if cond == 'ID':
            prompt, answer, model = free_text(cond, case, ID_TASK)
            rec = {'prompt': prompt, 'answer': answer, 'id': parse_id(answer)}
        elif cond in ('S_r2', 'S_analog'):
            target_case = case if cond == 'S_r2' else analog_case(case)
            prompt, answer, model = free_text(cond, target_case, P.FREE_TASK)
            m = re.search(r'<intermediates>(.*?)</intermediates>', answer, re.S)
            lines = [ln.strip().strip('`').strip() for ln in (m.group(1) if m else '').splitlines()]
            rec = {'prompt': prompt, 'intermediates': [ln for ln in lines if ln], 'tag_found': bool(m)}
            if cond == 'S_analog':
                rec['analog_site'] = target_case['analog_site']
        elif cond == 'E_selfname':
            ident = identification(case['case_id'])
            name = ident.get('named_reaction') or ident.get('reaction_class')
            if not name:
                return 'skip:no_identification'
            hint = NAME_HINT.format(name=name, cls=ident.get('reaction_class') or 'class not given')
            model = C.Model(P.MODEL, P.OUT / cond / 'traces' / f"{case['case_id']}.jsonl", base_url=P.BASE_URL, max_tokens=P.MAX_TOKENS)
            actions, errors = phase2.generate(model, A.user_prompt(case, hint=hint), cond)
            rec = {'hint': hint, 'actions_pgl': actions, 'parse_errors': errors}
        else:
            raise ValueError(cond)
    except Exception as exc:
        return f'error:{type(exc).__name__}'
    rec.update({'case_id': case['case_id'], 'cond': cond, 'model': P.MODEL, 'tokens': model.tokens, 'calls': len(model.events)})
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    return 'ok'


def run(jobs, workers):
    random.Random(13).shuffle(jobs)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(job, *j) for j in jobs]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            status = fut.result()
            if status.startswith('error') or done % 50 == 0:
                print(json.dumps({'done': done, 'of': len(jobs), 'status': status}), flush=True)


# ------------------------------------------------------------------ analysis
def ci(vals, ids):
    if not vals:
        return None
    lo, hi = metrics.clustered_bootstrap_ci(vals, ids, seed=20261102)
    return [round(statistics.mean(vals), 3), round(lo, 3), round(hi, 3), len(vals)]


def recalled(rec, info):
    return any(k in info['keys'] for smi in rec['intermediates'] for k, _ in P.smiles_keys(smi))


def analyze(cases):
    rows = []
    for cid, case in cases.items():
        ev = {t: i for t, i in P.hidden_events(case).items() if i['keys']}
        if not ev:
            continue
        rec = {c: P.load(c, cid) for c in ('E', 'E_r2', 'S', 'G', 'ID', 'E_selfname', 'S_r2', 'S_analog')}
        if any(rec[c] is None for c in ('E', 'E_r2', 'S', 'G', 'ID', 'S_r2')):
            continue
        used = {c: set(map(tuple, fm.score_case(case, rec[c]['actions_pgl'])['predicted_transient']))
                for c in ('E', 'E_r2', 'E_selfname') if rec[c]}
        ident = identification(cid)
        acase = analog_case(case) if rec['S_analog'] else None
        aev = P.hidden_events(acase) if acase else {}
        for t, info in ev.items():
            row = {'case': cid, 'partition': case['partition'], 'heavy': info['heavy'],
                   'E': t in used['E'], 'E_r2': t in used['E_r2'],
                   'K': recalled(rec['S'], info) or recalled(rec['G'], info),
                   'S': recalled(rec['S'], info), 'S_r2': recalled(rec['S_r2'], info),
                   'named': bool(ident.get('named_reaction')), 'confidence': float(ident.get('confidence') or 0),
                   'seen_problem': bool(ident.get('seen_problem'))}
            if 'E_selfname' in used:
                row['E_selfname'] = t in used['E_selfname']
            if acase and t in aev and aev[t]['keys']:
                row['S_analog'] = recalled(rec['S_analog'], aev[t])
            rows.append(row)
    (P.OUT / 'rows_ks.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')

    def block(rs):
        out = {'events': len(rs), 'cases': len({r['case'] for r in rs})}
        if not rs:
            return out
        ids = [r['case'] for r in rs]
        # KS-1: knowledge and use by self-identification
        for flag in ('named', 'seen_problem'):
            for val in (True, False):
                sub = [r for r in rs if r[flag] == val]
                sid = [r['case'] for r in sub]
                out[f'{flag}={val}'] = {'K': ci([float(r['K']) for r in sub], sid), 'E': ci([float(r['E']) for r in sub], sid),
                                        'K_given_E0': ci([float(r['K']) for r in sub if not r['E']], [r['case'] for r in sub if not r['E']])}
        hi = [r for r in rs if r['confidence'] >= 0.7]
        lo = [r for r in rs if r['confidence'] < 0.7]
        out['confidence>=0.7'] = {'K': ci([float(r['K']) for r in hi], [r['case'] for r in hi])}
        out['confidence<0.7'] = {'K': ci([float(r['K']) for r in lo], [r['case'] for r in lo])}
        for part in 'ABC':
            sub = [r for r in rs if r['partition'] == part]
            out[f'partition_{part}'] = {'K': ci([float(r['K']) for r in sub], [r['case'] for r in sub]),
                                        'E': ci([float(r['E']) for r in sub], [r['case'] for r in sub])}
        # KS-2: self-identification as a cue
        sn = [r for r in rs if 'E_selfname' in r]
        if sn:
            sid = [r['case'] for r in sn]
            out['E_selfname_minus_meanE'] = ci([float(r['E_selfname']) - (float(r['E']) + float(r['E_r2'])) / 2 for r in sn], sid)
            om = [r for r in sn if not r['E'] and not r['E_r2']]
            out['E_selfname_on_omitted_both'] = ci([float(r['E_selfname']) for r in om], [r['case'] for r in om])
        # KS-3: analog transfer
        an = [r for r in rs if 'S_analog' in r]
        if an:
            aid = [r['case'] for r in an]
            out['S_original_mean_of_2'] = ci([(float(r['S']) + float(r['S_r2'])) / 2 for r in an], aid)
            out['S_analog'] = ci([float(r['S_analog']) for r in an], aid)
            out['S_analog_minus_S_mean'] = ci([float(r['S_analog']) - (float(r['S']) + float(r['S_r2'])) / 2 for r in an], aid)
            out['S_r2_minus_S'] = ci([float(r['S_r2']) - float(r['S']) for r in an], aid)
        return out

    report = {'model': P.MODEL, 'heavy_atom_events': block([r for r in rows if r['heavy']]), 'all_events': block(rows)}
    idents = [identification(cid) for cid in cases if P.load('ID', cid)]
    report['identification'] = {'cases': len(idents), 'parsed': sum(bool(i) for i in idents),
                                'named': sum(bool(i.get('named_reaction')) for i in idents),
                                'seen_problem': sum(bool(i.get('seen_problem')) for i in idents)}
    report['analog_eligible_cases'] = sum(1 for cid, c in cases.items() if P.hidden_events(c) and analog_case(c))
    (P.OUT / 'SUMMARY_ks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=300)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stage', choices=['1', '2', 'analyze'], default='1',
                    help='1: ID, S_r2, S_analog; 2: E_selfname (needs ID)')
    ap.add_argument('--model')
    ap.add_argument('--base-url')
    ap.add_argument('--max-tokens', type=int)
    ap.add_argument('--pooled', action='store_true', help='--model serves the default model; keep the default output dirs')
    cli = ap.parse_args()
    P.configure(cli.model, cli.base_url, cli.max_tokens, pooled=cli.pooled)
    sys.path.insert(0, str(HERE.parent / 'agent_v2'))
    import j3_dev
    cases = j3_dev.load_cases()
    if cli.limit:
        cases = dict(list(cases.items())[:cli.limit])
    hidden = {cid: c for cid, c in cases.items() if any(i['keys'] for i in P.hidden_events(c).values())}
    if cli.stage == '1':
        jobs = [('ID', c) for c in cases.values()] + [('S_r2', c) for c in hidden.values()]
        jobs += [('S_analog', c) for c in hidden.values() if analog_case(c)]
        run(jobs, cli.workers)
    elif cli.stage == '2':
        run([('E_selfname', c) for c in hidden.values()], cli.workers)
    analyze(cases)


if __name__ == '__main__':
    main()
