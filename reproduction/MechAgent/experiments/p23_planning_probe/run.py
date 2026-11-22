"""P2/P3 pre-experiment on FukuyamaBench: what should be planned before arrows?

FukuyamaBench is where template retrieval covers nothing (P1b) and where the
agent's misses are mostly transient events (75% of missing changes). Four
arms, the same model, exactly two calls each, scored by the Fukuyama
bond-change metrics:

  direct      path, then an independent self-audit (exp1 direct, two turns)
  interm      call 1 plans key intermediates (no arrows), call 2 writes arrows
  latent      call 1 lists hidden/transient events and their dependencies,
              call 2 writes arrows
  oracle_lat  DIAGNOSTIC: the GT transient bond events are given (unordered,
              no steps); path, then self-audit. Not a fair arm.

interm vs direct: does explicit planning help (P2-B)?  latent vs interm: is
event-level planning better than state-level planning (P3)?  oracle_lat
bounds how much perfect knowledge of hidden events would buy (P2-D).

Inference never reads gt_step_changes except to build the oracle_lat hint.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
EXP1 = ROOT / 'outputs' / 'publication_exp1_20261018'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'exp1_core_comparison'))
sys.path.append(str(EXP1 / 'snapshot'))  # only exp1_agent; develop must stay the repo's
from develop import metrics
from develop import score_external_benchmark as sx
from develop import score_fukuyama as sf
from develop import standalone_chemistry as chem
from develop import fukuyama_metrics as fm

FUKUYAMA = ROOT / 'outputs' / 'fukuyama' / 'clean.jsonl'
SEED = 20261024
ARMS = ['direct', 'interm', 'latent', 'oracle_lat']

SCHEMA = '''Return <path>{"mechanism_summary":"...","actions":[{"transformation_type":"reaction","electron_transfers":[{"source":["O1","lone_pair"],"target":["O1","C2","new_sigma_bond"],"electrons":2,"kind":"bond_forming"}]}]}</path>.
Use actual provided atom labels. Arrays in endpoints encode PGL structural tuples. Return the COMPLETE path, not just one step. At most 16 elementary steps. Include all necessary coupled arrows. No GT path or step count is available.'''

AUDIT = ('Independently audit your entire path for mechanistic causality, omitted events, exact atom labels, '
         'electron sources, proton conservation and step boundaries. Correct errors if needed. Return the complete '
         'final path in the required schema. You have no environment execution feedback.')

INTERM_PLAN = ('Before writing any arrows, plan the mechanism as an ordered list of KEY INTERMEDIATES between the initial '
               'and target states. For each intermediate give: (a) the bonds formed, broken or changed in order and the '
               'formal-charge changes relative to the previous intermediate, using the exact atom labels; (b) one line of '
               'chemical justification. Include intermediates whose features are absent from both endpoints. Do NOT write '
               'electron transfers yet. Return the plan as <plan>...</plan>.')

LATENT_PLAN = ('Before writing any arrows, identify the HIDDEN EVENTS of this mechanism: changes that occur but cannot be read '
               'from the endpoint difference, such as a bond that forms and later breaks, a charge that is created and later '
               'neutralized, protonation/deprotonation shuttles, activation of a reagent, neighbouring-group participation, '
               'or a rearrangement through a transient intermediate. For each hidden event give the exact atoms, why the '
               'chemistry requires it, and which other events (hidden or visible) it must precede or follow; mark events '
               'that are concerted, and name a plausible alternative where the evidence is ambiguous. Then list the visible '
               'endpoint changes and attach each to the hidden events that enable it. Do NOT write electron transfers yet. '
               'Return <events>...</events>.')

WRITE_PATH = ('Now convert your plan into the complete executable electron-transfer path, one ReactionTransformation per '
              'elementary step, consistent with the plan unless you find an error in it. ' )


def generation_tokens(usage):
    completion = int(usage.get('completion_tokens') or 0)
    total = int(usage.get('total_tokens') or 0)
    prompt = int(usage.get('prompt_tokens') or 0)
    reasoning = int((usage.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0)
    return max(completion, total - prompt, reasoning)


class Model:
    def __init__(self, args, trace):
        self.args, self.trace, self.events = args, trace, []
        if not args.mock:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=args.base_url,
                                 timeout=args.timeout, max_retries=0)

    def complete(self, messages, phase):
        start = time.monotonic()
        if self.args.mock:
            answer, usage, model = '<path>{"mechanism_summary":"mock","actions":[]}</path>', {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, 'mock'
        else:
            for attempt in range(8):
                try:
                    stream = self.client.chat.completions.create(
                        model=self.args.model, messages=messages, max_tokens=self.args.max_tokens,
                        temperature=0.2, reasoning_effort='high', stream=True, stream_options={'include_usage': True})
                    parts, usage, model = [], None, None
                    for chunk in stream:
                        model = chunk.model
                        if chunk.usage:
                            usage = chunk.usage.model_dump()
                        if chunk.choices and chunk.choices[0].delta.content:
                            parts.append(chunk.choices[0].delta.content)
                    answer = ''.join(parts)
                    if not usage:
                        raise RuntimeError('usage is missing')
                    break
                except Exception as exc:
                    status = getattr(exc, 'status_code', None)
                    self._save({'phase': phase, 'api_error': type(exc).__name__, 'status': status, 'retry': attempt,
                                'detail': str(exc).replace(os.environ['OPENAI_API_KEY'], '[REDACTED]')[:500],
                                'elapsed_seconds': time.monotonic() - start})
                    if attempt == 7 or (status and status not in {408, 429, 500, 502, 503, 504}):
                        raise
                    time.sleep(min(30, 5 * 2 ** attempt))
        event = {'phase': phase, 'returned_model': model, 'usage': usage, 'generation_tokens': generation_tokens(usage),
                 'elapsed_seconds': time.monotonic() - start, 'answer': answer}
        self.events.append(event)
        self._save(event)
        return answer

    def _save(self, event):
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        with self.trace.open('a') as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + '\n')


def parse_path(answer):
    value = answer.strip()
    if '<path>' in value:
        value = value.split('<path>', 1)[1].split('</path>', 1)[0].strip()
    if value.startswith('```'):
        value = value.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    actions = json.loads(value).get('actions')
    if not isinstance(actions, list) or len(actions) > 16:
        raise ValueError('actions must be a list of at most 16 steps')
    out = []
    for a in actions:
        transfers = [chem.ElectronTransfer(source=t['source'] if isinstance(t['source'], str) else tuple(t['source']),
                                           target=t['target'] if isinstance(t['target'], str) else tuple(t['target']),
                                           electrons=t['electrons'], kind=t['kind']) for t in a['electron_transfers']]
        out.append(chem.ReactionTransformation(a.get('transformation_type', 'reaction'), transfers))
    return out


def transient_bonds(steps):
    """Bond keys that are formed and broken (or broken and re-formed) along a path."""
    seen = collections.defaultdict(set)
    for step in steps:
        for a, b in step['form']:
            seen[sf._bond_key('bond', a, b)].add('form')
        for a, b in step['break']:
            seen[sf._bond_key('bond', a, b)].add('break')
    return {k for k, v in seen.items() if v == {'form', 'break'}}


def oracle_hint(case):
    gold = [sx._step_from_clean(s) for s in case['gt_step_changes']]
    keys = sorted(transient_bonds(gold))
    if not keys:
        return 'Hint (oracle diagnostic): this mechanism has no transient bond that forms and later breaks.'
    pairs = ', '.join(f'{a}-{b}' for _, a, b in keys)
    return ('Hint (oracle diagnostic): the following bonds are TRANSIENT in the accepted mechanism, i.e. each is formed '
            'at some point and broken later (H* = some hydrogen): ' + pairs + '. Their order and step grouping are not given.')


def conditions_block(case):
    """Pilot A: the recorded reagents, solvents and temperatures, in order."""
    items = [f"{t['role']}: {t['text']}" for t in case['conditions'] if t['text']]
    return ('Reaction conditions, in the order of operations (reagents that take part are already in the initial '
            'state): ' + '; '.join(items) + '.')


def analog_block(case, args):
    """Pilot B: solved mechanisms of other FukuyamaBench reactions, retrieved by
    analogs.py from conditions and endpoint net change only (frozen before any
    cond_analog result was seen)."""
    analogs = json.loads((args.output / 'analogs.json').read_text())['cases'][case['case_id']]['analogs']
    body = '\n\n'.join(f"Analog {i + 1}:\n{a['text']}" for i, a in enumerate(analogs))
    return ('Solved mechanisms of OTHER reactions, retrieved because their conditions or net structural change resemble '
            'this one. They use their own atom labels, unrelated to this task, and may not apply. Use them only as '
            'chemical analogies, especially for intermediates and transient bonds that cannot be read from the '
            'endpoints of this reaction.\n' + body)


def degraded_hint(case, mode):
    """Diagnostic hints of controlled quality (no deployable arm uses GT).

    oracle_half:  a deterministic half of the GT transient bonds (recall 0.5).
    oracle_noisy: all GT transient bonds plus as many distractors (precision
                  0.5). Distractors join atoms of the reaction region (atoms
                  whose bonds or charges differ between the endpoints, plus
                  their heavy neighbours) that are bonded in neither endpoint.
    """
    gold = [sx._step_from_clean(s) for s in case['gt_step_changes']]
    keys = sorted(transient_bonds(gold))
    rank = lambda x: hashlib.sha256(f'{SEED}:{case["case_id"]}:{x}'.encode()).hexdigest()
    pairs = [(a, b) for _, a, b in keys]
    if mode == 'oracle_half':
        pairs = sorted(pairs, key=lambda p: rank(p))[:max(1, len(pairs) // 2)]
    else:
        ini, tgt = chem.parse_pgl_literal(case['initial_state_pgl']), chem.parse_pgl_literal(case['target_state_pgl'])
        atoms = dict(chem.pgl.iter_atoms(ini))
        bi, bt = set(chem.pgl.bond_map(ini)), set(chem.pgl.bond_map(tgt))
        ta = dict(chem.pgl.iter_atoms(tgt))
        changed = {x for pair in bi ^ bt for x in pair} | {n for n in ta if n in atoms and ta[n].charge != atoms[n].charge}
        region = {n for n in changed if atoms.get(n) and atoms[n].element != 'H'}
        region |= {y for pair in bi for x, y in (pair, pair[::-1]) if x in region and atoms[y].element != 'H'}
        hetero = sorted(n for n in region if atoms[n].element in ('O', 'N', 'S'))
        gold_pairs = {tuple(sorted(sf._bond_key('bond', a, b)[1:])) for a, b in pairs}
        bonded = {tuple(sorted(p)) for p in bi | bt}
        candidates = [tuple(sorted((a, b))) for i, a in enumerate(sorted(region)) for b in sorted(region)[i + 1:]
                      if tuple(sorted((a, b))) not in bonded] + [('H*', n) for n in hetero]
        candidates = [c for c in candidates if tuple(sorted(c)) not in gold_pairs]
        pairs = pairs + sorted(candidates, key=lambda p: rank(p))[:len(pairs)]
        pairs = sorted(pairs, key=lambda p: rank(('mix',) + p))
    listed = ', '.join(f'{a}-{b}' for a, b in pairs)
    if mode == 'oracle_half':
        return ('Hint (diagnostic): the following bonds are TRANSIENT in the accepted mechanism (formed at some point '
                'and broken later; H* = some hydrogen). This list may be INCOMPLETE: ' + listed + '.')
    return ('Hint (diagnostic): CANDIDATE transient bonds (formed at some point and broken later; H* = some hydrogen). '
            'About half of these candidates are WRONG; judge each chemically: ' + listed + '.')


def candidate_transients(case):
    """Deployable candidate generator (no GT, no chemistry rules).

    Region = heavy atoms whose bonds or charge differ between the supplied
    endpoints, plus their heavy neighbours. Candidates = every region pair
    bonded in neither endpoint, every pair of a region atom with an atom of an
    element other than C/H/O/N (catalysts, Lewis acids, metals), and H* on
    every region atom. Frozen after a recall check on the 33 transient cases
    of clean.jsonl (93% of GT transient bonds, median 58 candidates); final
    evaluation must use held-out cases.
    """
    ini, tgt = chem.parse_pgl_literal(case['initial_state_pgl']), chem.parse_pgl_literal(case['target_state_pgl'])
    atoms, ta = dict(chem.pgl.iter_atoms(ini)), dict(chem.pgl.iter_atoms(tgt))
    bi, bt = set(chem.pgl.bond_map(ini)), set(chem.pgl.bond_map(tgt))
    adj = collections.defaultdict(set)
    for a, b in bi | bt:
        adj[a].add(b)
        adj[b].add(a)
    region = {x for pair in bi ^ bt for x in pair} | {n for n in ta if n in atoms and ta[n].charge != atoms[n].charge}
    region = {n for n in region if atoms[n].element != 'H'}
    region |= {y for x in list(region) for y in adj[x] if atoms[y].element != 'H'}
    extra = sorted({n for n, a in atoms.items() if a.element not in ('C', 'H', 'O', 'N')} - region)
    bonded = {frozenset(p) for p in bi | bt}
    ordered = sorted(region)
    pairs = {frozenset((a, b)) for i, a in enumerate(ordered) for b in ordered[i + 1:]}
    pairs |= {frozenset((a, b)) for a in extra for b in ordered}
    pairs = sorted(tuple(sorted(p)) for p in pairs if p not in bonded)
    return pairs + [('H*', n) for n in ordered]


def enum_hint(case):
    listed = ', '.join(f'{a}-{b}' for a, b in candidate_transients(case))
    return ('Mechanically enumerated CANDIDATE transient bonds (a bond formed at some point and broken later; '
            'H* = some hydrogen). They are all bonds absent from both endpoints near the reacting atoms or to '
            'catalysts/reagents, so the large majority are NOT part of the mechanism, and the mechanism may have no '
            'transient bond at all. For each one ask whether the chemistry under these structures requires it; '
            'use only those that are justified: ' + listed + '.')


RANK_TASK = ('Below is a mechanically enumerated list of CANDIDATE transient bonds for this reaction: bonds absent from '
             'both endpoints that join atoms near the reacting centre or to catalysts/reagents (H* = some hydrogen). A '
             'transient bond is formed at some point of the mechanism and broken later, so it appears in neither '
             'endpoint. Most candidates are not part of the mechanism. Screen them chemically and shortlist the ones '
             'the mechanism most plausibly requires, at most 8, most likely first. Omitting a required bond is much '
             'more costly than including a wrong one, so include a candidate when it is chemically plausible. Return an '
             'empty list only if the mechanism needs no transient bond at all. Do not write a mechanism. Return exactly '
             '<shortlist>["A-B", ...]</shortlist> using the candidate spellings.')


def pair_key(a, b):
    return tuple(sorted(sf._anon(x) for x in (a, b)))


def rank_case(arm, case, args, modules):
    """Stage 1 of the shortlist arm: one call screens the enumerated candidates."""
    stem = case['case_id']
    dest = args.output / 'rank' / f'{stem}.json'
    if dest.exists() and json.loads(dest.read_text()).get('finished'):
        return arm, stem, 'cached'
    model = Model(args, args.output / 'rank' / 'traces' / f'{stem}.jsonl')
    prompts = modules['prompts']
    facts = modules['skills'].pgl_structure_change_facts(case['initial_state_pgl'], case['target_state_pgl'])
    candidates = candidate_transients(case)
    allowed = {pair_key(a, b): f'{a}-{b}' for a, b in candidates}
    user = ('INITIAL:\n' + case['initial_state_pgl'] + '\nTARGET:\n' + case['target_state_pgl'] +
            '\nEndpoint-derived facts:\n' + json.dumps(facts) + '\nCandidates: ' + ', '.join(allowed.values()) + '\n' + RANK_TASK)
    messages = [{'role': 'system', 'content': '\n\n'.join([prompts.PGL_STATE_GUIDE, prompts.PLANNER_DECOMPOSITION_GUIDE])},
                {'role': 'user', 'content': user}]
    shortlist, rejected, errors = [], [], []
    try:
        answer = model.complete(messages, 'rank')
        raw = answer.split('<shortlist>', 1)[1].split('</shortlist>', 1)[0] if '<shortlist>' in answer else answer
        for item in json.loads(raw.strip()):
            a, _, b = str(item).partition('-')
            key = pair_key(a.strip(), b.strip())
            if key in allowed and key not in shortlist:
                shortlist.append(key)
            else:
                rejected.append(item)
        shortlist = shortlist[:8]
    except Exception as exc:
        errors.append({'phase': 'api' if not model.events else 'parse', 'error': f'{type(exc).__name__}: {str(exc)[:300]}'})
    result = {'case_id': stem, 'finished': not any(e['phase'] == 'api' for e in errors), 'candidates': len(candidates),
              'shortlist': [list(k) for k in shortlist], 'rejected_items': rejected, 'errors': errors,
              'generation_tokens': sum(e['generation_tokens'] for e in model.events)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return arm, stem, 'ok' if result['finished'] else 'infrastructure_pending'


def shortlist_hint(case, args):
    ranked = json.loads((args.output / 'rank' / f"{case['case_id']}.json").read_text())['shortlist']
    if not ranked:
        return ('A separate screening step found no plausible transient bond (a bond formed at some point and broken '
                'later) for this reaction; it may be wrong.')
    return ('Candidate transient bonds (formed at some point and broken later; H* = some hydrogen) shortlisted by a '
            'separate screening step. Some may be WRONG and the list may be incomplete; judge each chemically: ' +
            ', '.join(f'{a}-{b}' for a, b in ranked) + '.')


def run_case(arm, case, args, modules):
    stem = f"{case['case_id']}"
    dest = args.output / arm / f'{stem}.json'
    if dest.exists() and json.loads(dest.read_text()).get('finished'):
        return arm, stem, 'cached'
    trace = args.output / arm / 'traces' / f'{stem}.jsonl'
    model = Model(args, trace)
    prompts = modules['prompts']
    system = '\n\n'.join([prompts.PGL_STATE_GUIDE, prompts.PGL_ACTION_GUIDE, prompts.PLANNER_DECOMPOSITION_GUIDE, SCHEMA])
    facts = modules['skills'].pgl_structure_change_facts(case['initial_state_pgl'], case['target_state_pgl'])
    user = ('Using only the initial and target molecular graphs, infer a complete stepwise elementary electron-transfer '
            'mechanism. Reaction name, class, GT intermediates and GT step count are hidden.\nINITIAL:\n' +
            case['initial_state_pgl'] + '\nTARGET:\n' + case['target_state_pgl'] + '\nEndpoint-derived facts:\n' + json.dumps(facts))
    # 'direct@2' is a further independent sample of the 'direct' arm (same
    # prompts, temperature and budget), stored in its own directory.
    kind = arm.split('@')[0]
    if kind == 'oracle_lat':
        user += '\n' + oracle_hint(case)
    if kind == 'enum':
        user += '\n' + enum_hint(case)
    if kind == 'shortlist':
        user += '\n' + shortlist_hint(case, args)
    if kind in ('oracle_half', 'oracle_noisy'):
        user += '\n' + degraded_hint(case, kind)
    if kind in ('cond', 'cond_analog'):
        user += '\n' + conditions_block(case)
    if kind == 'cond_analog':
        user += '\n' + analog_block(case, args)
    base = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    errors, path = [], []
    try:
        if kind in ('direct', 'oracle_lat', 'cond', 'cond_analog', 'oracle_half', 'oracle_noisy', 'enum', 'shortlist'):
            first = model.complete(base, f'{arm}.path')
            second = model.complete(base + [{'role': 'assistant', 'content': first}, {'role': 'user', 'content': AUDIT}], f'{arm}.audit')
            candidates = [second, first]
        else:
            plan_msg = INTERM_PLAN if kind == 'interm' else LATENT_PLAN
            plan = model.complete(base + [{'role': 'user', 'content': plan_msg}], f'{arm}.plan')
            final = model.complete(base + [{'role': 'user', 'content': plan_msg}, {'role': 'assistant', 'content': plan},
                                           {'role': 'user', 'content': WRITE_PATH + SCHEMA}], f'{arm}.path')
            candidates = [final]
        for text in candidates:
            try:
                path = parse_path(text)
                break
            except Exception as exc:
                errors.append({'phase': 'parse', 'error': str(exc)[:300]})
    except Exception as exc:
        errors.append({'phase': 'api', 'error': f'{type(exc).__name__}: {str(exc)[:300]}'})
    result = {'case_id': case['case_id'], 'arm': arm, 'partition': case['partition'], 'finished': not any(e['phase'] == 'api' for e in errors),
              'actions_pgl': [repr(a) for a in path], 'errors': errors, 'calls': len(model.events),
              'generation_tokens': sum(e['generation_tokens'] for e in model.events),
              'total_tokens': sum(int((e['usage'] or {}).get('total_tokens') or 0) for e in model.events)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return arm, stem, 'ok' if result['finished'] else 'infrastructure_pending'


def score(case, result):
    """Replay the executable prefix and compare per-step bond changes with GT."""
    state = chem.parse_pgl_literal(case['initial_state_pgl'])
    predicted, executed = [], 0
    for text in result['actions_pgl']:
        try:
            before = chem.state_signature(state)
            state, _ = chem.apply_action(state, chem.parse_pgl_literal(text).to_dict())
        except Exception:
            break
        executed += 1
        predicted.append(sx._step_from_signatures(before, chem.state_signature(state)))
    # Primary metrics (sigma_L1, resonance-normalized intermediates in order)
    # come from develop/fukuyama_metrics.py; the rest is the legacy reading.
    primary = fm.score_case(case, result['actions_pgl'])
    gold = [sx._step_from_clean(s) for s in case['gt_step_changes']]
    p = [sx.to_counter(s, normalized=True) for s in predicted]
    g = [sx.to_counter(s, normalized=True) for s in gold]
    cmp = metrics.compare_path(sx.as_actions(p), sx.as_actions(g), reaction_id=case['case_id'])
    bonds_only = metrics.compare_path(sx.as_actions(p, bonds_only=True), sx.as_actions(g, bonds_only=True)).transfer_set_match
    gt_t, pr_t = transient_bonds(gold), transient_bonds(predicted)
    return {'L1': cmp.transfer_set_match, 'L2': cmp.transfer_sequence_match, 'L3': cmp.step_partition_match,
            'bonds_only_L1': bonds_only, 'diagnosis': cmp.boundary_diagnosis,
            'set_f1_parts': (cmp.set_matched, cmp.predicted_transfers, cmp.gold_transfers),
            'sigma_L1': primary['sigma_L1'], 'gt_states_in_order': primary['intermediates_in_order'],
            'gt_state_prefix_hit': primary['intermediates_hit'],
            'executed_steps': executed, 'proposed_steps': len(result['actions_pgl']), 'gold_steps': len(gold),
            'gt_transient': len(gt_t), 'pred_transient': len(pr_t), 'transient_hit': len(gt_t & pr_t)}


def summarize(args, cases):
    by_id = {c['case_id']: c for c in cases}
    rows = []
    for arm in ARMS:
        for f in sorted((args.output / arm).glob('*.json')):
            r = json.loads(f.read_text())
            if r.get('finished'):
                rows.append({'arm': arm, 'case_id': r['case_id'], 'partition': r['partition'],
                             'generation_tokens': r['generation_tokens'], **score(by_id[r['case_id']], r)})
    summary = {}
    for arm in ARMS:
        sel = [r for r in rows if r['arm'] == arm]
        if not sel:
            continue
        m, pt, gt = (sum(r['set_f1_parts'][i] for r in sel) for i in range(3))
        hits = sum(r['transient_hit'] for r in sel)
        summary[arm] = {
            'n': len(sel),
            **{k: {'rate': statistics.mean(r[k] for r in sel),
                   'ci95': metrics.clustered_bootstrap_ci([int(r[k]) for r in sel], [r['case_id'] for r in sel], seed=SEED)}
               for k in ('sigma_L1', 'gt_states_in_order', 'L1', 'L2', 'L3', 'bonds_only_L1')},
            'gt_state_prefix_recall': sum(r['gt_state_prefix_hit'] for r in sel) / sum(r['gold_steps'] for r in sel),
            'by_partition_L1': {p: statistics.mean(r['L1'] for r in sel if r['partition'] == p)
                                for p in sorted({r['partition'] for r in sel})},
            'change_f1': 2 * m / (pt + gt) if pt + gt else 0,
            'transient_recall': hits / max(1, sum(r['gt_transient'] for r in sel)),
            'transient_precision': hits / max(1, sum(r['pred_transient'] for r in sel)),
            'fully_executable': statistics.mean(r['executed_steps'] == r['proposed_steps'] > 0 for r in sel),
            'mean_generation_tokens': statistics.mean(r['generation_tokens'] for r in sel),
            'diagnosis': dict(collections.Counter(r['diagnosis'] for r in sel)),
        }
    paired = {}
    for a, b in [('direct', 'interm'), ('interm', 'latent'), ('direct', 'latent'), ('direct', 'oracle_lat')]:
        A = {r['case_id']: r for r in rows if r['arm'] == a}
        B = {r['case_id']: r for r in rows if r['arm'] == b}
        keys = sorted(set(A) & set(B))
        if keys:
            block = {'n': len(keys)}
            for field in ('sigma_L1', 'gt_states_in_order', 'L1'):
                d = [int(B[k][field]) - int(A[k][field]) for k in keys]
                block[field] = {'delta': statistics.mean(d), 'improved': sum(x > 0 for x in d),
                                'regressed': sum(x < 0 for x in d), 'ci95': metrics.clustered_bootstrap_ci(d, keys, seed=SEED)}
            paired[f'{a}->{b}'] = block
    (args.output / 'scored_rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (args.output / 'summary.json').write_text(json.dumps({'arms': summary, 'paired': paired}, ensure_ascii=False, indent=2))
    print(json.dumps({'arms': {k: {'n': v['n'], 'L1': v['L1']['rate'], 'states_in_order': v['gt_states_in_order']['rate'],
                                   'transient_recall': v['transient_recall']}
                               for k, v in summary.items()}, 'paired': paired}, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'p23_planning_probe_20261024')
    ap.add_argument('--base-url', default='https://LLM-GATEWAY-2.example/v1')
    ap.add_argument('--model', default='gemini-3.8-flash-high')
    ap.add_argument('--per-partition', type=int, default=8)
    ap.add_argument('--arms', nargs='+', default=ARMS)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--timeout', type=float, default=300)
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--mock', action='store_true', help='no API calls; checks the pipeline end to end')
    ap.add_argument('--summarize-only', action='store_true')
    ap.add_argument('--transient-only', action='store_true', help='keep only cases whose GT has a transient bond')
    args = ap.parse_args()
    all_cases = [json.loads(l) for l in FUKUYAMA.read_text().splitlines()]
    cases = []
    for part in ['A', 'B', 'C']:
        pool = sorted((c for c in all_cases if c['partition'] == part),
                      key=lambda c: hashlib.sha256(f"{SEED}:{c['case_id']}".encode()).hexdigest())
        cases += pool[:args.per_partition]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.transient_only:
        # Selection uses GT only to pick the diagnostic subset; cases.json keeps all 24.
        run_cases = [c for c in cases if transient_bonds([sx._step_from_clean(s) for s in c['gt_step_changes']])]
    else:
        run_cases = cases
    (args.output / 'cases.json').write_text(json.dumps([c['case_id'] for c in cases]))
    if not args.summarize_only:
        modules = {'prompts': importlib.import_module('exp1_agent.prompts'),
                   'skills': importlib.import_module('exp1_agent.planner_skills')}
        jobs = [(arm, c) for c in run_cases for arm in args.arms]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for fut in concurrent.futures.as_completed([pool.submit(rank_case if a == "rank" else run_case, a, c, args, modules) for a, c in jobs]):
                print(json.dumps({'done': fut.result()}), flush=True)
    summarize(args, cases)


if __name__ == '__main__':
    main()
