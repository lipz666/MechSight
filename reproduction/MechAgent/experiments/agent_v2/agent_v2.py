"""MechAgent-v2: propose (motif library) -> execute (PGL) -> compare (LLM judge).

One case, one model:
  base   direct path + self-audit (2 calls)            [the plain-LLM answer]
  hint   same prompt + motif-library candidate transient bonds (2 calls)
  judge  base vs hint, anonymized, both orders (2 calls); the hint path is
         taken only if it wins in BOTH orders, otherwise the base path stays.
Compute-matched control: direct with 3 independent samples (6 calls) and a
majority vote over their sigma-event multisets.

All arms receive the same initial/target PGL, endpoint facts and conditions.
"""
from __future__ import annotations

import collections
import importlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
import motifs as M
from common import ROOT, chem, fm

sys.path.insert(0, str(ROOT / 'experiments' / 'p23_planning_probe'))
import run as P23  # SCHEMA, AUDIT, parse_path, prompts snapshot path

PROMPTS = importlib.import_module('exp1_agent.prompts')
SKILLS = importlib.import_module('exp1_agent.planner_skills')
SYSTEM = '\n\n'.join([PROMPTS.PGL_STATE_GUIDE, PROMPTS.PGL_ACTION_GUIDE, PROMPTS.PLANNER_DECOMPOSITION_GUIDE, P23.SCHEMA])
JUDGE_SYSTEM = ('You are an expert physical organic chemist who reviews proposed arrow-pushing mechanisms. Atom labels '
                'refer to the INITIAL state; every mechanism below is atom- and charge-balanced and executable from it.')
JUDGE_TASK = ('Two candidate stepwise mechanisms, A and B, are given for the same transformation. Each step lists the '
              'bonds formed and broken, bond-order and formal-charge changes, and the species formed or changed. Decide '
              'which mechanism is the chemically correct one under the given conditions, judging the plausibility of '
              'every intermediate, the order of events, and whether steps are concerted or stepwise. Reason briefly, '
              'then answer with exactly <choice>A</choice> or <choice>B</choice>.')


def conditions_text(case):
    cond = case.get('conditions')
    if isinstance(cond, str):
        return cond or 'not given'
    return '; '.join(f"{t['role']}: {t['text']}" for t in (cond or []) if t.get('text')) or 'not given'


def user_prompt(case, hint=None, target_block=None):
    facts = SKILLS.pgl_structure_change_facts(case['initial_state_pgl'], case['target_state_pgl']) if case.get('target_state_pgl') else None
    text = ('Using the initial and target molecular structures and the reaction conditions, infer a complete stepwise '
            'elementary electron-transfer mechanism. Reaction name, class, GT intermediates and GT step count are hidden.\n'
            f"INITIAL:\n{case['initial_state_pgl']}\n")
    text += target_block if target_block else f"TARGET:\n{case['target_state_pgl']}\n"
    if facts is not None:
        text += 'Endpoint-derived facts:\n' + json.dumps(facts) + '\n'
    text += f'Reaction conditions: {conditions_text(case)}\n'
    if hint:
        text += hint + '\n'
    return text


def motif_hint(keys):
    if not keys:
        return None
    listed = ', '.join(f'{a}-{b}' for a, b in keys)
    return ('Candidate transient bonds suggested by a mechanistic motif library mined from expert mechanisms (a transient '
            'bond is formed at some point and broken later; H* = some hydrogen). Most of these candidates are WRONG and '
            'the list may be incomplete; use a candidate only if the chemistry requires it: ' + listed + '.')


def generate(model, case, hint=None, target_block=None, phase='path'):
    base = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user_prompt(case, hint, target_block)}]
    first = model.complete(base, f'{phase}.path')
    second = model.complete(base + [{'role': 'assistant', 'content': first}, {'role': 'user', 'content': P23.AUDIT}], f'{phase}.audit')
    errors = []
    for text in (second, first):
        try:
            return [repr(a) for a in P23.parse_path(text)], errors
        except Exception as exc:
            errors.append(str(exc)[:200])
    return [], errors


def judge(model, case, text_a, text_b, target_block=None, phase='judge'):
    target = target_block if target_block else f"TARGET:\n{case['target_state_pgl']}\n"
    user = (f"INITIAL:\n{case['initial_state_pgl']}\n{target}Conditions: {conditions_text(case)}\n\n"
            f"Mechanism A:\n{text_a}\n\nMechanism B:\n{text_b}\n\n{JUDGE_TASK}")
    answer = model.complete([{'role': 'system', 'content': JUDGE_SYSTEM}, {'role': 'user', 'content': user}], phase)
    found = re.findall(r'<choice>\s*([AB])\s*</choice>', answer)
    return found[-1] if found else None


def render_actions(case, actions):
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    states, _ = fm.replay(case['initial_state_pgl'], actions)
    return C.render_path(initial, states)


def sigma_signature(case, actions):
    states, _ = fm.replay(case['initial_state_pgl'], actions)
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    steps = [fm.step_from_states(b, a) for b, a in zip([initial] + states[:-1], states)]
    return tuple(sorted(sum((fm.sigma_events(s) for s in steps), collections.Counter()).items()))


def vote(case, paths):
    sigs = [sigma_signature(case, p) for p in paths]
    counts = collections.Counter(sigs)
    best = max(counts.values())
    return next(i for i, s in enumerate(sigs) if counts[s] == best)
