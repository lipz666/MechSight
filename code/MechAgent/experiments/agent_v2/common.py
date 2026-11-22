"""Shared pieces for the J1-J4 experiments (MechAgent-v2).

* Model: OpenAI-compatible streaming client with retries and usage logging.
  The API key is read from OPENAI_API_KEY only and never written anywhere.
* GT paths: FukuyamaBench annotations record states, not arrows. The GT path
  is rebuilt by applying each step's recorded changes to the cumulative
  initial state, so GT and model paths live on the same atoms and labels.
* render_path: one text format for any path (GT or model), so a judge cannot
  tell them apart by formatting.
"""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(ROOT))
from develop import fukuyama_metrics as fm
from develop import standalone_chemistry as chem

DEFAULT_BASE_URL = 'http://LLM-GATEWAY-2.example/v1'
MODELS = ['gemini-4-argon-build', 'gemini-3.8-flash-build']
KIND_ORDER = {'single': 1.0, 'double': 2.0, 'triple': 3.0, 'aromatic': 1.5}


def generation_tokens(usage):
    usage = usage or {}
    completion = int(usage.get('completion_tokens') or 0)
    total = int(usage.get('total_tokens') or 0)
    prompt = int(usage.get('prompt_tokens') or 0)
    reasoning = int((usage.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0)
    return max(completion, total - prompt, reasoning)


class Model:
    def __init__(self, model, trace, base_url=DEFAULT_BASE_URL, timeout=600, max_tokens=8192, temperature=0.2):
        from openai import OpenAI
        self.model, self.trace, self.max_tokens, self.temperature = model, Path(trace), max_tokens, temperature
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=base_url, timeout=timeout, max_retries=0)
        self.events = []

    def complete(self, messages, phase):
        start = time.monotonic()
        for attempt in range(10):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=self.max_tokens, temperature=self.temperature,
                    reasoning_effort='high', stream=True, stream_options={'include_usage': True})
                parts, usage, returned = [], None, None
                for chunk in stream:
                    returned = chunk.model
                    if chunk.usage:
                        usage = chunk.usage.model_dump()
                    if chunk.choices and chunk.choices[0].delta.content:
                        parts.append(chunk.choices[0].delta.content)
                answer = ''.join(parts)
                if not answer.strip():
                    raise RuntimeError('empty answer')
                break
            except Exception as exc:
                status = getattr(exc, 'status_code', None)
                self._save({'phase': phase, 'api_error': type(exc).__name__, 'status': status, 'retry': attempt,
                            'detail': str(exc).replace(os.environ['OPENAI_API_KEY'], '[REDACTED]')[:400],
                            'elapsed_seconds': round(time.monotonic() - start, 1)})
                if attempt == 9 or (status and status not in {408, 409, 425, 429, 500, 502, 503, 504}):
                    raise
                time.sleep(min(60, 5 * 2 ** attempt))
        event = {'phase': phase, 'model': self.model, 'returned_model': returned, 'usage': usage,
                 'generation_tokens': generation_tokens(usage), 'elapsed_seconds': round(time.monotonic() - start, 1),
                 'answer': answer}
        self.events.append(event)
        self._save(event)
        return answer

    def _save(self, event):
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        with self.trace.open('a') as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + '\n')

    @property
    def tokens(self):
        return sum(e['generation_tokens'] for e in self.events)


# ----------------------------------------------------------------- GT paths
def gt_states(case):
    """States after each annotated step, on the cumulative initial state."""
    state = chem.parse_pgl_literal(case['initial_state_pgl'])
    out = []
    for step in case['gt_step_changes']:
        state = copy.deepcopy(state)
        atoms = dict(chem.pgl.iter_atoms(state))
        locations = chem._bond_locations(state)
        for a, b, kind in step['bonds_formed']:
            chem._find_substructure(state, a).bondlist.append(chem.Bond(a, b, kind, order=KIND_ORDER.get(kind, 1.0)))
        for a, b, _ in step['bonds_broken']:
            sub, bond = locations[chem._bond_key(a, b)]
            sub.bondlist.remove(bond)
        locations = chem._bond_locations(state)
        for a, b, _, new in step['bond_order_changes']:
            _, bond = locations[chem._bond_key(a, b)]
            bond.order, bond.kind = KIND_ORDER.get(new, 1.0), new
        for name, _, new in step['charge_changes']:
            atoms[name].charge = new
        out.append(state)
    return out


def gt_path_consistent(case):
    try:
        states = gt_states(case)
    except Exception:
        return False
    gold = [chem.parse_pgl_literal(s) for s in case['states_pgl']]
    return fm.intermediates_in_order(states, gold)[0]


# ----------------------------------------------------------------- rendering
def _molecule_smiles(state, members):
    from rdkit import Chem
    atoms = dict(chem.pgl.iter_atoms(state))
    rw = Chem.RWMol()
    index = {}
    for n in sorted(members):
        a = Chem.Atom(atoms[n].element)
        a.SetFormalCharge(int(atoms[n].charge or 0))
        a.SetNumExplicitHs(int(atoms[n].Hcount or 0))
        a.SetNoImplicit(True)
        index[n] = rw.AddAtom(a)
    kinds = {1.0: Chem.BondType.SINGLE, 2.0: Chem.BondType.DOUBLE, 3.0: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}
    for (x, y), bond in chem.pgl.bond_map(state).items():
        if x in index and y in index:
            rw.AddBond(index[x], index[y], kinds.get(float(bond.order or 1.0), Chem.BondType.SINGLE))
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
    except Exception:
        mol.UpdatePropertyCache(strict=False)
        mol = Chem.RemoveHs(mol, sanitize=False)
    return Chem.MolToSmiles(mol)


def _components(state):
    atoms = dict(chem.pgl.iter_atoms(state))
    adj = {n: set() for n in atoms}
    for (x, y) in chem.pgl.bond_map(state):
        adj[x].add(y)
        adj[y].add(x)
    seen, comps = set(), []
    for n in sorted(atoms):
        if n in seen:
            continue
        stack, comp = [n], set()
        while stack:
            m = stack.pop()
            if m not in comp:
                comp.add(m)
                stack.extend(adj[m] - comp)
        seen |= comp
        comps.append(comp)
    return comps


def step_text(before, after):
    sig_b, sig_a = chem.state_signature(before), chem.state_signature(after)
    bb, ba = sig_b['bonds'], sig_a['bonds']
    parts = []
    for key in sorted(set(ba) - set(bb)):
        parts.append('form ' + key.replace('|', '-'))
    for key in sorted(set(bb) - set(ba)):
        parts.append('break ' + key.replace('|', '-'))
    for key in sorted(set(ba) & set(bb)):
        if bb[key]['order'] != ba[key]['order']:
            parts.append(f"{key.replace('|', '-')} order {bb[key]['order']:g}->{ba[key]['order']:g}")
    for name in sorted(set(sig_a['atoms']) & set(sig_b['atoms'])):
        if sig_a['atoms'][name]['charge'] != sig_b['atoms'][name]['charge']:
            parts.append(f"charge {name} {sig_b['atoms'][name]['charge']:+d}->{sig_a['atoms'][name]['charge']:+d}")
    touched = {x for p in parts for x in p.replace('form ', '').replace('break ', '').replace('charge ', '').split(' ')[0].split('-')}
    changed = [c for c in _components(after) if c & touched]
    species = ', '.join(_molecule_smiles(after, c) for c in changed)
    return '; '.join(parts), species


def render_path(initial, states):
    lines, prev = [], initial
    for k, st in enumerate(states, 1):
        changes, species = step_text(prev, st)
        if not changes:
            continue
        lines.append(f'  Step {len(lines) + 1}: {changes}\n    species formed/changed: {species}')
        prev = st
    return '\n'.join(lines) if lines else '  (no change)'
