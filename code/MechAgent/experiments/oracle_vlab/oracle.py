"""Oracle validation of VirtualLab (protocol: doc/oracle_vlab_protocol_20261101.md).

Given the authors' own mechanism, does VirtualLab reproduce the mechanistic experiments the paper reports?
Stages (each a fresh context; outcomes are never shown to stages 1-3):
  mech     paper main text -> authors' final mechanism as components + elementary steps in words (no evidence);
           read with EXTRACT_MODEL, all later stages with MODEL
  encode   mechanism JSON -> VirtualLab DSL -> MechanismHypothesis, executed with conservation checks, <=3 repairs
  map      encoded hypothesis + card (type, perturbation, observable) -> probe parameters or not mappable
  direct   baseline B1: mechanism JSON + card -> the model predicts the outcome bin itself
  predict  VirtualLab.predict per mapped card (no model call)
  analyze  coverage, accuracy vs B0 (leave-one-paper-out majority) and B1, minority-outcome accuracy
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'agent_v2'))
sys.path.insert(0, str(HERE.parent / 'evidence_p0'))
import common as C
import extract as X
from common import ROOT
from develop import metrics
from develop import standalone_chemistry as sc
from develop.virtual_lab import (CatalyticSpec, ExperimentalProbe, LabState, MechanismHypothesis, MechanisticStep,
                                 VirtualLab, AtomMeta, execute_hypothesis)

OUT = Path(os.environ.get('ORACLE_OUT', ROOT / 'outputs' / 'oracle_vlab_20261101'))
CARDS = ROOT / 'outputs' / 'evidence_p0_20261101' / 'gemini-3.5-flash-lite'
MODEL = 'gemini-3.8-flash-high'
EXTRACT_MODEL = 'gemini-3.5-flash-lite'  # reading papers/SI uses the same model as the evidence-card extraction
BASE_URL = os.environ.get('ORACLE_BASE_URL', 'http://127.0.0.1:8317/v1')
TYPES = ['kie', 'radical_trap', 'radical_clock', 'light_on_off', 'quantum_yield', 'crossover', 'isotope_label', 'hammett',
         'kinetic_order', 'nonlinear_effect', 'control', 'stern_volmer', 'spectroscopy', 'stoichiometric_intermediate',
         'stereochemical_probe', 'competition']
REPAIRS = 3
SYSTEM = 'You are an expert physical organic chemist.'


def model(stage, pid):
    if stage == 'mech':
        return C.Model(EXTRACT_MODEL, OUT / stage / 'traces' / f'{pid}.jsonl', base_url=BASE_URL, max_tokens=16384, temperature=0.0)
    return C.Model(MODEL, OUT / stage / 'traces' / f'{pid}.jsonl', base_url=BASE_URL, max_tokens=16384)


def block(answer, tag):
    m = re.search(rf'<{tag}>(.*?)</{tag}>', answer, re.S)
    if not m:
        return None
    body = re.sub(r'^```(?:json)?|```$', '', m.group(1).strip()).strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def papers():
    return sorted(p.stem for p in CARDS.glob('PMC*.json'))


def eligible_cards(pid):
    cards = json.loads((CARDS / f'{pid}.json').read_text())['cards']
    return [c for c in cards if c['experiment_type'] in TYPES and c.get('outcome_bin') not in (None, '', 'None')]


# ------------------------------------------------------------------ stage 1: mechanism
MECH_ASK = """Below is the main text of a chemistry paper. Report the mechanism the AUTHORS finally propose for the model reaction (usually drawn in their "proposed mechanism" scheme).

Strict rule: describe only the mechanism itself. Do NOT mention any experiment, result or evidence (no control experiments, radical trapping, TEMPO, KIE values, quantum yields, light on/off, Stern-Volmer, CV, EPR, yields, "consistent with", "supported by"). If the authors say a step is rate-determining or that a radical chain operates, you may state that as part of the mechanism.

Answer with JSON inside <mechanism>...</mechanism>:
{{"model_reaction": "substrate(s) -> product under standard conditions, with SMILES where you can",
  "components": [{{"name": "short name used in the paper (e.g. 4CzIPN, NiBr2·glyme, TEMPO is NOT a component)", "role": "substrate|reagent|catalyst|photocatalyst|ligand|base|acid|additive|oxidant|reductant|solvent|light", "smiles": "SMILES or null"}}],
  "steps": ["1. ...", "2. ..."],
  "rate_determining_step": "step number or null",
  "radical_chain": true/false/null,
  "notes": "anything else about the mechanism (cycles, resting state, cage vs free radicals), still without evidence"}}

MAIN TEXT:
{text}"""

LEAK = re.compile(r'TEMPO|BHT|kH/kD|\bKIE\b|quantum yield|Stern|light on|on/off|\b\d{1,3}\s?%|cyclic voltam|\bEPR\b|in the absence|without|consistent with|supported by|confirmed', re.I)


def mech_job(pid):
    dest = OUT / 'mech' / f'{pid}.json'
    if dest.exists():
        return json.loads(dest.read_text())
    pages = X.pdf_pages(X.PAPERS / pid / 'main.pdf')
    text = '\n'.join(f'[page {i + 1}]\n{p}' for i, p in enumerate(pages) if p.strip())
    answer = model('mech', pid).complete([{'role': 'system', 'content': SYSTEM},
                                          {'role': 'user', 'content': MECH_ASK.format(text=text[:250000])}], 'mech')
    mech = block(answer, 'mechanism')
    rec = {'paper_id': pid, 'mechanism': mech, 'leak_hits': sorted(set(LEAK.findall(json.dumps(mech)))) if mech else None}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec


# ------------------------------------------------------------------ stage 2: encode
DSL = """Write the mechanism as an executable hypothesis for a deterministic simulator, as JSON:
{
 "species": [{"name": "4CzIPN", "role": "photocatalyst", "atoms": {"PC1": {"el": "C"}}, "bonds": []},
             {"name": "alkyl iodide", "role": "substrate", "atoms": {"C1": {"el": "C", "H": 2}, "C2": {"el": "C", "H": 3}, "I1": {"el": "I"}}, "bonds": [["C1", "C2", 1], ["C1", "I1", 1]]}],
 "steps": [{"name": "photoexcitation", "step_class": "photoexcitation", "photons": 1, "excited_set": ["PC1"]},
           {"name": "SET reduction of R-I", "step_class": "set", "excited_clear": ["PC1"], "charge_changes": {"PC1": 1, "I1": -1},
            "bonds_broken": [["C1", "I1"]], "radical_changes": {"C1": 1, "PC1": 1}, "cage_escaped": true},
           ...],
 "catalytic_spec": {"is_catalytic": true, "photocatalyst": "4CzIPN", "tdi_index": 0, "chain_length_estimate": 1.0, "dark_thermal_initiation": false,
                    "components": {"4CzIPN": {"atoms": ["PC1"], "role": "photocatalyst"}, "alkyl iodide": {"atoms": ["C1", "C2", "I1"], "role": "substrate"}},
                    "species_redox": {"alkyl iodide": {"E_red": -1.7}}}
}
Rules (the simulator checks them and rejects violations):
- Atom labels are unique across all species. Atoms are never created or destroyed: every species that is consumed or released (sacrificial donor, base, oxidant, CO, H2O, byproducts) must be in "species" from the start. "H" is the implicit H count; any H that MOVES in a step must be an explicit atom (e.g. "H1": {"el": "H"}) bonded to its owner.
- You may abstract: replace inert parts by CH3/H, and a metal complex or organic photocatalyst by one or a few atoms (e.g. "Ni1": {"el": "Ni"}, ligands omitted). Keep every atom whose bonds, charge or radical state changes, and the atoms an experiment would label.
- A step lists only changes: bonds_formed [[a, b, order]], bonds_broken [[a, b]], bond_order_changes [[a, b, new_order]], charge_changes {atom: delta}, radical_changes {atom: delta unpaired electrons}, photons (1 for light absorption), electrode_electrons (+n cathodic, -n anodic), excited_set / excited_clear [atoms].
- Per step, the sum of formal charges changes only by -electrode_electrons; the total H (explicit + implicit) is constant; in steps without a transition metal or electrode, the parity of the total unpaired electrons is unchanged (radicals are made and consumed in pairs, e.g. SET gives a radical cation + radical anion).
- Optional per-step annotations: step_class (polar|radical|set|hat|photoexcitation|ent|oxidative_addition|reductive_elimination|migratory_insertion|...), is_rds, reversible, is_chain_propagation, is_chain_initiation, cage_escaped (true for free diffusing radicals, false for caged/metal-bound), aryl_probe_atom, ts_charge_shift.
- catalytic_spec.components maps EACH reaction component (the names given in the mechanism, plus "light" is not needed) to its atom labels and role; tdi_index = index of the resting state (state 0 = initial); species_redox optional potentials (V vs SCE: E_ox, E_red, E_T in kcal/mol).
- The steps, applied in order to the initial species, should reach the product and regenerate catalysts."""

ENCODE_ASK = """{dsl}

MECHANISM (as reported by the authors):
{mech}

Give the hypothesis JSON inside <hypothesis>...</hypothesis>."""

REPAIR_ASK = """The simulator rejected the hypothesis: {error}
Fix it and give the complete corrected JSON inside <hypothesis>...</hypothesis>."""

ROLE = {'substrate': 'reactant', 'reagent': 'reagent', 'catalyst': 'catalyst', 'photocatalyst': 'catalyst'}


def build(spec, pid):
    species, meta = [], {}
    for s in spec['species']:
        atoms = {a: sc.Atom(v['el'], Hcount=int(v.get('H', 0)), charge=int(v.get('charge', 0)) or None) for a, v in s['atoms'].items()}
        for a, v in s['atoms'].items():
            if v.get('radical'):
                meta[a] = AtomMeta(radicals=int(v['radical']))
        bonds = [sc.Bond(a, b, sc._order_kind(float(o)), order=float(o)) for a, b, o in s.get('bonds', [])]
        species.append(sc.Species(role=ROLE.get(s.get('role'), s.get('role') or 'reactant'), state='stable',
                                  molecule=sc.Molecule(name=s['name'], substructures=[sc.Substructure(name=s['name'], atomdict=atoms, bondlist=bonds)])))
    steps = []
    for k, st in enumerate(spec['steps']):
        steps.append(MechanisticStep(
            step_class=st.get('step_class', 'polar'), name=st.get('name', f'step{k + 1}'),
            bonds_formed=[(a, b, float(o)) for a, b, o in st.get('bonds_formed', [])],
            bonds_broken=[(a, b) for a, b in st.get('bonds_broken', [])],
            bond_order_changes=[(a, b, float(o)) for a, b, o in st.get('bond_order_changes', [])],
            charge_changes={a: int(d) for a, d in st.get('charge_changes', {}).items()},
            radical_changes={a: int(d) for a, d in st.get('radical_changes', {}).items()},
            excited_Set=list(st.get('excited_set', [])), excited_Clear=list(st.get('excited_clear', [])),
            photons=int(st.get('photons', 0)), electrode_electrons=int(st.get('electrode_electrons', 0)),
            reversible=bool(st.get('reversible', False)), is_rds=bool(st.get('is_rds', False)),
            is_chain_propagation=bool(st.get('is_chain_propagation', False)), is_chain_initiation=bool(st.get('is_chain_initiation', False)),
            cage_escaped=st.get('cage_escaped'), aryl_probe_atom=st.get('aryl_probe_atom'), ts_charge_shift=st.get('ts_charge_shift')))
    cs = spec.get('catalytic_spec', {})
    comps = {n: {'atoms': list(v.get('atoms', [])), 'role': v.get('role')} for n, v in cs.get('components', {}).items()}  # no 'essential'
    spec_obj = CatalyticSpec(is_catalytic=bool(cs.get('is_catalytic', False)), tdi_index=int(cs.get('tdi_index', 0)),
                             components=comps, photocatalyst=cs.get('photocatalyst'), species_redox=cs.get('species_redox') or {},
                             chain_length_estimate=float(cs.get('chain_length_estimate', 1.0)),
                             dark_thermal_initiation=bool(cs.get('dark_thermal_initiation', False)))
    init = LabState.from_pgl(sc.ReactionState(species=species), atom_meta=meta)
    return MechanismHypothesis(hypothesis_id=f'{pid}_author', name=f'{pid} author mechanism', initial_state=init, steps=steps,
                               catalytic_spec=spec_obj)


def try_build(spec, pid):
    try:
        h = build(spec, pid)
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'
    traj = execute_hypothesis(h)
    return (h, None) if traj.valid else (None, traj.error)


def encode_job(pid):
    dest = OUT / 'encode' / f'{pid}.json'
    if dest.exists():
        return json.loads(dest.read_text())
    mech = json.loads((OUT / 'mech' / f'{pid}.json').read_text())['mechanism']
    m = model('encode', pid)
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': ENCODE_ASK.format(dsl=DSL, mech=json.dumps(mech, indent=1, ensure_ascii=False))}]
    errors, spec = [], None
    for attempt in range(REPAIRS + 1):
        answer = m.complete(messages, f'encode{attempt}')
        spec = block(answer, 'hypothesis')
        if spec is None:
            err = 'no parsable JSON inside <hypothesis> tags'
        else:
            _, err = try_build(spec, pid)
        if err is None:
            break
        errors.append(err)
        messages += [{'role': 'assistant', 'content': answer}, {'role': 'user', 'content': REPAIR_ASK.format(error=err)}]
    rec = {'paper_id': pid, 'valid': err is None, 'repairs': len(errors), 'errors': errors, 'spec': spec}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec


# ------------------------------------------------------------------ stage 3: map cards to probes
MAP_ASK = """A deterministic simulator will predict the outcome of mechanistic experiments on the hypothesis below. For each experiment, give the simulator parameters, or "mappable": false if the experiment cannot be expressed on this hypothesis (e.g. a solvent or ligand screen, a substrate the hypothesis does not contain).

HYPOTHESIS (species with atom labels, components, steps):
{hyp}

Parameter conventions by experiment_type:
- control: {{"omitted_component": "<a key of components>"}} for omitting a component; {{"omitted": "light"}} for running in the dark. Replacements of one reagent by another, other solvents, temperatures, atmospheres -> mappable false (unless the replacement removes the component's function entirely).
- radical_trap: {{"reagent": "TEMPO|BHT|1,1-DPE|DMPO", "observable_mode": "yield|adduct"}}
- radical_clock: {{"clock_type": "cyclopropylcarbinyl|oxiranylcarbinyl|trans_2_phenylcyclopropylcarbinyl", "clock_atom": "<atom>"}}
- light_on_off, quantum_yield: {{}}
- stern_volmer: {{"quencher": "<a key of components>", "photocatalyst": "<name>"}}
- kie: {{"labeled_atoms": ["<carbon atom whose H is replaced by D>"], "protocol": "parallel|intermolecular_competition|intramolecular_competition"}}
- isotope_label: {{"labeled_atoms": {{"<atom>": "13C|18O|15N|2H"}}}} or {{"solvent_isotope": "2H"}} for D2O/CD3OD
- kinetic_order: {{"component": "<a key of components>"}}
- spectroscopy: {{"modality": "EPR|NMR|IR|UV-Vis|HRMS", "formula": "<Hill formula of the detected species in the hypothesis>" or "characteristic_bonds": [["a","b"]]}}
- stoichiometric_intermediate: {{"intermediate_atoms": ["<atoms of the proposed intermediate>"]}} or {{"characteristic_bonds": [["a","b"]]}}
- hammett: {{"probe_atom": "<reaction-center atom conjugated to the aryl ring>"}}
- competition: {{"edg_partner": "...", "ewg_partner": "..."}}
- crossover, nonlinear_effect, stereochemical_probe: {{}} or {{"chiral_atom": "<atom>"}}

EXPERIMENTS:
{cards}

Answer with a JSON array inside <probes>...</probes>, one object per experiment: {{"id": "...", "mappable": true/false, "parameters": {{...}}}}."""


def hyp_summary(spec):
    return json.dumps({'species': spec['species'], 'components': spec.get('catalytic_spec', {}).get('components'),
                       'steps': [s.get('name') for s in spec['steps']]}, ensure_ascii=False)


def card_view(c):
    return {'id': c['id'], 'experiment_type': c['experiment_type'], 'perturbation': c['perturbation'], 'observable': c['observable']}


def map_job(pid):
    dest = OUT / 'map' / f'{pid}.json'
    if dest.exists():
        return json.loads(dest.read_text())
    enc = json.loads((OUT / 'encode' / f'{pid}.json').read_text())
    cards = eligible_cards(pid)
    if not enc['valid'] or not cards:
        rec = {'paper_id': pid, 'probes': None}
    else:
        answer = model('map', pid).complete([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': MAP_ASK.format(
            hyp=hyp_summary(enc['spec']), cards=json.dumps([card_view(c) for c in cards], indent=1, ensure_ascii=False))}], 'map')
        rec = {'paper_id': pid, 'probes': block(answer, 'probes')}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec


# ------------------------------------------------------------------ stage B1: direct LLM prediction
DIRECT_ASK = """Assume the mechanism below is correct. For each experiment, predict the observed outcome as one bin.

MECHANISM:
{mech}

Bins by experiment_type:
{bins}

EXPERIMENTS:
{cards}

Answer with a JSON array inside <predictions>...</predictions>: {{"id": "...", "outcome_bin": "<bin word, e.g. suppressed, primary, observed, positive:1-3, product:major, S1:1>"}}."""


def direct_job(pid):
    dest = OUT / 'direct' / f'{pid}.json'
    if dest.exists():
        return json.loads(dest.read_text())
    mech = json.loads((OUT / 'mech' / f'{pid}.json').read_text())['mechanism']
    cards = eligible_cards(pid)
    answer = model('direct', pid).complete([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': DIRECT_ASK.format(
        mech=json.dumps(mech, indent=1, ensure_ascii=False), bins=json.dumps(X.SCHEMA['bins'], indent=1),
        cards=json.dumps([card_view(c) for c in cards], indent=1, ensure_ascii=False))}], 'direct')
    rec = {'paper_id': pid, 'predictions': block(answer, 'predictions')}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec


# ------------------------------------------------------------------ stage 4: VirtualLab predictions
def predict_paper(pid):
    enc = json.loads((OUT / 'encode' / f'{pid}.json').read_text())
    probes = json.loads((OUT / 'map' / f'{pid}.json').read_text())['probes'] or []
    by_id = {p.get('id'): p for p in probes if isinstance(p, dict)}
    h = build(enc['spec'], pid) if enc['valid'] else None
    lab, out = VirtualLab(), {}
    for c in eligible_cards(pid):
        p = by_id.get(c['id'])
        if h is None or not p or not p.get('mappable'):
            out[c['id']] = {'pred': None, 'why': 'not encoded' if h is None else 'not mappable'}
            continue
        params = dict(p.get('parameters') or {})
        if c['experiment_type'] == 'control' and str(params.get('omitted', '')).lower() == 'light':
            pert = 'in the dark'
        else:
            pert = c['perturbation']
        try:
            pred = lab.predict(h, ExperimentalProbe(c['experiment_type'], perturbation=pert, observable=c['observable'],
                                                    parameters=params, probe_id=c['id']))
            out[c['id']] = {'pred': pred.outcome_bin, 'why': pred.rationale[:300]}
        except Exception as exc:
            out[c['id']] = {'pred': None, 'why': f'{type(exc).__name__}: {exc}'[:300]}
    return out


# ------------------------------------------------------------------ scoring
def norm(etype, value, observable=''):
    if value is None:
        return None
    v = X.canonical_bin(value)
    if v is None:
        return None
    v = v.lower().strip()
    v = re.sub(r'^(radical_trap_adduct|radical_trap|spectroscopy|control|kie|light_on_off|quantum_yield):', '', v)
    if etype == 'radical_trap' and ';' in v:  # 'suppressed; adduct_detected' -> keep the part matching the observable
        parts = [x.strip() for x in v.split(';')]
        v = next((x for x in parts if 'adduct' in x), parts[0]) if re.search(r'adduct|hrms|ms|detect', observable, re.I) else parts[0]
    if etype in ('isotope_label', 'kinetic_order', 'competition') and ':' in v:
        v = v.split(':', 1)[1].strip()
    v = {'competent (gives product)': 'competent', 'competent': 'competent'}.get(v, v)
    return v.split()[0] if etype not in ('hammett',) else v.replace(' ', '')


def analyze(only=None):
    rows = []
    for pid in only or papers():
        cards = eligible_cards(pid)
        if not cards:
            continue
        vl = predict_paper(pid) if (OUT / 'map' / f'{pid}.json').exists() else {}
        d = json.loads((OUT / 'direct' / f'{pid}.json').read_text()) if (OUT / 'direct' / f'{pid}.json').exists() else {}
        dpred = {p.get('id'): p.get('outcome_bin') for p in (d.get('predictions') or []) if isinstance(p, dict)}
        for c in cards:
            t, obs = c['experiment_type'], norm(c['experiment_type'], c['outcome_bin'], c['observable'])
            rows.append({'paper': pid, 'id': c['id'], 'type': t, 'obs': obs,
                         'vl': norm(t, (vl.get(c['id']) or {}).get('pred'), c['observable']),
                         'vl_why': (vl.get(c['id']) or {}).get('why'),
                         'b1': norm(t, dpred.get(c['id']), c['observable']), 'perturbation': c['perturbation']})
    # B0: leave-one-paper-out majority per type
    for r in rows:
        pool = collections.Counter(x['obs'] for x in rows if x['type'] == r['type'] and x['paper'] != r['paper'])
        r['b0'] = pool.most_common(1)[0][0] if pool else None
        r['minority'] = r['obs'] != r['b0']
    enc = {pid: json.loads((OUT / 'encode' / f'{pid}.json').read_text()) for pid in papers() if (OUT / 'encode' / f'{pid}.json').exists()}
    mech = {pid: json.loads((OUT / 'mech' / f'{pid}.json').read_text()) for pid in papers() if (OUT / 'mech' / f'{pid}.json').exists()}

    def acc(sub, key):
        if not sub:
            return None
        d = [float(r[key] is not None and r[key] == r['obs']) for r in sub]
        lo, hi = metrics.clustered_bootstrap_ci(d, [r['paper'] for r in sub], seed=20261101) if len(set(r['paper'] for r in sub)) > 1 else (None, None)
        return {'n': len(sub), 'acc': round(statistics.mean(d), 4), 'ci95': [lo and round(lo, 4), hi and round(hi, 4)]}

    def paired(sub, a, b):
        d = [float(r[a] == r['obs']) - float(r[b] == r['obs']) for r in sub]
        lo, hi = metrics.clustered_bootstrap_ci(d, [r['paper'] for r in sub], seed=20261101)
        return {'delta': round(statistics.mean(d), 4), 'ci95': [round(lo, 4), round(hi, 4)]}

    covered = [r for r in rows if r['vl'] is not None]
    report = {
        'papers': len({r['paper'] for r in rows}), 'eligible_cards': len(rows),
        'mech_leak_papers': {pid: sorted({h.lower() for h in LEAK.findall(json.dumps(m['mechanism']))}) for pid, m in mech.items() if LEAK.search(json.dumps(m['mechanism']))},
        'encoded_valid': sum(e['valid'] for e in enc.values()), 'encoded_total': len(enc),
        'encode_repairs': dict(collections.Counter(e['repairs'] for e in enc.values() if e['valid'])),
        'vl_coverage': round(len(covered) / len(rows), 4) if rows else None,
        'on_covered': {k: acc(covered, k) for k in ('vl', 'b0', 'b1')},
        'on_covered_paired': {'vl-b0': paired(covered, 'vl', 'b0'), 'vl-b1': paired(covered, 'vl', 'b1')} if covered else None,
        'all_eligible': {k: acc(rows, k) for k in ('vl', 'b0', 'b1')},
        'minority_on_covered': {k: acc([r for r in covered if r['minority']], k) for k in ('vl', 'b0', 'b1')},
        'by_type_covered': {t: {k: acc([r for r in covered if r['type'] == t], k)['acc'] for k in ('vl', 'b0', 'b1')} | {'n': sum(r['type'] == t for r in covered)}
                            for t in sorted({r['type'] for r in covered})},
        'vl_pred_distribution': {t: dict(collections.Counter(r['vl'] for r in covered if r['type'] == t)) for t in sorted({r['type'] for r in covered})},
    }
    bal = {}
    for k in ('vl', 'b0', 'b1'):
        per = []
        for t in {r['type'] for r in covered}:
            sub = [r for r in covered if r['type'] == t]
            per.append(statistics.mean(statistics.mean(float(r[k] == r['obs']) for r in sub if r['obs'] == o) for o in {r['obs'] for r in sub}))
        bal[k] = round(statistics.mean(per), 4) if per else None
    report['balanced_acc_covered_macro_over_types'] = bal
    (OUT / 'rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (OUT / 'REPORT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['mech', 'encode', 'map', 'direct', 'analyze'])
    ap.add_argument('--papers', nargs='*')
    ap.add_argument('--workers', type=int, default=30)
    args = ap.parse_args()
    if args.stage == 'analyze':
        return analyze(args.papers)
    job = {'mech': mech_job, 'encode': encode_job, 'map': map_job, 'direct': direct_job}[args.stage]
    todo = args.papers or papers()
    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(job, pid): pid for pid in todo}
        for f in concurrent.futures.as_completed(futs):
            try:
                r = f.result()
                print(json.dumps({'paper': futs[f], 'ok': True, **{k: r.get(k) for k in ('valid', 'repairs', 'leak_hits') if k in r}}), flush=True)
            except Exception as exc:
                print(json.dumps({'paper': futs[f], 'error': f'{type(exc).__name__}: {exc}'[:300]}), flush=True)


if __name__ == '__main__':
    main()
