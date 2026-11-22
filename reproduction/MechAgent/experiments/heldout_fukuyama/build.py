"""Build the held-out FukuyamaBench test set for the frozen shortlist method.

Nothing here calls a model or reads any result of the frozen method.

Recovery
  * broken-mapping cases: 97% of broken links are explicit hydrogens that
    carry an atom-map number in a product but not in the reactants. The
    converter maps heavy atoms only and tracks hydrogens with its proton
    ledger, so the hydrogen map numbers are removed (in a copy of the source;
    RMR is not modified) and the case is converted again.
  * low-confidence cases (worst per-step mapper confidence < 0.5): kept only
    if they pass the GT sanity checks below, which do not use the confidence.

GT sanity (applied to every candidate, recovered or not)
  * converter quality: no broken link, no unsourced proton, no unmapped
    heavy atom, complete initial state, no zero-change step;
  * every annotated state sanitizes in RDKit (valences);
  * no step has more sigma-bond changes than the most seen in a step of the
    high-confidence clean set; a mis-permuted mapping shows up as extra
    form/break pairs.

Exclusions
  * the 24 dev cases and their siblings (same number, other letter);
  * every memory case that was shown as an analog in a dev prompt;
  * any case with an endpoint Morgan Tanimoto >= 0.7 to a dev case endpoint.
"""
from __future__ import annotations

import collections
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'p23_planning_probe'))
from develop import fukuyama_metrics as fm
from develop import fukuyama_to_pgl as conv
from develop import standalone_chemistry as chem
from rdkit import Chem, DataStructs, RDLogger

import analogs as AN

RDLogger.DisableLog('rdApp.*')
SRC = ROOT.parent / 'RMR' / 'fukuyama_bench' / 'mechanisms'
OUT = ROOT / 'outputs' / 'heldout_fukuyama_20261024'
DEV = ROOT / 'outputs' / 'p23_planning_probe_20261024'
H_MAP = re.compile(r'\[(H[+-]?\d*):\d+\]')


def strip_hydrogen_maps(case_dir: Path, dest_root: Path) -> Path:
    dest = dest_root / case_dir.name
    shutil.copytree(case_dir, dest, dirs_exist_ok=True)
    data = json.loads((dest / 'mapped_rxn.json').read_text())
    for step in data['mechanism_rxn']:
        if step.get('mapped_rxn'):
            step['mapped_rxn'] = H_MAP.sub(r'[\1]', step['mapped_rxn'])
    (dest / 'mapped_rxn.json').write_text(json.dumps(data, indent=2))
    return dest


def convert(case_dir: Path):
    record = conv.convert_case(case_dir)
    record['conditions'] = conv._read_conditions(case_dir)
    return record


def sigma_per_step(record):
    return [len(s['bonds_formed']) + len(s['bonds_broken']) for s in record['gt_step_changes']]


def states_sanitize(record):
    for text in record['states_pgl']:
        mol = AN.to_mol(chem.parse_pgl_literal(text))
        try:
            Chem.SanitizeMol(Chem.Mol(mol))
        except Exception:
            return False
    return True


def quality_problems(record, max_sigma):
    q = record['quality']
    problems = []
    if q['broken_map_links']:
        problems.append('broken_links')
    if q['unsourced_hydrogens']:
        problems.append('unsourced_H')
    if q['unmapped_heavy_atoms']:
        problems.append('unmapped_heavy')
    if not q['initial_state_complete']:
        problems.append('initial_incomplete')
    if q['steps_without_change']:
        problems.append('zero_change_step')
    if problems:
        return problems
    if not states_sanitize(record):
        problems.append('state_fails_valence')
    if max(sigma_per_step(record), default=0) > max_sigma:
        problems.append('step_exceeds_clean_max_sigma')
    return problems


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = {r['case_id']: r for r in map(json.loads, (ROOT / 'outputs/fukuyama/all.jsonl').read_text().splitlines())}
    clean = {r['case_id']: r for r in map(json.loads, (ROOT / 'outputs/fukuyama/clean.jsonl').read_text().splitlines())}
    dev = set(json.loads((DEV / 'cases.json').read_text()))
    shown = {a['case_id'] for v in json.loads((DEV / 'analogs.json').read_text())['cases'].values() for a in v['analogs']}
    max_sigma = max(max(sigma_per_step(r)) for r in clean.values())

    repaired_root = OUT / 'repaired_sources'
    records, log = {}, collections.Counter()
    source_dirs = sorted(p for p in SRC.iterdir() if p.is_dir())
    for case_dir in source_dirs:
        cid = case_dir.name
        rec = original.get(cid)
        route = 'original'
        if rec is None:
            log['source_not_convertible_originally'] += 1
        if rec is None or rec['quality']['broken_map_links']:
            try:
                rec = convert(strip_hydrogen_maps(case_dir, repaired_root))
                route = 'repaired_H_maps'
            except Exception as exc:
                log['repair_failed:' + str(exc)[:50]] += 1
                continue
        rec['_route'] = route
        records[cid] = rec

    dev_fps = []
    for cid in dev:
        c = clean[cid]
        dev_fps.append((AN.fingerprint(chem.parse_pgl_literal(c['initial_state_pgl'])),
                        AN.fingerprint(chem.parse_pgl_literal(c['target_state_pgl']))))
    dev_numbers = {re.sub(r'_[a-z]$', '', c) for c in dev}

    table, kept = [], []
    for cid, rec in sorted(records.items()):
        row = {'case_id': cid, 'partition': rec['partition'], 'route': rec['_route'],
               'was_clean': cid in clean, 'min_conf': rec['quality']['mapping_confidence_min'],
               'steps': rec['num_steps'], 'exclusion': None, 'problems': []}
        if cid in dev:
            row['exclusion'] = 'dev_case'
        elif re.sub(r'_[a-z]$', '', cid) in dev_numbers:
            row['exclusion'] = 'dev_sibling'
        elif cid in shown:
            row['exclusion'] = 'shown_as_dev_analog'
        else:
            try:
                fi = AN.fingerprint(chem.parse_pgl_literal(rec['initial_state_pgl']))
                ft = AN.fingerprint(chem.parse_pgl_literal(rec['target_state_pgl']))
                sim = max(DataStructs.TanimotoSimilarity(x, y) for pair in dev_fps
                          for x, y in zip((fi, ft), pair) if x is not None and y is not None)
            except Exception:
                sim = 0.0
            row['max_tanimoto_to_dev'] = round(sim, 3)
            if sim >= 0.7:
                row['exclusion'] = 'near_duplicate_of_dev'
        if row['exclusion'] is None:
            row['problems'] = quality_problems(rec, max_sigma)
            if not row['problems']:
                transient = fm.sigma_scores([], fm.gold_steps_from_clean(rec))['gold_transient']
                row['transient_bonds'] = len(transient)
                kept.append(rec)
        table.append(row)

    for rec in kept:
        rec.pop('_route', None)
    (OUT / 'heldout.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept))
    (OUT / 'build_table.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in table))

    def count(rows, key):
        return dict(collections.Counter(key(r) for r in rows))
    usable = [r for r in table if r['exclusion'] is None and not r['problems']]
    summary = {
        'source_cases': len(source_dirs), 'converted': len(records), 'conversion_log': dict(log),
        'clean_max_sigma_changes_per_step': max_sigma,
        'exclusions': count([r for r in table if r['exclusion']], lambda r: r['exclusion']),
        'quality_failures_first_reason': count([r for r in table if r['exclusion'] is None and r['problems']], lambda r: r['problems'][0]),
        'usable_total': len(usable),
        'usable_by_origin': count(usable, lambda r: 'clean_nondev (already scored in historical runs and enum recall check)' if r['was_clean']
                                  else ('recovered_low_confidence' if r['route'] == 'original' else 'recovered_broken_mapping')),
        'usable_fresh_by_partition': count([r for r in usable if not r['was_clean']], lambda r: r['partition']),
        'usable_fresh_with_transient': sum(1 for r in usable if not r['was_clean'] and r.get('transient_bonds')),
        'usable_clean_nondev_with_transient': sum(1 for r in usable if r['was_clean'] and r.get('transient_bonds')),
        'usable_fresh_min_conf_bins': count([r for r in usable if not r['was_clean']],
                                           lambda r: '>=0.5' if (r['min_conf'] or 0) >= 0.5 else '0.3-0.5' if (r['min_conf'] or 0) >= 0.3 else '<0.3'),
    }
    (OUT / 'BUILD_SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
