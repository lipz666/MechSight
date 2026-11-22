"""LLM pre-labels for the class task. The reviewer sees them only on request and confirms or overrides."""
from __future__ import annotations

import concurrent.futures
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402
from build import C, P, chem  # noqa: E402

CLASSES = [
    'carbocation / 1,2-shift / skeletal rearrangement',
    'carbanion / enolate / organometallic adduct',
    'onium ion or ylide (oxonium, ammonium, sulfonium, phosphonium, halonium, aziridinium, episulfonium)',
    'carbene / carbenoid / ketene / nitrene',
    'tetrahedral or addition adduct',
    'neighbouring-group participation / bridged intermediate',
    'cyclic transient adduct (e.g. ozonide, oxaphosphetane, cycloadduct before retro-reaction)',
    'metal-bound intermediate (oxidative addition, insertion, metallacycle)',
    'other',
]
TASK = ('Reaction (SMILES): {r} >> {p}\nConditions: {cond}\n'
        'Reference mechanism, species after each step:\n{steps}\n\n'
        'In this mechanism a bond between atoms {pair} forms and later breaks again, so it is invisible in the overall '
        'equation. The key species carrying it is: {smi}\n\n'
        'Classify this hidden intermediate. Choose one or more of:\n{classes}\n'
        'Answer with JSON inside <label></label>: {{"classes": [...], "note": "one short sentence"}}.')


def steps_text(case):
    lines = []
    for k, st in enumerate(C.gt_states(case), 1):
        lines.append(f'{k}. ' + ' . '.join(C._molecule_smiles(st, c) for c in C._components(st)))
    return '\n'.join(lines)


def label(case, card):
    model = C.Model(P.MODEL, P.OUT / 'prelabel' / 'traces' / f"{case['case_id']}.jsonl", base_url=P.BASE_URL, max_tokens=P.MAX_TOKENS)
    ep = B.endpoints(case)
    prompt = TASK.format(r=ep['reactants'], p=ep['products'], cond=ep['conditions'], steps=steps_text(case), pair=card['pair'],
                         smi=card['species_smiles'], classes='\n'.join(f'- {c}' for c in CLASSES))
    answer = model.complete([{'role': 'user', 'content': prompt}], 'prelabel')
    m = re.search(r'<label>(.*?)</label>', answer, re.S)
    try:
        return json.loads(m.group(1)) if m else None
    except json.JSONDecodeError:
        return None


def main():
    import j3_dev
    cases = j3_dev.load_cases()
    out_path = HERE / 'prelabels.json'
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    jobs = []
    for cid, case in cases.items():
        for t, card in B.hidden_cards(case).items():
            item_id = f"{cid}:{'-'.join(t)}"
            if out.get(item_id) is None:
                jobs.append((item_id, case, card))
    with concurrent.futures.ThreadPoolExecutor(100) as pool:
        futs = {pool.submit(label, case, card): item_id for item_id, case, card in jobs}
        for fut in concurrent.futures.as_completed(futs):
            try:
                out[futs[fut]] = fut.result()
            except Exception as exc:
                print('error', futs[fut], type(exc).__name__, flush=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(len(out), 'labels;', sum(v is None for v in out.values()), 'unparsed')


if __name__ == '__main__':
    main()
