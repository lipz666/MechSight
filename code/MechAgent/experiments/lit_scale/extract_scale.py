"""Literature-memory scaling study, step 3: screen each sampled paper and extract its model reaction (argon-build).

Source per paper: main PDF when the PMC OA bucket has it (pdf mode), otherwise title + abstract (abstract mode).
Calibration: a fixed subset of PDF papers is extracted in BOTH modes (R1 is later run on both inputs).
Eligible = one specific organic transformation (model reaction) whose mechanism the paper studies, with parsable
reactant and product SMILES.

Also writes, for every eligible paper, a 'swap' input: the same transformation on a different substrate designed by
argon (reaction-type control: if the model only recognises the reaction class, it will still name the paper).

python extract_scale.py --workers 100
"""
import argparse
import base64
import json
import os
import sys
import threading
import time
from pathlib import Path

from openai import OpenAI
from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pilots_v4'))
sys.path.insert(0, str(HERE.parent / 'lit150'))
import common4 as K  # noqa: E402

RDLogger.DisableLog('rdApp.*')
OUT = K.OUT / 'lit_scale'
MODEL = 'gemini-4-argon-build'
BASE = 'http://LLM-GATEWAY-1.example:8080/v1'
_SEM = threading.Semaphore(int(os.environ.get('ARGON_CAP', '100')))

EXTRACT = '''{source}

Decide whether this paper reports ONE specific organic transformation (a model reaction with defined reactants and
products) and studies or proposes its mechanism. If yes, write that model reaction exactly as reported (the actual
substrates of the model/optimisation reaction, not a generic scheme; instantiate R groups with the model substrate).
Answer with ONE JSON object between [ANSWER] and [/ANSWER]:
{{"eligible": true|false, "reason": "one sentence",
  "model_reaction": {{"reactants": [SMILES], "catalysts": [SMILES of catalysts/photocatalysts/key reagents],
                      "products": [SMILES of the main product(s)], "conditions": "solvent, temperature, light, additives"}}}}'''

SWAP = '''Model reaction from a paper:
reactants: {reactants}
catalysts/reagents: {catalysts}
products: {products}
conditions: {conditions}

Write the SAME transformation (same reaction type, same catalysts, reagents and conditions) applied to a clearly
different but ordinary substrate: change the substrate skeleton and substituents (e.g. another aryl or alkyl group,
another ring size or heteroatom pattern) so that the substrate pair is NOT the one of this paper, while the reaction
would still plausibly work. Give the new reactants and the corresponding product.
Answer with ONE JSON object between [ANSWER] and [/ANSWER]:
{{"reactants": [SMILES], "catalysts": [SMILES], "products": [SMILES], "conditions": "..."}}'''


def call(tag, key, text, pdf=None):
    dest = OUT / tag / f'{key}.json'
    if dest.exists():
        return json.loads(dest.read_text())['answer']
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = [{'type': 'text', 'text': text}]
    if pdf is not None:
        content.append({'type': 'file', 'file': {'file_data': 'data:application/pdf;base64,'
                                                              + base64.b64encode(pdf.read_bytes()).decode(),
                                                 'filename': 'main.pdf'}})
    client = OpenAI(api_key=os.environ['FALLBACK_API_KEY'], base_url=BASE, timeout=900, max_retries=0)
    for attempt in range(6):
        try:
            with _SEM:
                r = client.chat.completions.create(model=MODEL, temperature=0.2, max_tokens=16384,
                                                   messages=[{'role': 'user', 'content': content}])
            ans = r.choices[0].message.content or ''
            if '[ANSWER]' not in ans:
                raise RuntimeError('no [ANSWER]')
            dest.write_text(json.dumps({'id': key, 'tag': tag, 'model': MODEL, 'returned_model': r.model,
                                        'prompt': text, 'answer': ans}, ensure_ascii=False, indent=1))
            return ans
        except Exception:  # noqa: BLE001
            time.sleep(min(60, 5 * 2 ** attempt))
    return None


def parse(ans):
    import re
    m = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', ans or '', re.S)
    body = re.sub(r'```(?:json)?', '', m.group(1) if m else '')
    lo, hi = body.find('{'), body.rfind('}')
    try:
        return json.loads(re.sub(r',\s*([}\]])', r'\1', body[lo:hi + 1])) if lo >= 0 else None
    except Exception:  # noqa: BLE001
        return None


def _ok(lst):
    lst = [s for s in (lst or []) if isinstance(s, str) and s.strip()]
    return [Chem.MolToSmiles(m) for s in lst if (m := Chem.MolFromSmiles(s)) is not None]


def clean_reaction(mr):
    mr = mr or {}
    reac, prod = _ok(mr.get('reactants')), _ok(mr.get('products'))
    if not reac or not prod or set(prod) <= set(reac):
        return None
    return {'reactants': reac, 'catalysts': _ok(mr.get('catalysts')), 'products': prod,
            'conditions': mr.get('conditions') or ''}


def source_text(p, mode):
    if mode == 'pdf':
        return 'The attached PDF is a chemistry paper.'
    return f'Title: {p["title"]}\nAbstract: {p["abstract"]}\n(Only the title and abstract are available.)'


def job(p, mode):
    key = p['id']
    pdf = HERE / 'papers' / (p.get('pmcid') or '_') / 'main.pdf'
    if mode == 'pdf' and not pdf.exists():
        return 'no pdf'
    a = call(f'extract_{mode}', key, EXTRACT.format(source=source_text(p, mode)), pdf if mode == 'pdf' else None)
    d = parse(a)
    if not d:
        return 'error' if a is None else 'unparsed'
    mr = clean_reaction(d.get('model_reaction')) if d.get('eligible') else None
    rec = {'id': key, 'mode': mode, 'eligible': bool(mr), 'reason': d.get('reason'), 'reaction': mr}
    if mr and mode == ('pdf' if p['source'] == 'pdf' else 'abstract'):
        s = parse(call('swap', key, SWAP.format(**{k: json.dumps(v) if isinstance(v, list) else v
                                                   for k, v in mr.items()})))
        sw = clean_reaction(s)
        if sw and set(sw['reactants']) != set(mr['reactants']):
            rec['swap'] = sw
    (OUT / 'records').mkdir(parents=True, exist_ok=True)
    (OUT / 'records' / f'{key}__{mode}.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return 'eligible' if mr else 'ineligible'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=100)
    cli = ap.parse_args()
    sample = json.loads((HERE / 'sample.json').read_text())
    jobs = [(p, p['source']) for p in sample['papers']]
    jobs += [(p, 'abstract') for p in sample['papers'] if p['id'] in set(sample['calibration'])]
    K.run_pool(job, jobs, cli.workers)


if __name__ == '__main__':
    main()
