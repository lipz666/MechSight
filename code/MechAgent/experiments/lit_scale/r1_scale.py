"""Literature-memory scaling study, step 5: source recall (R1) on the eligible sample.

Inputs per paper (from extract_scale.py records):
  main   the paper's model reaction (PDF extraction for PDF papers, abstract extraction otherwise)
  swap   the same transformation on a different substrate (reaction-type control)
  cal    abstract-mode extraction of a calibration PDF paper (input-source calibration)
Models: gemini-3.8-flash (common4.call, as in the earlier probes), gemini-4-argon-build (117 gateway),
gemini-3.5-flash-lite (vps gateway). Not gemini-4-argon.
Samples: main 4 x T=1.0 for every model; swap and cal 4 x T=1.0, flash only.
Prompt and hit rule identical to p33 (DOI match, or same year and first/last author surname).

python r1_scale.py --workers 200
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pilots_v4'))
import common4 as K  # noqa: E402
import p33_m1_m4 as P33  # noqa: E402

OUT = K.OUT / 'lit_scale'
BASE = 'http://LLM-GATEWAY-1.example:8080/v1'
MODELS = {'flash': 'gemini-3.8-flash', 'argon': 'gemini-4-argon-build', 'lite': 'gemini-3.5-flash-lite',
          'flash36': 'gemini-3.6-flash'}
# gateway per directly-called model: argon-build on the 117 gateway, flash-lite on the vps gateway (both offline models)
GATEWAY = {'argon': (BASE, 'FALLBACK_API_KEY'), 'lite': ('https://LLM-GATEWAY-2.example/v1', 'OPENAI_API_KEY'),
           'flash36': (BASE, 'FALLBACK_API_KEY')}
NS = 4
_SEM = {'argon': threading.Semaphore(int(os.environ.get('ARGON_CAP', '100'))),
        'lite': threading.Semaphore(int(os.environ.get('LITE_CAP', '100'))),
        'flash36': threading.Semaphore(int(os.environ.get('FLASH36_CAP', '200')))}


def direct(model, tag, key, prompt, temperature=1.0):
    dest = OUT / tag / f'{key}.json'
    if dest.exists():
        return json.loads(dest.read_text())['answer']
    dest.parent.mkdir(parents=True, exist_ok=True)
    base, key_env = GATEWAY[model]
    client = OpenAI(api_key=os.environ[key_env], base_url=base, timeout=900, max_retries=0)
    for attempt in range(6):
        try:
            with _SEM[model]:
                r = client.chat.completions.create(model=MODELS[model], temperature=temperature, max_tokens=16384,
                                                   messages=[{'role': 'user', 'content': prompt}])
            ans = r.choices[0].message.content or ''
            if '[ANSWER]' not in ans:
                raise RuntimeError('no [ANSWER]')
            dest.write_text(json.dumps({'id': key, 'tag': tag, 'model': MODELS[model], 'returned_model': r.model,
                                        'temperature': temperature, 'prompt': prompt, 'answer': ans},
                                       ensure_ascii=False, indent=1))
            return ans
        except Exception:  # noqa: BLE001
            time.sleep(min(60, 5 * 2 ** attempt))
    return None


def prompt(rx):
    return P33.R1.format(reactants=json.dumps(rx['reactants'] + rx['catalysts']), products=json.dumps(rx['products']),
                         conditions=rx['conditions'])


def tag(model, inp, s):
    return f'lit_scale/R1_{model}_{inp}_s{s}' if model == 'flash' else f'R1_{model}_{inp}_s{s}'


def answer(model, inp, s, key):
    t = tag(model, inp, s)
    return K.cached(t, key) if model == 'flash' else (
        json.loads((OUT / t / f'{key}.json').read_text())['answer'] if (OUT / t / f'{key}.json').exists() else None)


def job(model, inp, s, key, rx):
    p = prompt(rx)
    a = K.call(tag(model, inp, s), key, p, temperature=1.0) if model == 'flash' else direct(model, tag(model, inp, s),
                                                                                             key, p)
    return 'ok' if a else 'error'


def inputs():
    """{key: {'main': rx, 'swap': rx?, 'cal': rx?}} for eligible papers."""
    sample = json.loads((HERE / 'sample.json').read_text())
    out = {}
    for p in sample['papers']:
        f = OUT / 'records' / f"{p['id']}__{p['source']}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        if not r['eligible']:
            continue
        d = {'main': r['reaction']}
        if r.get('swap'):
            d['swap'] = r['swap']
        fc = OUT / 'records' / f"{p['id']}__abstract.json"
        if p['id'] in sample['calibration'] and fc.exists():
            rc = json.loads(fc.read_text())
            if rc['eligible']:
                d['cal'] = rc['reaction']
        out[p['id']] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=200)
    ap.add_argument('--models', default='flash,argon,lite')
    ap.add_argument('--control-model', default='flash', help='model that also runs the swap and cal inputs')
    cli = ap.parse_args()
    ins = inputs()
    jobs = []
    for key, d in ins.items():
        for m in cli.models.split(','):
            jobs += [(m, 'main', s, key, d['main']) for s in range(1, NS + 1)]
        if cli.control_model in cli.models.split(','):
            for inp in ('swap', 'cal'):
                if inp in d:
                    jobs += [(cli.control_model, inp, s, key, d[inp]) for s in range(1, NS + 1)]
    print(json.dumps({'eligible': len(ins), 'jobs': len(jobs)}), flush=True)
    for _ in range(3):
        K.run_pool(job, jobs, cli.workers)
        jobs = [j for j in jobs if answer(j[0], j[1], j[2], j[3]) is None]
        if not jobs:
            break


if __name__ == '__main__':
    main()
