"""Old exposure strata, step 2: citations (Europe PMC + OpenAlex), main PDFs, and a gemini-4-argon-build screen
that reads each PDF (replaces lit150's regex count on Europe PMC XML, which author manuscripts do not have).

Strata (both citation sources must agree): high >= 100, low <= 15, mid otherwise (reserve, M3 parents).
Eligible = a proposed mechanism with >= 1 drawn substrate intermediate for one model reaction, >= 3 categories of
mechanistic experiments (lit150 categories), and no biology/medicine content.

python screen_old.py --phase download            (OpenAlex + PDFs, no LLM)
python screen_old.py --phase screen --workers 100 (argon-build; needs FALLBACK_API_KEY)
python screen_old.py --phase summary
"""
import argparse
import json
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'lit150'))
sys.path.insert(0, str(HERE.parent / 'pilots_v4'))
import select_download as SD  # noqa: E402

SD.PAPERS = HERE / 'papers'
SD.OLD = HERE / '_none'
UA = {'User-Agent': 'mech-origin-study/1.0'}
CATS = ['kie', 'radical_trap', 'radical_clock', 'light_on_off', 'quantum_yield', 'stern_volmer', 'isotope_label',
        'crossover', 'hammett', 'kinetic_order', 'hrms_trap', 'epr', 'cv', 'isolation', 'in_situ_nmr', 'control', 'dft']

SCREEN = '''The attached PDF is a chemistry paper. Screen it for a study of organic reaction mechanisms.

Answer these questions from the text AND the schemes/figures:
1. "model_reaction": one line describing the reaction whose mechanism is studied (or null).
2. "mechanism_proposed": true if the authors propose a mechanism (catalytic cycle or stepwise pathway) for it.
3. "n_substrate_intermediates": how many distinct intermediates containing substrate atoms are DRAWN as structures
   in the proposed mechanism (catalyst-only species do not count).
4. "experiments": which of these mechanistic experiment categories the paper reports (list of keys):
   {cats}
   (kie = kinetic isotope effect; radical_trap = TEMPO/BHT/DPE trapping; hrms_trap = intermediate or adduct detected
   by MS; isolation = isolated or crystallised intermediate; control = control experiments; dft = calculations).
5. "bio": true if the work involves biology, biochemistry or medicine in any way (enzymes, proteins, peptides,
   nucleic acids, cells, organisms, drugs or bioactivity, natural-product biosynthesis, bioconjugation,
   biomass, biomimetic or bio-inspired systems), with "bio_reason".
6. "reaction_class": one of "transition-metal catalysis", "photoredox", "radical (non-photo)", "organocatalysis",
   "Lewis/Brønsted acid", "pericyclic/rearrangement", "main-group/other".
Answer with ONE JSON object between [ANSWER] and [/ANSWER].'''.format(cats=', '.join(CATS))


def openalex(doi):
    try:
        req = urllib.request.Request(f'https://api.openalex.org/works/doi:{doi}', headers=UA)
        return doi, json.loads(urllib.request.urlopen(req, timeout=60).read()).get('cited_by_count')
    except Exception:
        return doi, None


def band(r):
    cs = [c for c in (r['cited_by'], r.get('openalex')) if c is not None]
    return 'high' if min(cs) >= 100 else 'low' if max(cs) <= 15 else 'mid'


def rows():
    cands = json.loads((HERE / 'candidates_old.json').read_text())
    oa = json.loads((HERE / 'openalex_cache.json').read_text()) if (HERE / 'openalex_cache.json').exists() else {}
    out = []
    for c in cands:
        r = dict(c, openalex=oa.get(c.get('doi')))
        r['stratum'] = band(r)
        out.append(r)
    return out


def download():
    cands = json.loads((HERE / 'candidates_old.json').read_text())
    cache = HERE / 'openalex_cache.json'
    oa = json.loads(cache.read_text()) if cache.exists() else {}
    with ThreadPoolExecutor(2) as ex:  # OpenAlex rate-limits unauthenticated bursts
        oa.update(dict(ex.map(openalex, [c['doi'] for c in cands if c.get('doi') and c['doi'] not in oa])))
    cache.write_text(json.dumps(oa))
    rs = rows()
    with ThreadPoolExecutor(8) as ex:
        status = dict(ex.map(SD.fetch_pdf, [r['pmcid'] for r in rs]))
    (HERE / 'pdf_status.json').write_text(json.dumps(status, indent=1))
    ok = {p for p, s in status.items() if s in ('ok', 'cached')}
    print('pdf', Counter(status.values()), 'with pdf by stratum', Counter(r['stratum'] for r in rs if r['pmcid'] in ok))


def _p29():
    import common4 as K
    import p29_gold_rebuild as P29
    P29.PAPERS = SD.PAPERS
    P29.OUTD = K.OUT / 'lit_old_argon'
    return K, P29


def job_screen(pid):
    _, P29 = _p29()
    return 'ok' if P29.call('screen', pid, SCREEN) else 'error'


def screened():
    _, P29 = _p29()
    out = {}
    for r in rows():
        f = P29.OUTD / 'screen' / f'{r["pmcid"]}.json'
        if f.exists():
            out[r['pmcid']] = P29.parse(json.loads(f.read_text())['answer']) or {}
    return out


def summary():
    sc = screened()
    rs = rows()
    sel = []
    why = Counter()
    for r in rs:
        s = sc.get(r['pmcid'])
        if s is None:
            continue
        exps = [e for e in (s.get('experiments') or []) if e in CATS]
        reasons = []
        if s.get('bio'):
            reasons.append('bio')
        if not s.get('mechanism_proposed') or not (s.get('n_substrate_intermediates') or 0) >= 1:
            reasons.append('no drawn mechanism')
        if len(exps) < 3:
            reasons.append('<3 experiment categories')
        why.update(reasons or ['eligible'])
        if not reasons:
            sel.append(dict(r, experiments=exps, n_cats=len(exps), reaction_class=s.get('reaction_class'),
                            model_reaction=s.get('model_reaction'), n_int=s.get('n_substrate_intermediates')))
    (HERE / 'selected_old.json').write_text(json.dumps(sel, indent=1, ensure_ascii=False))
    print(json.dumps({'screened': len(sc), 'outcome': why, 'eligible_by_stratum': Counter(r['stratum'] for r in sel),
                      'class_by_stratum': {b: Counter(r['reaction_class'] for r in sel if r['stratum'] == b)
                                           for b in ('high', 'mid', 'low')}}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['download', 'screen', 'summary'], required=True)
    ap.add_argument('--workers', type=int, default=100)
    cli = ap.parse_args()
    if cli.phase == 'download':
        download()
    elif cli.phase == 'screen':
        K, _ = _p29()
        status = json.loads((HERE / 'pdf_status.json').read_text())
        ids = [p for p, s in status.items() if s in ('ok', 'cached')]
        K.run_pool(job_screen, [(p,) for p in ids], cli.workers)
        summary()
    else:
        summary()
