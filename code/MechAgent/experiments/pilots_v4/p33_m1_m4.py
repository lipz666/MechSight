"""P33: M1 (does the model remember mechanism papers?) and M4 (evidence vs literature), plus the internet check.

Papers: the M2 set (244 in-scope old papers, three exposure strata; 165 recent lit172 papers as negative control).
Pre-registered in manuscript/机理能力来源_预注册_v1.md (addendum M1/M4, 2026-11-06).

  net    20 recent papers: "exact title of PMCID X?"; >= 1 correct title would mean live web access.
  R1     reaction (reactants, products, conditions) -> up to 5 papers reporting/studying it; hit = DOI match, or
         same year AND first- or last-author surname match. 2 samples, T = 1.0. Listed DOIs checked on Crossref.
  A / C  official prompt (with products), T = 1.0, 2 samples; C adds the citation (authors, title, journal, year) and
         "write the mechanism this paper proposed". Memory index = recall(C) - recall(A), strict gold keys.
  M4     papers whose gold mechanism has an organic radical substrate intermediate; official prompt + fixed
         observation set pointing to a radical path (consistent) or to a polar path (contradictory), 2 samples,
         T = 0.7. Outcome (program): mechanism without any organic radical intermediate = polar.

python p33_m1_m4.py --workers 200
python p33_m1_m4.py --analyze-only
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import random
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rdkit import Chem, RDLogger

import common4 as K
import p30_lit172_gold as P30
import p8_observations as P8

RDLogger.DisableLog('rdApp.*')
LIT_OLD = K.HERE.parent / 'lit_old'
SCOPE = json.loads((LIT_OLD / 'gold_scope_old.json').read_text())
META_OLD = {p['pmcid']: p for p in SCOPE}
LIT172_IDS = list(P30.IDS)
GOLD_OLD, GOLD_NEW = K.OUT / 'lit_old_argon', K.OUT / 'lit_gold_argon'
DS_OLD, DS_NEW = 'litoldg', 'lit172g'
RUN = K.OUT / 'm1_m4_20261106'
UA = {'User-Agent': 'mech-origin-study/1.0'}
SEED, NS = 20261106, 2
METALS = P30.GOLD.METALS

R1 = '''Below is a chemical reaction.

Reactants and reagents (SMILES): {reactants}
Products (SMILES): {products}
Conditions: {conditions}

Which published papers report this reaction (this exact transformation, or its model reaction) and study its
mechanism? List up to 5 candidates you actually remember, most likely first. Do not invent references; if you do not
know any, return an empty list. Answer with ONE JSON list between [ANSWER] and [/ANSWER], items
{{"first_author": surname, "last_author": surname, "journal": "...", "year": int, "title": "...", "doi": "..." or null}}.'''

CITE = ('\n\n## Literature source\n\nThis reaction was studied in: {cite}\n\nWrite the mechanism that this paper '
        'proposed.\n\n## Input')

NET = '''What is the exact title of the journal article with PubMed Central identifier {pmcid}? If you do not know,
answer UNKNOWN. Answer with the title only between [ANSWER] and [/ANSWER].'''

OBS_RADICAL = ('1) Adding TEMPO (2.0 equiv) shut down product formation completely, and a TEMPO adduct of a '
               'substrate-derived fragment was detected by HRMS. 2) A cyclopropyl-containing radical-clock analogue '
               'of the substrate gave only ring-opened products. 3) EPR spectroscopy of the reaction mixture showed a '
               'strong organic radical signal.')
OBS_POLAR = ('1) Adding TEMPO (2.0 equiv) or BHT had no effect on yield or rate, and no TEMPO adduct was detected by '
             'HRMS. 2) A cyclopropyl-containing radical-clock analogue of the substrate gave the product with the '
             'cyclopropane fully intact (no ring-opened products). 3) EPR spectroscopy of the reaction mixture showed '
             'no radical signal. 4) Hammett study: rho = -4.1, indicating substantial positive charge build-up in the '
             'rate-determining step.')


# ---------------------------------------------------------------- metadata

def _epmc(pmcid):
    q = urllib.parse.quote(f'PMCID:{pmcid}')
    url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&format=json&resultType=core'
    for attempt in range(5):
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())
            res = (r.get('resultList') or {}).get('result') or []
            if not res:
                return pmcid, None
            x = res[0]
            auth = [a.get('lastName') or a.get('fullName', '').split()[-1]
                    for a in (x.get('authorList') or {}).get('author', []) if a.get('lastName') or a.get('fullName')]
            return pmcid, {'title': x.get('title'), 'journal': (x.get('journalInfo') or {}).get('journal', {}).get('title')
                           or x.get('journalTitle'), 'year': int(x.get('pubYear') or 0), 'doi': x.get('doi'),
                           'authorString': x.get('authorString'), 'authors': auth}
        except Exception:  # noqa: BLE001
            time.sleep(3 + 5 * attempt)
    return pmcid, None


def metadata(ids):
    path = RUN / 'meta.json'
    meta = json.loads(path.read_text()) if path.exists() else {}
    todo = [i for i in ids if not meta.get(i)]
    with ThreadPoolExecutor(4) as ex:
        meta.update(dict(ex.map(_epmc, todo)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return meta


def citation(m):
    return f"{m['authorString']} {m['title']} {m['journal']} ({m['year']})."


# ---------------------------------------------------------------- cases

def all_cases():
    P30.DS, P30.IDS, P30.GOLD.OUTD = DS_OLD, [p['pmcid'] for p in SCOPE], GOLD_OLD
    old = P30.cases()
    P30.DS, P30.IDS, P30.GOLD.OUTD = DS_NEW, LIT172_IDS, GOLD_NEW
    new = P30.cases()
    out = {}
    for cid, c in old.items():
        out[cid] = dict(c, stratum=META_OLD[cid]['exposure'], gold_dir=str(GOLD_OLD), ds=DS_OLD,
                        log_cites=math.log10(1 + META_OLD[cid]['cites_gm']), year=META_OLD[cid]['year'])
    for cid, c in new.items():
        out[cid] = dict(c, stratum='recent', gold_dir=str(GOLD_NEW), ds=DS_NEW)
    return out


def radical_gold(c):
    for s in c['author']:
        m = Chem.MolFromSmiles(s)
        if m and any(a.GetNumRadicalElectrons() and a.GetSymbol() not in METALS for a in m.GetAtoms()):
            return True
    return False


# ---------------------------------------------------------------- jobs

def job_net(cid):
    return 'ok' if K.call('m1/net', cid, NET.format(pmcid=cid), temperature=0.2) else 'error'


def job_r1(c, s):
    p = R1.format(reactants=json.dumps(c['reactants']), products=json.dumps(c['products']), conditions=c['conditions'])
    return 'ok' if K.call(f'm1/R1_s{s}', c['id'], p, temperature=1.0) else 'error'


def job_a(c, s):
    return 'ok' if K.call(f'm1/A_s{s}', c['id'], K.official_prompt(c), temperature=1.0) else 'error'


def job_c(c, s, meta):
    p = K.official_prompt(c).replace('\n\n## Input', CITE.format(cite=citation(meta[c['id']])), 1)
    return 'ok' if K.call(f'm1/C_s{s}', c['id'], p, temperature=1.0) else 'error'


def job_m4(c, arm, s):
    obs = OBS_RADICAL if arm == 'consistent' else OBS_POLAR
    p = K.official_prompt(c).replace('\n\n## Input', P8.OBS.format(obs=obs), 1)
    return 'ok' if K.call(f'm4/{arm}_s{s}', c['id'], p, temperature=0.7) else 'error'


# ---------------------------------------------------------------- scoring

def _norm(s):
    s = unicodedata.normalize('NFKD', str(s or '')).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z]', '', s)


def _refs(ans):
    m = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', ans or '', re.S)
    body = re.sub(r'```(?:json)?', '', m.group(1) if m else '')
    lo, hi = body.find('['), body.rfind(']')
    try:
        out = json.loads(body[lo:hi + 1]) if lo >= 0 else []
    except Exception:  # noqa: BLE001
        return []
    return [r for r in out if isinstance(r, dict)] if isinstance(out, list) else []


def r1_hit(refs, meta):
    doi = (meta.get('doi') or '').lower()
    names = {_norm(a) for a in (meta['authors'][:1] + meta['authors'][-1:])}
    for r in refs:
        if doi and (r.get('doi') or '').lower().strip().removeprefix('https://doi.org/') == doi:
            return 1.0
        try:
            yr = int(r.get('year') or 0)
        except (TypeError, ValueError):
            yr = 0
        if yr == meta['year'] and ({_norm(r.get('first_author')), _norm(r.get('last_author'))} & names) - {''}:
            return 1.0
    return 0.0


def crossref(doi):
    url = 'https://api.crossref.org/works/' + urllib.parse.quote(doi)
    for attempt in range(4):
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read())
            return doi, (r['message'].get('title') or [''])[0]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return doi, False
            time.sleep(2 + 3 * attempt)
        except Exception:  # noqa: BLE001
            time.sleep(2 + 3 * attempt)
    return doi, None


def check_dois(cs):
    path = RUN / 'crossref.json'
    cache = json.loads(path.read_text()) if path.exists() else {}
    dois = {(r.get('doi') or '').strip().removeprefix('https://doi.org/') for c in cs.values() for s in range(1, NS + 1)
            for r in _refs(K.cached(f'm1/R1_s{s}', c['id']))}
    todo = [d for d in dois if d and '/' in d and d not in cache]
    with ThreadPoolExecutor(6) as ex:
        cache.update(dict(ex.map(crossref, todo)))
    path.write_text(json.dumps(cache, indent=1))
    return cache


def recall(c, ans):
    m = K.parse_mech(ans)
    ps = P30.V.keys_of(m)
    ks = [k for k in (K.species_keys(s) for s in c['author']) if k]
    return sum(bool(k & ps) for k in ks) / len(ks) if ks else 0.0


def is_radical_mech(ans):
    m = K.parse_mech(ans)
    if not m:
        return None
    for st in m:
        mol = Chem.MolFromSmiles(st.get('intermediate_smiles') or '')
        if mol and any(a.GetNumRadicalElectrons() and a.GetSymbol() not in METALS for a in mol.GetAtoms()):
            return True
    return False


def boot(rows, fn, n=2000):
    rng = random.Random(SEED)
    try:
        est = fn(rows)
    except Exception:  # noqa: BLE001
        return None
    bs = []
    for _ in range(n):
        try:
            bs.append(fn([rows[rng.randrange(len(rows))] for _ in rows]))
        except Exception:  # noqa: BLE001
            pass
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [round(100 * est, 2), round(100 * lo, 2), round(100 * hi, 2), len(rows)]


def mean_of(k):
    return lambda rs: float(np.mean([r[k] for r in rs if r.get(k) is not None]))


def slope_of(k):
    def fn(rs):
        rs = [r for r in rs if r['stratum'] != 'recent' and r.get(k) is not None]
        X = np.array([[1.0, r['log_cites'], r['year']] for r in rs])
        return float(np.linalg.lstsq(X, np.array([r[k] for r in rs]), rcond=None)[0][1])
    return fn


def analyze(cs, meta):
    net = []
    for cid in NET_IDS(cs):
        a = K.cached('m1/net', cid) or ''
        m = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', a, re.S)
        t = (m.group(1) if m else a).strip()
        net.append(difflib.SequenceMatcher(None, _norm(t), _norm(meta[cid]['title'])).ratio() >= 0.6)
    xref = check_dois(cs)
    rows = []
    listed = verified = 0
    for cid, c in cs.items():
        if not meta.get(cid):
            continue
        r = {'id': cid, 'stratum': c['stratum'], 'log_cites': c.get('log_cites'), 'year': c.get('year')}
        refs = [_refs(K.cached(f'm1/R1_s{s}', cid)) for s in range(1, NS + 1)]
        r['R1'] = float(np.mean([r1_hit(x, meta[cid]) for x in refs]))
        r['n_refs'] = float(np.mean([len(x) for x in refs]))
        for x in refs:
            for ref in x:
                d = (ref.get('doi') or '').strip().removeprefix('https://doi.org/')
                if d and d in xref and xref[d] is not None:
                    listed += 1
                    verified += bool(xref[d])
        r['A'] = float(np.mean([recall(c, K.cached(f'm1/A_s{s}', cid)) for s in range(1, NS + 1)]))
        r['C'] = float(np.mean([recall(c, K.cached(f'm1/C_s{s}', cid)) for s in range(1, NS + 1)]))
        r['CminusA'] = r['C'] - r['A']
        r['Direct'] = recall(c, K.cached(f"{c['ds']}/base_default", cid))
        if radical_gold(c):
            for arm in ('consistent', 'contradictory'):
                v = [is_radical_mech(K.cached(f'm4/{arm}_s{s}', cid)) for s in range(1, NS + 1)]
                v = [x for x in v if x is not None]
                r[f'polar_{arm}'] = float(np.mean([not x for x in v])) if v else None
            b = is_radical_mech(K.cached(f"{c['ds']}/base_default", cid))
            r['polar_none'] = None if b is None else float(not b)
        rows.append(r)
    S = ('low', 'mid', 'high', 'recent')
    by = {s: [r for r in rows if r['stratum'] == s] for s in S}
    rep = {'net_check': {'n': len(net), 'titles_correct': int(sum(net))},
           'R1_recall_rate': {s: boot(by[s], mean_of('R1')) for s in S},
           'R1_slope_per_10x_cites': boot(rows, slope_of('R1')),
           'refs_listed_mean': {s: round(float(np.mean([r['n_refs'] for r in by[s]])), 2) for s in S},
           'listed_doi_exists_on_crossref': [verified, listed, round(verified / max(listed, 1), 3)],
           'A': {s: boot(by[s], mean_of('A')) for s in S},
           'C': {s: boot(by[s], mean_of('C')) for s in S},
           'memory_index_CminusA': {s: boot(by[s], mean_of('CminusA')) for s in S},
           'memory_index_slope_per_10x_cites': boot(rows, slope_of('CminusA')),
           'old_all_memory_index': boot([r for r in rows if r['stratum'] != 'recent'], mean_of('CminusA'))}
    hi = by['high']
    rec, nrec = [r for r in hi if r['R1'] > 0], [r for r in hi if r['R1'] == 0]
    rep['high_Direct_by_R1'] = {'recalled': boot(rec, mean_of('Direct')) if len(rec) > 1 else None,
                                'not_recalled': boot(nrec, mean_of('Direct')) if len(nrec) > 1 else None}
    old = [r for r in rows if r['stratum'] != 'recent']
    rec, nrec = [r for r in old if r['R1'] > 0], [r for r in old if r['R1'] == 0]
    rep['old_Direct_by_R1'] = {'recalled': boot(rec, mean_of('Direct')) if len(rec) > 1 else None,
                               'not_recalled': boot(nrec, mean_of('Direct')) if len(nrec) > 1 else None}
    m4 = [r for r in rows if r.get('polar_consistent') is not None and r.get('polar_contradictory') is not None]
    rep['M4'] = {'n': len(m4), 'by_stratum': {s: sum(r['stratum'] == s for r in m4) for s in S},
                 'polar_rate': {a: boot(m4, mean_of(f'polar_{a}')) for a in ('none', 'consistent', 'contradictory')},
                 'PRIMARY_contradictory_minus_consistent':
                     boot(m4, lambda rs: float(np.mean([r['polar_contradictory'] - r['polar_consistent'] for r in rs]))),
                 'old_only': boot([r for r in m4 if r['stratum'] != 'recent'],
                                  lambda rs: float(np.mean([r['polar_contradictory'] - r['polar_consistent']
                                                            for r in rs])))}
    (RUN / 'SUMMARY_M1_M4.json').write_text(json.dumps({'summary': rep, 'rows': rows}, indent=1))
    print(json.dumps(rep, indent=1))


def NET_IDS(cs):  # noqa: N802
    rec = sorted(cid for cid, c in cs.items() if c['stratum'] == 'recent')
    return random.Random(SEED).sample(rec, min(20, len(rec)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=200)
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    cli = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    cs = all_cases()
    if cli.limit:
        keep = sorted(cs)[:cli.limit] + sorted(cs)[-cli.limit:]
        cs = {k: cs[k] for k in keep}
    meta = metadata(list(cs))
    cs = {k: v for k, v in cs.items() if meta.get(k)}
    rad = [c for c in cs.values() if radical_gold(c)]
    print(json.dumps({'cases': len(cs), 'radical_gold': len(rad)}), flush=True)
    if not cli.analyze_only:
        w = cli.workers
        jobs = [(job_net, (cid,)) for cid in (NET_IDS(cs) if not cli.limit else [])]
        jobs += [(job_r1, (c, s)) for c in cs.values() for s in range(1, NS + 1)]
        jobs += [(job_a, (c, s)) for c in cs.values() for s in range(1, NS + 1)]
        jobs += [(job_c, (c, s, meta)) for c in cs.values() for s in range(1, NS + 1)]
        jobs += [(job_m4, (c, a, s)) for c in rad for a in ('consistent', 'contradictory') for s in range(1, NS + 1)]

        def run(fn, args):
            return fn(*args)
        for _ in range(3):
            K.run_pool(run, jobs, w)
            jobs = [(fn, a) for fn, a in jobs if not _done(fn, a)]
            if not jobs:
                break
    analyze(cs, meta)


def _done(fn, a):
    c = a[0]
    cid = c if isinstance(c, str) else c['id']
    tag = {job_net: 'm1/net', job_r1: f'm1/R1_s{a[-1] if fn is job_r1 else 0}',
           job_a: f'm1/A_s{a[1] if fn is job_a else 0}', job_c: f'm1/C_s{a[1] if fn is job_c else 0}'}.get(fn)
    if fn is job_m4:
        tag = f'm4/{a[1]}_s{a[2]}'
    return K.cached(tag, cid) is not None


if __name__ == '__main__':
    main()
