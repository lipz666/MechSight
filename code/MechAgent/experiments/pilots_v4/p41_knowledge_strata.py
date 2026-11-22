"""2026 literature: split author intermediates by whether the clue-free MechSight-Auto pool already contains them.

in-pool  : proposed by any unguided pathway sample (Direct, 4 stochastic samples, hidden-intermediate enumeration)
           or any RDKit-checked gap-completion state
out-pool : never proposed without experimental clues (literature analogue of a blind spot)
Recall per stratum for Direct, MechSight-Auto, Direct + Clues and full MechSight-Guided, paper-clustered bootstrap.
Run from MechAgent/experiments/pilots_v4:  python pool_strata.py {dev|test} out.json
"""
import json, sys
sys.path.insert(0, '.')
import common4 as K
import p11_mechagent as M
import p30_lit172_gold as P
import p31_guided71 as T

which, out = sys.argv[1], sys.argv[2]
T.configure(which)
DS = P.DS
cs = P.cases(); P.CS = cs
POOL = ['base_default', 'samp1', 'samp2', 'samp3', 'samp4', 'enum3']


def guided_full(cid):
    a1 = K.cached(f'{DS}/A1_place', cid)
    return a1 if a1 else K.cached(f'{DS}/G71_rev', cid)


ARMS = {'Direct': lambda c: K.cached(f'{DS}/base_default', c),
        'MechSight-Auto': lambda c: M.final(DS, c, 2),
        'Direct + Clues': lambda c: K.cached(f'{DS}/O_llm', c),
        'MechSight-Guided': guided_full}

ST=('direct','inert','out')
hits = {a: {s: [] for s in ST} for a in ARMS}
ids = {s: [] for s in ST}
n_papers = 0
for cid, c in cs.items():
    if not c['clues'] or not any(K.species_keys(s) for s in c['author']):
        continue
    if K.cached(f'{DS}/G71_rev', cid) is None:
        continue
    n_papers += 1
    pool = []
    for t in POOL:
        pool += T.species(K.parse_mech(K.cached(f'{DS}/{t}', cid)))
    for t in ('V5_comp1', 'V5_comp2'):
        a = K.cached(f'{DS}/{t}', cid) or ''
        pool += [k for _, s in M.checked_states(c, a) if (k := K.species_keys(s))]
    preds = {a: T.species(K.parse_mech(f(cid))) for a, f in ARMS.items()}
    for g in (k for s in c['author'] if (k := K.species_keys(s))):
        stratum = 'direct' if any(g & p for p in preds['Direct']) else ('inert' if any(g & p for p in pool) else 'out')
        ids[stratum].append(cid)
        for a in ARMS:
            hits[a][stratum].append(float(any(g & p for p in preds[a])))

rep = {'set': which, 'papers': n_papers, **{'n_'+s: len(ids[s]) for s in ST}}
for st in ST:
    rep[st] = {a: K.phase2.ci(hits[a][st], ids[st]) for a in ARMS}
    base = {}
    for hi, lo in (('MechSight-Auto', 'Direct'), ('MechSight-Guided', 'MechSight-Auto'),
                   ('MechSight-Guided', 'Direct + Clues'), ('Direct + Clues', 'Direct')):
        d = [x - y for x, y in zip(hits[hi][st], hits[lo][st])]
        base[f'{hi} - {lo}'] = K.phase2.ci(d, ids[st])
    rep[st + '_diffs'] = base
json.dump(rep, open(out, 'w'), indent=1)
print(json.dumps(rep, indent=1))
