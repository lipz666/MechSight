import json, sys, random
sys.path.insert(0, '.')
import common4 as K, p11_mechagent as M, p12_controls as C, p44_stepwise as Q
items = json.loads(Q.STEPS.read_text())
sets = M.sets(('T3_dev', 'T3_confirm'))
res = {'hidden': [], 'visible_only': []}
nh = 0
for it in items:
    if it['set'] != 'fuku': continue
    case = sets.get(it['split'], {}).get(it['cid'])
    ans = K.cached(Q.TAG, it['sid'])
    if case is None or ans is None: continue
    hid, vis = C.gt_intermediates(case)
    prevk = Q.keyset(it['prev'], K.species_keys); nextk = Q.keyset(it['next'], K.species_keys)
    forms_hidden = any((k := K.species_keys(h)) and k <= nextk and not k <= prevk for h in hid)
    st, eq = Q.hit(it, Q.parse_state(ans))
    res['hidden' if forms_hidden else 'visible_only'].append((st, eq))
for k, v in res.items():
    print(k, len(v), 'strict', round(100*sum(x[0] for x in v)/len(v), 1), 'equiv', round(100*sum(x[1] for x in v)/len(v), 1))
