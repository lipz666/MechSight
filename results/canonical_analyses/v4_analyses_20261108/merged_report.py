import json, random, sys, collections
S = sys.argv[1]
R = json.load(open(f'{S}/rows_dev.json')) + json.load(open(f'{S}/rows_test.json'))
A = ['Direct', 'Auto', 'Direct+Clues', 'MV7', 'Guided']
def rec(r, a): return sum(s[a] for s in r['strata']) / len(r['strata'])
def comp(r, a): return float(all(s[a] for s in r['strata']))
def ci(vals, B=4000, seed=0):
    rnd = random.Random(seed); n = len(vals)
    bs = sorted(sum(vals[rnd.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return round(100 * sum(vals) / n, 1), round(100 * bs[int(.025 * B)], 1), round(100 * bs[int(.975 * B)], 1)
out = {}
for name, f in (('all_436', lambda r: True), ('apr_oct', lambda r: r['date'] >= '2026-04'),
                ('may_oct', lambda r: r['date'] >= '2026-05'), ('jan_mar', lambda r: r['date'] < '2026-04')):
    X = [r for r in R if f(r)]
    o = {'papers': len(X), 'by_set': dict(collections.Counter(r['set'] for r in X)),
         'intermediates': sum(len(r['strata']) for r in X)}
    for a in A: o[a] = {'recall': ci([rec(r, a) for r in X]), 'complete': ci([comp(r, a) for r in X])}
    for hi, lo in (('Guided', 'Direct+Clues'), ('Guided', 'MV7'), ('Auto', 'Direct'), ('Direct+Clues', 'Direct'), ('Guided', 'Direct')):
        o[f'{hi}-{lo}'] = {'recall': ci([rec(r, hi) - rec(r, lo) for r in X]), 'complete': ci([comp(r, hi) - comp(r, lo) for r in X])}
    st = collections.defaultdict(list); ids = collections.defaultdict(list)
    for r in X:
        for s in r['strata']: st[s['st']].append(s)
    N = sum(len(v) for v in st.values())
    o['strata'] = {k: {'share': round(100 * len(v) / N, 1), 'n': len(v), **{a: round(100 * sum(s[a] for s in v) / len(v), 1) for a in A}}
                   for k, v in st.items()}
    out[name] = o
json.dump(out, open(f'{S}/merged_apr_oct.json', 'w'), indent=1)
for name, o in out.items():
    print(f"== {name}: {o['papers']} papers {o['by_set']}, {o['intermediates']} intermediates")
    for a in A: print(f"   {a:13s} recall {o[a]['recall']}  complete {o[a]['complete']}")
    for k in ('Guided-Direct+Clues', 'Guided-MV7', 'Auto-Direct', 'Direct+Clues-Direct'): print(f"   {k:20s} {o[k]}")
    for k in ('direct', 'inert', 'blind'): print('   ', k, o['strata'].get(k))
