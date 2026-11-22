import json, random, sys
S = sys.argv[1]
R = json.load(open(f'{S}/bench_eq/rows_dev.json')) + json.load(open(f'{S}/bench_eq/rows_test.json'))
A = ['Direct', 'Auto', 'Direct+Clues', 'MV7', 'Guided']
def rec(r, a): return sum(s[a] for s in r['strata']) / len(r['strata'])
def comp(r, a): return float(all(s[a] for s in r['strata']))
def ci(v):
    g = random.Random(0); n = len(v); bs = sorted(sum(v[g.randrange(n)] for _ in range(n)) / n for _ in range(4000))
    return f"{100*sum(v)/n:.1f} [{100*bs[100]:.1f}, {100*bs[3899]:.1f}]"
G = {'285 (Apr-Oct)': lambda r: r['date'] >= '2026-04', '167 not used in design': lambda r: r['date'] >= '2026-04' and r['set'] == 'test',
     '118 used in design': lambda r: r['date'] >= '2026-04' and r['set'] == 'dev', '151 Jan-Mar': lambda r: r['date'] < '2026-04'}
for g, f in G.items():
    X = [r for r in R if f(r)]
    print('==', g, len(X), sum(len(r['strata']) for r in X))
    for a in A:
        print(f"  {a:13s} rec {100*sum(rec(r,a+'_strict') for r in X)/len(X):.1f} / {100*sum(rec(r,a) for r in X)/len(X):.1f}   comp {100*sum(comp(r,a+'_strict') for r in X)/len(X):.1f} / {100*sum(comp(r,a) for r in X)/len(X):.1f}")
    print('  G-DC rec_e', ci([rec(r,'Guided')-rec(r,'Direct+Clues') for r in X]), 'comp_e', ci([comp(r,'Guided')-comp(r,'Direct+Clues') for r in X]))
    print('  G-DC rec_s', ci([rec(r,'Guided_strict')-rec(r,'Direct+Clues_strict') for r in X]), 'comp_s', ci([comp(r,'Guided_strict')-comp(r,'Direct+Clues_strict') for r in X]))
