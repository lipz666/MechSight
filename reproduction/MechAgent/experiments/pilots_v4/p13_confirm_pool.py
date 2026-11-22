"""P13: method-independent inputs on the confirmation sets (T3_confirm 48, oMe-Gold 196).

Both freeze candidates (V5 and M_ms_none) use the same pool and baselines, so these calls are
made before the freeze decision: base_default, base_cot, samp1..4 (T=0.7), enum3 (P2, T=0.7).
"""
from __future__ import annotations

import argparse

import common4 as K
import p2_enumerate as P2
import p6_baselines as P6


def sets():
    return {'T3_confirm': K.load_fukuyama('T3_confirm'), 'gold': K.load_gold()}


def job(ds, arm, case):
    if arm == 'enum3':
        return P2.job(ds, case)
    return P6.job(ds, arm, case)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=400)
    cli = ap.parse_args()
    arms = ['base_default', 'base_cot'] + [f'samp{i}' for i in range(1, 5)] + ['enum3']
    K.run_pool(job, [(ds, a, c) for ds, cs in sets().items() for a in arms for c in cs.values()], cli.workers)
