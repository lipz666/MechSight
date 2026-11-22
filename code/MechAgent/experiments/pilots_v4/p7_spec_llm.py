"""LLM direct baselines on the sets used for the specialised-model comparison that P6 does not
cover: T3_confirm (48) and the mech-USPTO-31k sample (200). Official default prompt, T=0.2."""
from __future__ import annotations

import argparse

import common4 as K


def job(ds, case):
    return 'ok' if K.call(f'{ds}/base_default', case['id'], K.official_prompt(case)) else 'error'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=20)
    cli = ap.parse_args()
    s = {'T3_confirm': K.load_fukuyama('T3_confirm'), 'uspto31k': K.load_uspto()}
    K.run_pool(job, [(ds, c) for ds, cs in s.items() for c in cs.values()], cli.workers)
