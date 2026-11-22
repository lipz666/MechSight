"""Free-rollout rerun of the specialists on the lab 4090 server (2026-11-08): identical search settings to run_spec.py
(ChRIMP best-first, node budget 50; FlowER beam 5, nbest 3, depth 15), only the time limit is raised (4060 laptop:
300/240 s soft -> here 900 s soft on a 4090).

Original notes:
Specialised models (ChRIMP/MechSMILES T5 equ_min_n_flower, FlowER) on the shared benchmarks, on the 4060 host.

Input: spec_inputs.json (reactants + main product). Order fixed before running: T3_confirm, uspto31k, gold
(sorted ids), so partial runs cover whole benchmarks first. A worker process holds the model; inside it
SIGALRM stops a search after SOFT seconds (FlowER keeps its partial graph); if the worker is stuck in
native code the parent kills it after HARD seconds and records a timeout.
Usage (flower env): python run_spec.py {chrimp|flower} spec_inputs.json out_dir shard/n [rev]
(rev walks the same shard backwards, so a second process can work from the other end; cases already
written are skipped, so the two meet in the middle.)
Outputs: one JSON per case, the same formats as run_chrimp.py / flower_smiles_driver.py (spec_score.py).
"""
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_common as SC

ORDER = ['T3_confirm', 'uspto31k', 'gold']
LIMITS = {'chrimp': (900, 1100), 'flower': (900, 1100)}  # (soft, hard) seconds per case


def unmap(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return smi
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    return Chem.MolToSmiles(m)


class Soft(Exception):
    pass


def _alarm(*_):
    raise Soft()


def chrimp_solver():
    agent = SC.load_chrimp()

    def solve(c, soft):
        rec = {}
        signal.alarm(soft)
        try:
            mechs = agent.search_mech('.'.join(c['reactants']) + '>>' + c['product'], max_node_budget=50)
            rec['mechanisms'] = [[unmap(s.split('|')[0]) for s in m[1:]] for m in mechs]
            rec['raw'] = mechs
            rec['timed_out'] = False
        except Soft:
            rec['mechanisms'], rec['timed_out'] = [], True
        finally:
            signal.alarm(0)
        return rec
    return solve


def flower_solver():
    import networkx as nx
    BP, args, model, flow = SC.load_flower(nbest=3)
    D = types.SimpleNamespace(BP=BP)

    class S:
        @staticmethod
        def mapped_smiles(reactants):
            return BP.reactant_process('.'.join(reactants))

    def solve(c, soft):
        rec = {}
        smi = S.mapped_smiles(c['reactants'])
        g = nx.DiGraph()
        g.add_node(smi, depth=1)
        signal.alarm(soft)
        try:
            D.BP.beam_search(args, model, flow, {0: [smi]}, [(g, smi, (smi, []))])
            rec['timed_out'] = False
        except Soft:
            rec['timed_out'] = True
        finally:
            signal.alarm(0)
        nodes = list(g.nodes)
        idx = {x: i for i, x in enumerate(nodes)}
        rec['graph'] = {'root': idx[smi], 'nodes': nodes,
                        'edges': [[idx[u], idx[v], int(d.get('count', 0))] for u, v, d in g.edges(data=True)]}
        return rec
    return solve


def worker(kind, conn):
    signal.signal(signal.SIGALRM, _alarm)
    solve = chrimp_solver() if kind == 'chrimp' else flower_solver()
    conn.send('ready')
    while True:
        c = conn.recv()
        try:
            rec = solve(c, LIMITS[kind][0])
        except Exception as e:
            rec = {'mechanisms': [], 'error': f'{type(e).__name__}: {str(e)[:200]}'}
        conn.send(rec)
        if 'CUDA' in rec.get('error', ''):  # a broken CUDA context fails every later case: exit, parent restarts
            os._exit(1)


def main():
    kind, inputs, out, sh = sys.argv[1:5]
    shard, n = map(int, sh.split('/'))
    cases = [c for c in json.load(open(inputs)) if c['bench'] in ORDER]
    cases = sorted(cases, key=lambda c: (ORDER.index(c['bench']), c['id']))[shard::n]
    if sys.argv[5:6] == ['rev']:
        cases = cases[::-1]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context('spawn')
    proc = conn = None
    pending, retried, fails = list(cases), set(), 0
    while pending:
        c = pending.pop(0)
        dest = out / f"{c['bench']}__{c['id']}.json"
        if dest.exists():
            continue
        if proc is None or not proc.is_alive():
            conn, child = ctx.Pipe()
            proc = ctx.Process(target=worker, args=(kind, child), daemon=True)
            proc.start()
            if not conn.poll(600) or conn.recv() != 'ready':
                proc.kill()
                proc = None
                fails += 1
                print('worker failed to start', fails, flush=True)
                if fails >= 10:
                    sys.exit(1)  # no 'done' line, so the watchdog restarts this shard later
                pending.insert(0, c)
                time.sleep(120)
                continue
            fails = 0
        t0 = time.time()
        conn.send(c)
        if conn.poll(LIMITS[kind][1]):
            rec = conn.recv()
        else:
            proc.kill()
            proc.join()
            proc = None
            rec = {'mechanisms': [], 'timed_out': True, 'killed': True}
        if 'CUDA' in rec.get('error', ''):  # worker exits after a CUDA error; retry the case once with a fresh one
            if proc is not None:
                proc.join(30)
            proc = None
            if c['id'] not in retried:
                retried.add(c['id'])
                pending.insert(0, c)
                print(c['bench'], c['id'], 'cuda error, retry', flush=True)
                continue
        rec.update({'id': c['id'], 'bench': c['bench'], 'seconds': round(time.time() - t0, 1)})
        dest.write_text(json.dumps(rec))
        print(c['bench'], c['id'], 'timeout' if rec.get('timed_out') else 'ok', rec.get('error', ''), rec['seconds'],
              flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
