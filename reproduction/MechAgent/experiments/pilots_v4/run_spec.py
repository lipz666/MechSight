"""Specialised models (ChRIMP/MechSMILES T5 equ_min_n_flower, FlowER) on the shared benchmarks, on the 4060 host.

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
from pathlib import Path

ORDER = ['T3_confirm', 'uspto31k', 'gold']
LIMITS = {'chrimp': (240, 420), 'flower': (300, 480)}  # (soft, hard) seconds per case


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
    from chrimp.agent.t5_agent import T5Agent
    agent = T5Agent(inference_input_format='retro')
    agent.load_model(tokenizer_path='SchwallerGroup/MechSMILES_tokenizer',
                     model_path='SchwallerGroup/chrimp_T5_equ_min_n_flower')

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
    sys.path.insert(0, os.path.expanduser('~/mech_agent/MechAgent/experiments/flower_compare'))
    import networkx as nx
    import torch
    import flower_driver as D
    import flower_smiles_driver as S
    args = D.Args
    args.local_rank, args.device = -1, torch.device('cuda')
    state = torch.load(os.path.join(args.model_path, args.model_name), weights_only=False, map_location='cpu')
    pre = state['args']
    pre.load_from, pre.device, pre.local_rank = None, args.device, -1
    model, flow, _ = D.init_model(pre)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['state_dict'].items()})
    model.eval()

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
