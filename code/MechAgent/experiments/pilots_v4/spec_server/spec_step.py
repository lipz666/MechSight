"""Single-step prediction from the reference previous state (P44) with FlowER or ChRIMP.

python spec_step.py {flower|chrimp} steps.json out.jsonl shard/n
Appends one JSON line per step: {sid, preds: [ranked next-state SMILES, up to 5], error?}. Steps already in out.jsonl
are skipped, so a requeued job resumes.
FlowER: the mapped, H-explicit state is expanded once (64 flow samples, top 5 distinct outcomes by count).
ChRIMP: input '[prod]goal[reac]state[mech]' (its search format), 10 beams, moves applied with MechSmiles, top 5 distinct
products in beam order. ChRIMP needs a goal product, so FlowER's own test steps are skipped for it.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_common as SC


def done_ids(path):
    if not os.path.exists(path):
        return set()
    out = set()
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if 'OutOfMemory' in r.get('error', '') or 'CUDA' in r.get('error', ''):
            out.discard(r['sid'])  # retried on the next run; the scorer keeps the last record per sid
        else:
            out.add(r['sid'])
    return out


def run_flower(items, out):
    BP, args, model, flow = SC.load_flower(nbest=5)
    from train import init_loader
    from utils.data_utils import ReactionDataset

    def expand(smis):
        ds = ReactionDataset(args, smis, reactant_only=True)
        loader = init_loader(args, ds, batch_size=args.test_batch_size, shuffle=False, epoch=None, use_sort=False)
        return BP.expand(args, model, flow, loader)

    def mapped(it):
        return it.get('prev_mapped') or BP.reactant_process('.'.join(it['prev']))

    for i in range(0, len(items), 16):
        chunk = items[i:i + 16]
        recs = {}
        smis = {}
        for it in chunk:
            try:
                smis[it['sid']] = mapped(it)
            except Exception as e:
                recs[it['sid']] = {'sid': it['sid'], 'preds': [], 'error': f'map: {e}'[:200]}
        try:
            res = expand(list(dict.fromkeys(smis.values())))
            for sid, s in smis.items():
                recs[sid] = {'sid': sid, 'preds': list(res.get(s, {}).keys())}
        except Exception:
            import torch
            torch.cuda.empty_cache()
            for sid, s in smis.items():  # isolate the failing state
                torch.cuda.empty_cache()
                try:
                    recs[sid] = {'sid': sid, 'preds': list(expand([s]).get(s, {}).keys())}
                except Exception as e:
                    recs[sid] = {'sid': sid, 'preds': [], 'error': f'{type(e).__name__}: {e}'[:200]}
        with open(out, 'a') as f:
            for r in recs.values():
                f.write(json.dumps(r) + '\n')
        print(i + len(chunk), 'of', len(items), flush=True)


def run_chrimp(items, out):
    agent = SC.load_chrimp()
    import torch
    from chrimp.agent.supervised_search import canonicalize_special, try_canonicalize_prod_msmi
    from chrimp.world.mechsmiles import MechSmiles
    tok, model = agent.tokenizer, agent.model
    model.eval().to('cuda')
    tok.padding_side = 'right'
    for n, it in enumerate(items, 1):
        rec = {'sid': it['sid'], 'preds': []}
        try:
            curr = canonicalize_special('.'.join(it['prev']))
            goal = canonicalize_special(it['product'])
            inp = agent.human_to_input(f'{curr}>>{goal}', format=agent.inference_input_format, already_canonicalized=False)
            enc = {k: v.to('cuda') for k, v in tok([inp], return_tensors='pt', padding=True).items()
                   if k in ('input_ids', 'attention_mask')}  # as in ChRIMP's own search
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=512, num_beams=10, num_return_sequences=10, do_sample=False)
            seen = []
            for txt in tok.batch_decode(gen, skip_special_tokens=True):
                try:
                    prod, ok = try_canonicalize_prod_msmi(MechSmiles(txt.replace(' ', ''), context=curr))
                except Exception:
                    continue
                if ok and prod not in seen:
                    seen.append(prod)
            rec['preds'] = seen[:5]
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {e}'[:200]
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        if n % 50 == 0:
            print(n, 'of', len(items), flush=True)


def main():
    kind, steps, out, sh = sys.argv[1:5]
    shard, n = map(int, sh.split('/'))
    items = json.load(open(steps))
    if kind == 'chrimp':
        items = [it for it in items if it['product']]
    items = items[shard::n]
    seen = done_ids(out)
    items = [it for it in items if it['sid'] not in seen]
    print(kind, 'todo', len(items), flush=True)
    t0 = time.time()
    (run_flower if kind == 'flower' else run_chrimp)(items, out)
    print('done', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
