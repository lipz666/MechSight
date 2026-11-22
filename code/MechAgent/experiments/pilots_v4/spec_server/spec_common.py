"""Model loading for FlowER and ChRIMP on the lab 4090 server (~/mechspec). Shared by spec_step.py and spec_rollout.py."""
import os
import sys
import types

HOME = os.path.expanduser('~/mechspec')
FLOWER = os.path.join(HOME, 'FlowER')


def _stub_visual():
    # ChRIMP imports drawing libraries at module level; they are not used here and libcairo is absent.
    for name in ('cairosvg', 'svgutils', 'svgutils.transform'):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules['svgutils'].transform = sys.modules['svgutils.transform']


def flower_env(nbest):
    os.environ.update({'DATA_NAME': 'flower_new_dataset', 'EXP_NAME': 'best_large_hyperparam', 'EMB_DIM': '256',
                       'RBF_HIGH': '12', 'RBF_GAP': '0.1', 'SIGMA': '0.15', 'MODEL_NAME': 'model.2940000_97.pt',
                       'SCALE': '1', 'TEST_BATCH_SIZE': '2048', 'NUM_WORKERS': '0', 'NUM_GPUS_PER_NODE': '1',
                       'NBEST': str(nbest),
                       'MODEL_PATH': os.path.join(FLOWER, 'checkpoints/flower_new_dataset/best_large_hyperparam') + '/',
                       'RESULT_PATH': '/dev/null/'})
    if FLOWER not in sys.path:
        sys.path.insert(0, FLOWER)


def load_flower(nbest=5):
    flower_env(nbest)
    import torch
    import beam_predict as BP
    from settings import Args
    from train import init_model
    args = Args
    args.local_rank, args.device = -1, torch.device('cuda')
    state = torch.load(os.path.join(args.model_path, args.model_name), weights_only=False, map_location='cpu')
    pre = state['args']
    pre.load_from, pre.device, pre.local_rank = None, args.device, -1
    model, flow, _ = init_model(pre)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['state_dict'].items()})
    model.eval()
    # The 4090s are shared, so large states can run out of memory. On OOM, retry the same 64 flow samples in more,
    # smaller sample chunks (FlowER's own `split` argument); the predictions are unchanged in distribution.
    orig = BP.predict_batch

    def safe_predict(a, batch_idx, data_batch, m, f, split, rand_matrix=None):
        s = split
        while True:
            try:
                return orig(a, batch_idx, data_batch, m, f, s, rand_matrix)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if s >= args.sample_size:
                    raise
                s *= 2
    BP.predict_batch = safe_predict
    return BP, args, model, flow


def load_chrimp():
    _stub_visual()
    sys.path.insert(0, os.path.join(HOME, 'chrimp_src'))
    from chrimp.agent.t5_agent import T5Agent
    agent = T5Agent(inference_input_format='retro')
    agent.load_model(tokenizer_path='SchwallerGroup/MechSMILES_tokenizer',
                     model_path='SchwallerGroup/chrimp_T5_equ_min_n_flower')
    return agent
