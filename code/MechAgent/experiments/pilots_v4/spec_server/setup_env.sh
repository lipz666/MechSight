#!/usr/bin/env bash
# GPU environment setup for FlowER and ChRIMP specialist baselines (1 GPU, 8 CPUs, 16 GiB RAM).
set -euo pipefail
cd ~/mechspec
UV="${UV:-uv}"
export UV_HTTP_TIMEOUT=300 UV_LINK_MODE=copy
rm -rf .venv
$UV venv -q --python /usr/bin/python3 .venv
$UV pip install -q --python .venv/bin/python 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
$UV pip install -q --python .venv/bin/python 'numpy==2.2.6' 'rdkit==2025.9.5' torchdiffeq iteround 'networkx==3.4.2' \
  "transformers>=4.40,<4.50" accelerate datasets colorama pandas matplotlib pillow scipy tqdm wandb huggingface_hub
export HF_HOME=~/mechspec/.hf
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for r in ('SchwallerGroup/MechSMILES_tokenizer', 'SchwallerGroup/chrimp_T5_equ_min_n_flower'):
    print(snapshot_download(r))
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
.venv/bin/python -c "import torch,numpy,rdkit;print('cuda', torch.__version__, numpy.__version__, rdkit.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo SETUP_OK
