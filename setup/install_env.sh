#!/bin/bash
# Build the training environment (.venv in the repo root) with uv.
#
# Pinned to the exact stack the gate1 runs used on bt-gaming (verified
# 2026-11-09 by listing the training venv):
#   python 3.10, torch 2.14.1+cu130, transformers 4.57.1, trl 0.26.0,
#   peft 0.17.1, accelerate 1.11.0, datasets 3.6.0, wandb 0.22.3
# vLLM is intentionally NOT installed: the vLLM route was tried and
# rejected (WSL2/8GB CuMem failure); see docs/gate1_results.md.
set -eux
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
[ -d .venv ] || uv venv --python 3.10 .venv
export VIRTUAL_ENV="$PWD/.venv"
uv pip install \
  torch==2.14.1 \
  transformers==4.57.1 \
  trl==0.26.0 \
  peft==0.17.1 \
  accelerate==1.11.0 \
  datasets==3.6.0 \
  wandb==0.22.3
.venv/bin/python -c "
import torch, transformers, trl, peft, accelerate, datasets
print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('trl', trl.__version__)
print('peft', peft.__version__)
print('accelerate', accelerate.__version__)
print('datasets', datasets.__version__)
"
echo "INSTALL_DONE"
