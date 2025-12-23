#!/bin/bash
set -euo pipefail

# ==== 可按需修改 ====
ACTOR_PATH_B="../actor.pt"  
HANABI_NAME="Hanabi-Full"
NUM_AGENTS=2
HIDDEN_SIZE=512
LAYER_N=2
USE_RECURRENT_FLAG=""         
HOST="0.0.0.0"
PORT=8000
CONDA_ENV="marl"
CUDA_FLAG="--cuda"
WEB_DIR="$(pwd)/on-policy/onpolicy/web/hanabi"


cd on-policy


python -m onpolicy.scripts.hanabi_web \
  --actor_path_b "${ACTOR_PATH_B}" \
  --hanabi_name "${HANABI_NAME}" \
  --num_agents ${NUM_AGENTS} \
  --hidden_size ${HIDDEN_SIZE} \
  --layer_N ${LAYER_N} \
  ${USE_RECURRENT_FLAG} \
  ${CUDA_FLAG} \
  --host "${HOST}" --port ${PORT} \
  --web_dir "${WEB_DIR}"
