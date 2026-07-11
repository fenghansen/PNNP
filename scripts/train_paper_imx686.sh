#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${1:-9}"
python3 trainer_PNNP_LRID.py \
  -f runfiles/IMX686/PNNP_paper_stage1.yml \
  --mode train \
  --gpu "${GPU_ID}"

python3 trainer_PNNP_LRID.py \
  -f runfiles/IMX686/PNNP_paper_stage2.yml \
  --mode train \
  --gpu "${GPU_ID}"
