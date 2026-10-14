#!/usr/bin/env bash
# Full training pipeline: stage 1 (input corrections) -> stage 2 (+ output corrections),
# keep the best-validation epoch of each stage, copy the stage-2 weights into submission/
# and evaluate the submission under flopscope.
#
#   EPOCHS1=3 EPOCHS2=3 scripts/train_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JEMALLOC=$(ls /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 2>/dev/null || true)
[ -n "${JEMALLOC}" ] && export LD_PRELOAD="${JEMALLOC}" MALLOC_CONF="thp:always,dirty_decay_ms:60000,muzzy_decay_ms:60000"
OUT=${OUT:-runs}
python scripts/train.py --stage 1 --epochs "${EPOCHS1:-3}" --lr "${LR1:-1e-3}" --out "${OUT}/s1" ${TRAIN_ARGS:-}
python scripts/train.py --stage 2 --epochs "${EPOCHS2:-3}" --lr "${LR2:-5e-4}" --init "${OUT}/s1/best.pt" --out "${OUT}/s2" ${TRAIN_ARGS:-}
cp "${OUT}/s2/fm_weights.npz" submission/fm_weights.npz
python scripts/eval_fnp.py --split val
