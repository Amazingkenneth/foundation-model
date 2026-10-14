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
python scripts/train.py --stage 1 --epochs "${EPOCHS1:-3}" --lr "${LR1:-1e-4}" --out "${OUT}/s1" ${TRAIN_ARGS:-}
python scripts/train.py --stage 2 --epochs "${EPOCHS2:-3}" --lr "${LR2:-5e-5}" --init "${OUT}/s1/best.pt" --out "${OUT}/s2" ${TRAIN_ARGS:-}
# ship the operator only if it beat the analytic chain on validation
if python - "${OUT}/s2/history.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
best = min(e["val"] for e in h)
sys.exit(0 if best < h[0]["base"] else 1)
PY
then
  cp "${OUT}/s2/fm_weights.npz" submission/fm_weights.npz
  echo "operator improved validation MSE: submission/fm_weights.npz updated"
else
  echo "operator did not beat the analytic chain on validation: weights NOT shipped"
fi
python scripts/eval_fnp.py --split val
