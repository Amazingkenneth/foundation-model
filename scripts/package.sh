#!/usr/bin/env bash
# Validate and package the submission folder into dist/submission-<git sha>.tar.gz.
#
#   scripts/package.sh            # validate (contract) + smoke check + package
#   SKIP_SMOKE=1 scripts/package.sh
#   SUBMISSION_DIR=candidates/v5b_rfhub scripts/package.sh   # package another folder
#
# Folder mode: everything in submission/ that is not ignored ships (estimator.py, the
# fm_* modules, backend_fnp.py, the operator weights k34.npz + mp.npz or fm_weights.npz,
# LICENSE_504aldo). Caps: 50 MiB, 50 files.
set -euo pipefail
cd "$(dirname "$0")/.."

SUB_DIR="${SUBMISSION_DIR:-submission}"
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo local)
DIRTY=$(git status --porcelain "${SUB_DIR}" 2>/dev/null | head -1 || true)
[ -n "$DIRTY" ] && SHA="${SHA}-dirty"
mkdir -p dist
TAG=$(basename "${SUB_DIR}")
OUT="dist/${TAG}-${SHA}.tar.gz"

echo "== whest validate (contract: shapes, finiteness, setup)"
whest validate --estimator "${SUB_DIR}/estimator.py" --format plain

if [ "${SKIP_SMOKE:-0}" != "1" ] && [ "${SUB_DIR}" = "submission" ]; then
  echo "== smoke: non-suite shapes and the stdlib-only import rule"
  python scripts/smoke.py
fi

echo "== whest package -> ${OUT}"
rm -f "${OUT}"
whest package --estimator "${SUB_DIR}" --output "${OUT}" --yes --format plain
whest validate-package "${OUT}" --format plain 2>/dev/null || whest validate-package "${OUT}"
ls -l "${OUT}"
echo "${OUT}" > dist/LATEST
