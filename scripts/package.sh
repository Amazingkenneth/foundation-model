#!/usr/bin/env bash
# Validate and package the submission folder into dist/submission-<git sha>.tar.gz.
#
#   scripts/package.sh            # validate (contract) + smoke check + package
#   SKIP_SMOKE=1 scripts/package.sh
#
# Folder mode: everything in submission/ that is not ignored ships (estimator.py, the
# fm_* modules, backend_fnp.py, the operator weights k34.npz + mp.npz or fm_weights.npz,
# LICENSE_504aldo). Caps: 50 MiB, 50 files.
set -euo pipefail
cd "$(dirname "$0")/.."

SHA=$(git rev-parse --short HEAD 2>/dev/null || echo local)
DIRTY=$(git status --porcelain submission 2>/dev/null | head -1 || true)
[ -n "$DIRTY" ] && SHA="${SHA}-dirty"
mkdir -p dist
OUT="dist/submission-${SHA}.tar.gz"

echo "== whest validate (contract: shapes, finiteness, setup)"
whest validate --estimator submission/estimator.py --format plain

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "== smoke: non-suite shapes and the stdlib-only import rule"
  python scripts/smoke.py
fi

echo "== whest package -> ${OUT}"
rm -f "${OUT}"
whest package --estimator submission --output "${OUT}" --yes --format plain
whest validate-package "${OUT}" --format plain 2>/dev/null || whest validate-package "${OUT}"
ls -l "${OUT}"
echo "${OUT}" > dist/LATEST
