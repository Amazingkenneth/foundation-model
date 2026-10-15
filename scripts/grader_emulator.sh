#!/usr/bin/env bash
# Run an estimator folder the way the AIcrowd grader does: flopscope-server 0.12.0 in one
# process, the estimator against flopscope-client 0.12.0 (every op is an IPC round trip) in
# another. In-process flopscope hides three things that fail graded submissions:
#   * client/server incompatibilities (e.g. tuple arguments arrive as lists),
#   * the server's per-result array cap (FLOPSCOPE_MAX_ARRAY_BYTES; default 100 MiB, the
#     grader evidently allows ~135 MB: a 134,873,088-byte result passed, 137,682,944 failed),
#   * the op-count-dependent wall / residual time of the client (rfhub, which grades fine,
#     runs at ~145-163 s wall and ~0.82-0.87 s residual per MLP here).
#
#   scripts/grader_emulator.sh submission [n_mlps]
#   scripts/grader_emulator.sh candidates/v5b_rfhub 2
#
# First run creates two virtualenvs under ${EMU_DIR:-/tmp/whest-emu}.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
DIR="${1:?estimator folder}"
N="${2:-2}"
EMU="${EMU_DIR:-/tmp/whest-emu}"
mkdir -p "${EMU}/stub/whestbench"
if [ ! -x "${EMU}/srv/bin/python" ]; then
  python3 -m venv "${EMU}/srv"
  "${EMU}/srv/bin/pip" install -q "flopscope[server]==0.12.0"
fi
if [ ! -x "${EMU}/cli/bin/python" ]; then
  python3 -m venv "${EMU}/cli"
  "${EMU}/cli/bin/pip" install -q "flopscope-client==0.12.0" numpy
fi
# the estimator imports only these three names from whestbench (the full package needs numpy +
# datasets, which the graded client container does not have either)
cat > "${EMU}/stub/whestbench/__init__.py" <<'EOF'
class BaseEstimator:
    def setup(self, ctx):
        pass


class SetupContext:
    pass
EOF
cat > "${EMU}/stub/whestbench/domain.py" <<'EOF'
class MLP:
    def __init__(self, width, depth, weights, seed=0):
        self.width, self.depth, self.weights, self.seed = width, depth, weights, seed
EOF
cat > "${EMU}/drive.py" <<'EOF'
import importlib.util, os, sys, time, traceback
sys.path.insert(0, os.environ["EMU_STUB"]); sys.path.insert(0, os.environ["EMU_ROOT"])
import flopscope as flops, flopscope.numpy as fnp
from whestbench.domain import MLP
from fm.mlps import make_weights
d, n = sys.argv[1], int(sys.argv[2])
spec = importlib.util.spec_from_file_location("cand", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["cand"] = mod; spec.loader.exec_module(mod)
class C:
    seed = 0
    submission_dir = d
est = mod.Estimator()
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    est.setup(C())
for rep in range(n):
    t = time.time()
    try:
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            ws = [fnp.asarray(w) for w in make_weights(1000 + rep)]
            est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**41)
        print(f"MLP {rep}: ok  wall {time.time() - t:.1f} s  C/B {ctx.flops_used / 2**41:.4f}  "
              f"residual {ctx.residual_wall_time_s:.3f} s", flush=True)
    except Exception as e:
        print(f"MLP {rep}: FAILED after {time.time() - t:.1f} s: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(limit=8)
        sys.exit(1)
EOF
SOCK="${EMU}/fs.sock"
rm -f "${SOCK}"
FLOPSCOPE_MAX_ARRAY_BYTES="${MAX_ARRAY_BYTES:-136000000}" \
  "${EMU}/srv/bin/python" -m flopscope_server --url "ipc://${SOCK}" --timeout 3600 > "${EMU}/server.log" 2>&1 &
SRV=$!
trap 'kill ${SRV} 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do [ -S "${SOCK}" ] && break; sleep 1; done
FLOPSCOPE_SERVER_URL="ipc://${SOCK}" EMU_STUB="${EMU}/stub" EMU_ROOT="${ROOT}" \
  "${EMU}/cli/bin/python" "${EMU}/drive.py" "${ROOT}/${DIR}" "${N}"
