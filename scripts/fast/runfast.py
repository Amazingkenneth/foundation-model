"""runfast.py <dir> <tag> <seeds>: local accuracy screening only. Same as run.py, but every Strassen call
(strassen2.mm / LeftTree) is replaced by a plain batched matmul -- identical math, different float32
rounding, several times faster. Costs are NOT measured here (C/B printed is meaningless)."""
import os, sys, importlib.util, numpy as np
sys.path.insert(0, "/home/user/foundation-model")
import flopscope as flops, flopscope.numpy as fnp
from whestbench.domain import MLP
from fm.mlps import make_weights
d, tag = sys.argv[1], sys.argv[2]; seeds = [int(s) for s in sys.argv[3].split(",")]
sspec = importlib.util.spec_from_file_location("strassen2", os.path.join(d, "strassen2.py"))
s2 = importlib.util.module_from_spec(sspec); sys.modules["strassen2"] = s2; sspec.loader.exec_module(s2)
class _LT:
    def __init__(self, X, lev, w, tag="T", mb=None):
        self.X = X; self.lev = 0; self.c = []
def _mm(X, R, lev, out, ksum=False, mb=None):
    Xa = X.X if isinstance(X, _LT) else X
    if ksum:
        fnp.copyto(out, fnp.sum(fnp.matmul(Xa, R), axis=0, keepdims=True))
    else:
        fnp.matmul(Xa, R, out=out)
    return out
s2.LeftTree = _LT; s2.mm = _mm
spec = importlib.util.spec_from_file_location("e", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["e"] = mod; spec.loader.exec_module(mod)
assert mod.LeftTree is _LT and mod._smm is _mm
class C: seed = 0; submission_dir = d
est = mod.Estimator(); est.setup(C())
os.makedirs(f"/tmp/claude-0/acc/{tag}", exist_ok=True)
import time
for sd in seeds:
    out = f"/tmp/claude-0/acc/{tag}/pred_{sd}.npy"
    if os.path.exists(out): continue
    t0 = time.time()
    ws = [fnp.asarray(w) for w in make_weights(sd)]
    with flops.BudgetContext(flop_budget=2**43, quiet=True) as ctx:
        p = np.asarray(est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**43), dtype=np.float64)
    np.save(out, p); print(tag, sd, "%.0fs" % (time.time() - t0), flush=True)
