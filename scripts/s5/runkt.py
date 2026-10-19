# wall split inside the Strassen kernel: fused leaf vs batched combos / assembly vs leaf matmuls, one prediction
import os, sys, time, importlib.util, numpy as np
sys.path.insert(0, "/home/user/foundation-model")
d = sys.argv[1]; sys.path.insert(0, d)
import strassen4 as S
import flopscope as flops, flopscope.numpy as fnp
T = {}
def wrap(name):
    f = getattr(S, name)
    def g(*a, **k):
        t0 = time.perf_counter(); r = f(*a, **k); T[name] = T.get(name, 0.0) + time.perf_counter() - t0; return r
    setattr(S, name, g)
for nm in ("_leaf_fused", "_combos_L", "_combos_R", "_assemble"):
    wrap(nm)
_mm0 = fnp.matmul
from whestbench.domain import MLP
from fm.mlps import make_weights
spec = importlib.util.spec_from_file_location("e", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["e"] = mod; spec.loader.exec_module(mod)
class C: seed = 0; submission_dir = d
est = mod.Estimator(); est.setup(C())
ws = [fnp.asarray(w) for w in make_weights(1)]
t0 = time.time()
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    p = est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**41)
print("wall %.1f C/B %.5f" % (time.time() - t0, ctx.flops_used / 2**41), {k: round(v, 1) for k, v in T.items()})
