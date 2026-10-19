# count Strassen mm calls by (batch, levels, ksum) and flopscope ops, one prediction
import os, sys, time, importlib.util, collections, numpy as np
sys.path.insert(0, "/home/user/foundation-model")
d = sys.argv[1]; sys.path.insert(0, d)
kern = sys.argv[2]
S = __import__(kern)
import flopscope as flops, flopscope.numpy as fnp
from flopscope import _budget
CNT = collections.Counter(); OPS = [0]
mm0 = S.mm
def mm(X, R, lev, out, ksum=False, mb=None):
    CNT[(R.shape[0], lev, ksum)] += 1
    return mm0(X, R, lev, out, ksum=ksum, mb=mb)
S.mm = mm
ci0 = _budget._call_numpy_impl
def ci(*a, **k):
    OPS[0] += 1
    return ci0(*a, **k)
_budget._call_numpy_impl = ci
from whestbench.domain import MLP
from fm.mlps import make_weights
spec = importlib.util.spec_from_file_location("e", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["e"] = mod; spec.loader.exec_module(mod)
mod._smm = mm
class C: seed = 0; submission_dir = d
est = mod.Estimator(); est.setup(C())
ws = [fnp.asarray(w) for w in make_weights(1)]
t0 = time.time()
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    p = est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**41)
print("wall %.1f C/B %.5f ops %d mm calls %d" % (time.time() - t0, ctx.flops_used / 2**41, OPS[0], sum(CNT.values())))
for k, v in sorted(CNT.items()): print("  B %d lev %d ksum %d: %d calls" % (k[0], k[1], k[2], v))
