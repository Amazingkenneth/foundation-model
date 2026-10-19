# count flopscope __getitem__ calls and fnp op calls by caller file:line, one prediction
import os, sys, time, importlib.util, collections
sys.path.insert(0, "/home/user/foundation-model")
d = sys.argv[1]; sys.path.insert(0, d)
import flopscope as flops, flopscope.numpy as fnp
from flopscope import _budget
import flopscope._ndarray as nd
SL = collections.Counter(); OP = collections.Counter()
cls = [c for c in vars(nd).values() if isinstance(c, type) and "__getitem__" in vars(c)]
for c in cls:
    g0 = c.__getitem__
    def g(self, key, _g0=g0):
        f = sys._getframe(1); SL[f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}"] += 1
        return _g0(self, key)
    c.__getitem__ = g
ci0 = _budget._call_numpy_impl
def ci(*a, **k):
    f = sys._getframe(1)
    while f and ("flopscope" in f.f_code.co_filename or "numpy" in f.f_code.co_filename): f = f.f_back
    OP[f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}" if f else "?"] += 1
    return ci0(*a, **k)
_budget._call_numpy_impl = ci
from whestbench.domain import MLP
from fm.mlps import make_weights
spec = importlib.util.spec_from_file_location("e", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["e"] = mod; spec.loader.exec_module(mod)
class C: seed = 0; submission_dir = d
est = mod.Estimator(); est.setup(C())
ws = [fnp.asarray(w) for w in make_weights(1)]
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    p = est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**41)
print("slices", sum(SL.values()), "ops", sum(OP.values()), "residual %.3f" % ctx.residual_wall_time_s)
print("top slice sites:"); [print("  %6d %s" % (v, k)) for k, v in SL.most_common(15)]
F = collections.Counter(); [F.__setitem__(k.split(":")[0], F[k.split(":")[0]] + v) for k, v in OP.items()]
print("ops by file:", dict(F))
print("top op sites:"); [print("  %6d %s" % (v, k)) for k, v in OP.most_common(25)]
