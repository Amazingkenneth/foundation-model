# count client-side sync points (bool/float/int/index/item/tolist/array conversion of flopscope arrays) by call site
import os, sys, importlib.util, collections
sys.path.insert(0, "/home/user/foundation-model")
d = sys.argv[1]; sys.path.insert(0, d)
import flopscope as flops, flopscope.numpy as fnp
import flopscope._ndarray as nd
SY = collections.Counter()
def site():
    f = sys._getframe(2)
    while f and ("flopscope" in f.f_code.co_filename or "/numpy/" in f.f_code.co_filename): f = f.f_back
    return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}" if f else "?"
classes = [c for c in vars(nd).values() if isinstance(c, type)]
for c in classes:
    for nm in ("__bool__", "__float__", "__int__", "__index__", "item", "tolist", "__array__", "__len__", "__iter__"):
        if nm in vars(c):
            f0 = vars(c)[nm]
            def w(self, *a, _f0=f0, _nm=nm, **k):
                SY[(_nm, site())] += 1
                return _f0(self, *a, **k)
            setattr(c, nm, w)
from whestbench.domain import MLP
from fm.mlps import make_weights
spec = importlib.util.spec_from_file_location("e", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["e"] = mod; spec.loader.exec_module(mod)
class C: seed = 0; submission_dir = d
est = mod.Estimator(); est.setup(C())
ws = [fnp.asarray(w) for w in make_weights(1)]
SY.clear()
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    p = est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**41)
print("sync-point calls inside predict:", sum(SY.values()))
for (nm, s), v in SY.most_common(40): print("  %6d %-10s %s" % (v, nm, s))
