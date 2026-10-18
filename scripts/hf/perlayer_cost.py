import os, sys, importlib.util, numpy as np
sys.path.insert(0, "/home/user/foundation-model")
import flopscope as flops, flopscope.numpy as fnp
from whestbench.domain import MLP
from fm.mlps import make_weights
d = sys.argv[1]
spec = importlib.util.spec_from_file_location("e", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["e"] = mod; spec.loader.exec_module(mod)
class C: seed = 0; submission_dir = d
est = mod.Estimator(); est.setup(C())
ws = [fnp.asarray(w) for w in make_weights(1)]
marks = []
class WL(list):
    def __iter__(self):
        for w in list.__iter__(self):
            marks.append(CTX[0].flops_used); yield w
CTX = [None]
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    CTX[0] = ctx
    p = est.predict(MLP(width=1024, depth=16, weights=WL(ws), seed=0), 2**41)
marks.append(ctx.flops_used)
per = np.diff(marks) / 2**41 * 100
print("total C/B %.4f" % (ctx.flops_used / 2**41))
print("per-layer %B (cost of layer li = linear+nonlinear of li):")
print(" ".join("%d:%.2f" % (i, v) for i, v in enumerate(per)))
