# FLOPs by op name from the in-process op log, one prediction
import os, sys, importlib.util, collections
sys.path.insert(0, "/home/user/foundation-model")
d = sys.argv[1]; sys.path.insert(0, d)
import flopscope as flops, flopscope.numpy as fnp
from whestbench.domain import MLP
from fm.mlps import make_weights
spec = importlib.util.spec_from_file_location("e", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["e"] = mod; spec.loader.exec_module(mod)
class C: seed = 0; submission_dir = d
est = mod.Estimator(); est.setup(C())
ws = [fnp.asarray(w) for w in make_weights(1)]
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    p = est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**41)
F = collections.Counter(); N = collections.Counter()
for r in ctx._op_log:
    nm = getattr(r, "op_name", None) or getattr(r, "name", None) or str(type(r))
    fl = getattr(r, "flop_cost", None)
    if fl is None: fl = getattr(r, "flops", 0)
    F[nm] += fl or 0; N[nm] += 1
print("total %.6e  C/B %.5f" % (sum(F.values()), ctx.flops_used / 2**41))
for k, v in F.most_common(14): print("  %-22s %.4e  (%d ops)" % (k, v, N[k]))
