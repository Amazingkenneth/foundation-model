# flops and backend seconds by outer call site (first estimator.py frame inside _predict_core/_dslices/helpers)
import os, sys, importlib.util, numpy as np, collections, pickle, time
sys.path.insert(0, "/home/user/foundation-model")
import flopscope as flops, flopscope.numpy as fnp
from flopscope._budget import BudgetContext
from whestbench.domain import MLP
from fm.mlps import make_weights
d, sd = sys.argv[1], int(sys.argv[2])
spec = importlib.util.spec_from_file_location("e", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["e"] = mod; spec.loader.exec_module(mod)
class C: seed = 0; submission_dir = d
est = mod.Estimator(); est.setup(C())
OUTER = ("_predict_core", "_dslices", "_k34_step", "_mp_step", "_star", "_mlp3")
agg = collections.defaultdict(lambda: [0, 0.0, 0])
orig = BudgetContext._append_op_record
def rec(self, r, *a, **k):
    f = sys._getframe(1); key = None
    while f is not None:
        if f.f_code.co_filename.endswith("estimator.py") and f.f_code.co_name in OUTER:
            key = (f.f_code.co_name, f.f_lineno); break
        f = f.f_back
    g = agg[key]; g[0] += r.flop_cost; g[1] += (r.flopscope_backend_duration_s or 0.0); g[2] += 1
    return orig(self, r, *a, **k)
BudgetContext._append_op_record = rec
ws = [fnp.asarray(w) for w in make_weights(sd)]
for rep in range(2):   # warm pass, then the measured pass
    agg.clear(); t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        p = est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**41)
    wall = time.time() - t0
tf = sum(v[0] for v in agg.values()); tt = sum(v[1] for v in agg.values())
print("C/B %.4f wall %.1f s backend %.1f s ops %d" % (ctx.flops_used / 2**41, wall, tt, sum(v[2] for v in agg.values())))
for key, (fl, t, n) in sorted(agg.items(), key=lambda x: -x[1][1])[:30]:
    print("%-28s flops %5.1f%%  backend %5.1f%%  calls %d" % (key, 100 * fl / tf, 100 * t / tt, n))
