# per-call-site flop profile: python prof.py <dir> <seed>
import os, sys, importlib.util, numpy as np, collections, pickle
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
agg = collections.defaultdict(lambda: [0, 0])   # (est line, strassen line, op) -> flops, calls
orig = BudgetContext._append_op_record
def rec(self, r, *a, **k):
    f = sys._getframe(1); el = None; sl = None
    while f is not None:
        fn = f.f_code.co_filename
        if fn.endswith("estimator.py") and el is None: el = f.f_lineno
        if fn.endswith("strassen2.py") and sl is None: sl = f.f_lineno
        f = f.f_back
    key = (el, sl, r.op_name); agg[key][0] += r.flop_cost; agg[key][1] += 1
    return orig(self, r, *a, **k)
BudgetContext._append_op_record = rec
ws = [fnp.asarray(w) for w in make_weights(sd)]
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    p = est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**41)
tot = ctx.flops_used
print("C/B %.4f" % (tot / 2**41))
pickle.dump(dict(agg), open(f"/tmp/claude-0/sc/prof/agg_{os.path.basename(d)}_{sd}.pkl", "wb"))
byline = collections.defaultdict(int)
for (el, sl, op), (fl, n) in agg.items(): byline[el] += fl
for el, fl in sorted(byline.items(), key=lambda x: -x[1])[:45]:
    print("%6s %6.2f%%" % (el, 100 * fl / tot))
byop = collections.defaultdict(int)
for (el, sl, op), (fl, n) in agg.items(): byop[op] += fl
print({k: round(100 * v / tot, 2) for k, v in sorted(byop.items(), key=lambda x: -x[1])[:12]})
