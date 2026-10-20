"""prof_fn.py <candidate dir> [seed]: one in-process prediction (1024 x 16, fm.mlps.make_weights(seed)) with every
billed flopscope op attributed to the innermost estimator.py function and source line. Prints C/B, the share of C by
function and by op name, and the top call sites; pickles the raw table next to the candidate as prof_<seed>.pkl."""
import os, sys, importlib.util, collections, pickle
sys.path.insert(0, "/home/user/foundation-model")
import flopscope as flops, flopscope.numpy as fnp
from flopscope._budget import BudgetContext
from whestbench.domain import MLP
from fm.mlps import make_weights
d = sys.argv[1]; sd = int(sys.argv[2]) if len(sys.argv) > 2 else 1
spec = importlib.util.spec_from_file_location("e", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["e"] = mod; spec.loader.exec_module(mod)
class C: seed = 0; submission_dir = d
est = mod.Estimator(); est.setup(C())
agg = collections.defaultdict(lambda: [0, 0])   # (function, line, op) -> [flops, calls]
orig = BudgetContext._append_op_record
def rec(self, r, *a, **k):
    f = sys._getframe(1); key = None
    while f is not None:
        if f.f_code.co_filename.endswith("estimator.py"):
            key = (f.f_code.co_name, f.f_lineno); break
        f = f.f_back
    key = (key or ("?", 0)) + (r.op_name,)
    agg[key][0] += r.flop_cost; agg[key][1] += 1
    return orig(self, r, *a, **k)
BudgetContext._append_op_record = rec
ws = [fnp.asarray(w) for w in make_weights(sd)]
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**41)
tot = ctx.flops_used
print("C/B %.5f  total %.4e FLOPs" % (tot / 2**41, tot))
for title, idx in (("by function", 0), ("by op", 2)):
    s = collections.defaultdict(lambda: [0, 0])
    for k, (fl, c) in agg.items(): s[k[idx]][0] += fl; s[k[idx]][1] += c
    print(f"--- {title} (share of C, calls)")
    for k, (fl, c) in sorted(s.items(), key=lambda x: -x[1][0])[:14]:
        print("  %-24s %6.2f%%  %7d" % (k, 100.0 * fl / tot, c))
print("--- top call sites (function:line op)")
for k, (fl, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:25]:
    print("  %-34s %6.2f%%  %6d" % ("%s:%d %s" % k, 100.0 * fl / tot, c))
pickle.dump(dict(agg), open(os.path.join("/tmp", f"prof_{os.path.basename(os.path.normpath(d))}_{sd}.pkl"), "wb"))
