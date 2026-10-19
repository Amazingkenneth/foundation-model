# cProfile of one predict, restricted to our code (estimator / strassen): tottime = Python outside callees
import os, sys, time, importlib.util, cProfile, pstats, io
sys.path.insert(0, "/home/user/foundation-model")
import flopscope as flops, flopscope.numpy as fnp
from whestbench.domain import MLP
from fm.mlps import make_weights
d = sys.argv[1]; sys.path.insert(0, d)
spec = importlib.util.spec_from_file_location("e", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["e"] = mod; spec.loader.exec_module(mod)
class C: seed = 0; submission_dir = d
est = mod.Estimator(); est.setup(C())
ws = [fnp.asarray(w) for w in make_weights(1)]
pr = cProfile.Profile()
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    pr.enable(); p = est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**41); pr.disable()
print("residual %.3f" % ctx.residual_wall_time_s)
st = pstats.Stats(pr); st.sort_stats("tottime")
rows = []
for (fn, ln, name), (cc, nc, tt, ct, callers) in st.stats.items():
    if "foundation-model" in fn or "strassen" in fn or fn.startswith("<"):
        rows.append((tt, nc, ct, f"{os.path.basename(fn)}:{ln}({name})"))
rows.sort(reverse=True)
for tt, nc, ct, nm in rows[:25]: print("%7.3f s tot %8d calls %8.2f s cum  %s" % (tt, nc, ct, nm))
# flopscope getitem cost
for (fn, ln, name), (cc, nc, tt, ct, callers) in st.stats.items():
    if name in ("__getitem__",) and "flopscope" in fn:
        print("flopscope __getitem__ %s:%d calls %d cum %.2f s" % (os.path.basename(fn), ln, nc, ct))
