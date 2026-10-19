# cProfile of one prediction (in-process 0.12.3, one thread): where the wall goes
import os, sys, time, importlib.util, numpy as np, cProfile, pstats, io
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
pr = cProfile.Profile()
t0 = time.time()
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    pr.enable(); p = est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**41); pr.disable()
print("wall %.1f C/B %.5f" % (time.time() - t0, ctx.flops_used / 2**41))
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(25); print(s.getvalue()[:6000])
