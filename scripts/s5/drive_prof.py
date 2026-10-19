import importlib.util, os, sys, time, cProfile, pstats, io
sys.path.insert(0, os.environ["EMU_STUB"]); sys.path.insert(0, os.environ["EMU_ROOT"])
import flopscope as flops, flopscope.numpy as fnp
from whestbench.domain import MLP
from fm.mlps import make_weights
d = sys.argv[1]
sys.path.insert(0, d)
spec = importlib.util.spec_from_file_location("cand", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["cand"] = mod; spec.loader.exec_module(mod)
class C:
    seed = 0
    submission_dir = d
est = mod.Estimator()
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    est.setup(C())
ws = None
for rep in range(2):
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        ws = [fnp.asarray(w) for w in make_weights(1000 + rep)]
        pr = cProfile.Profile() if rep == 1 else None
        t = time.time()
        if pr: pr.enable()
        est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**41)
        if pr: pr.disable()
    print(f"MLP {rep}: wall {time.time() - t:.1f} s residual {ctx.residual_wall_time_s:.3f} s C/B {ctx.flops_used / 2**41:.4f}", flush=True)
st = pstats.Stats(pr); st.sort_stats("tottime")
s = io.StringIO(); st.stream = s; st.print_stats(30); print(s.getvalue()[:7000])
