import importlib.util, os, sys, time, traceback, random
sys.path.insert(0, os.environ["EMU_STUB"]); sys.path.insert(0, os.environ["EMU_ROOT"])
import flopscope as flops, flopscope.numpy as fnp
from whestbench.domain import MLP
d = sys.argv[1]
spec = importlib.util.spec_from_file_location("cand", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["cand"] = mod; spec.loader.exec_module(mod)
class C:
    seed = 0
    submission_dir = d
est = mod.Estimator()
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    est.setup(C())
rng = random.Random(0)
for (w, L) in [(4, 2), (8, 3), (16, 4), (32, 4), (64, 6), (128, 8), (100, 5), (256, 4)]:
    try:
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            ws = [fnp.asarray([[rng.gauss(0, (2 / w) ** 0.5) for _ in range(w)] for _ in range(w)], dtype=fnp.float32) for _ in range(L)]
            p = est.predict(MLP(width=w, depth=L, weights=ws, seed=0), 2**41)
            ok = bool(fnp.all(fnp.isfinite(p)))
        print(w, L, "ok", ok, flush=True)
    except Exception as e:
        print(w, L, "FAILED", type(e).__name__, str(e)[:300], flush=True); traceback.print_exc(limit=4)
