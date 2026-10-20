# count client->server round trips by op name for one prediction (client/server mode)
import importlib.util, os, sys, time, collections, msgpack
sys.path.insert(0, os.environ["EMU_STUB"]); sys.path.insert(0, os.environ["EMU_ROOT"])
import flopscope as flops, flopscope.numpy as fnp
from flopscope import _connection
from whestbench.domain import MLP
from fm.mlps import make_weights
CNT = collections.Counter(); ON = [False]
sr0 = _connection.Connection.send_recv
def sr(self, raw):
    if ON[0]:
        try: CNT[msgpack.unpackb(raw, raw=False).get("op", "?")] += 1
        except Exception: CNT["?"] += 1
    return sr0(self, raw)
_connection.Connection.send_recv = sr
d = sys.argv[1]; sys.path.insert(0, d)
spec = importlib.util.spec_from_file_location("cand", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["cand"] = mod; spec.loader.exec_module(mod)
class C:
    seed = 0
    submission_dir = d
est = mod.Estimator()
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    est.setup(C())
for rep in range(2):
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        ws = [fnp.asarray(w) for w in make_weights(1000 + rep)]
        CNT.clear(); ON[0] = True; t = time.time()
        est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**41)
        ON[0] = False
    print(f"MLP {rep}: wall {time.time() - t:.1f} s residual {ctx.residual_wall_time_s:.3f} s round trips {sum(CNT.values())}", flush=True)
print("by op:", CNT.most_common(25))
