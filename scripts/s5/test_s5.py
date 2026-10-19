import sys, time, numpy as np
sys.path.insert(0, "/tmp/claude-0/sc/s5")
import flopscope as flops, flopscope.numpy as fnp
import strassen4 as S4, strassen5 as S5
rng = np.random.default_rng(0)
def run(mod, X, R, lev, out, ksum, tree, mb, tw):
    with flops.BudgetContext(flop_budget=2**45, quiet=True) as c:
        t0 = time.perf_counter()
        Xa = mod.LeftTree(X, lev, tw, "T", mb) if tree else X
        mod.mm(Xa, R, lev, out, ksum=ksum, mb=mb)
        dt = time.perf_counter() - t0
    return np.asarray(out).astype(np.float64), c.flops_used, dt
cases = [  # (B, m, k, w, lev, ksum, tree, mb)
    (1, 1024, 1024, 1024, 5, False, True, None),
    (2, 1024, 1024, 1024, 5, False, True, None),
    (2, 1024, 1024, 768, 5, False, True, None),
    (2, 1024, 960, 1024, 5, False, True, 24),
    (2, 1024, 1024, 1024, 5, False, False, None),
    (2, 1024, 512, 1024, 5, True, False, None),
    (2, 1024, 1024, 1024, 4, True, False, 20),
    (3, 512, 1024, 512, 3, False, False, None),
    (2, 1024, 1024, 1024, 1, False, True, None),
    (2, 1024, 1024, 1024, 2, True, False, None),
]
for (B, m, k, w, lev, ksum, tree, mb) in cases:
    X = fnp.asarray(rng.standard_normal((1 if tree else B, m, k)).astype(np.float32))
    Rr = rng.standard_normal((B, w, k) if ksum else (B, k, w)).astype(np.float32)
    R = fnp.swapaxes(fnp.asarray(Rr), -1, -2) if ksum else fnp.asarray(Rr)
    outs = []
    for mod in (S4, S5):
        o = fnp.zeros((1, m, w) if ksum else (B, m, w), dtype=fnp.float32)
        run(mod, X, R, lev, o, ksum, tree, mb, w)       # warm (pool)
        o2 = fnp.zeros((1, m, w) if ksum else (B, m, w), dtype=fnp.float32)
        outs.append(run(mod, X, R, lev, o2, ksum, tree, mb, w))
    ref = np.einsum("bmk,bkw->mw" if ksum else "bmk,bkw->bmw", np.asarray(X).astype(np.float64) if not tree else np.broadcast_to(np.asarray(X).astype(np.float64), (B, m, k)), np.asarray(R).astype(np.float64))
    (a, fa, ta), (b, fb, tb) = outs
    sc = np.abs(ref).max()
    print(f"B{B} {m}x{k}x{w} lev{lev} ksum{int(ksum)} tree{int(tree)} mb{mb}: |s4-s5| {np.abs(a-b).max()/sc:.1e} |s5-ref| {np.abs(b.reshape(ref.shape)-ref).max()/sc:.1e} "
          f"bill s5/s4 {fb/fa:.4f} time s4 {ta*1e3:.0f} ms s5 {tb*1e3:.0f} ms")
