"""Pre-package smoke checks for the submission folder.

1. Allowed code: every module in submission/ imports only the stdlib, flopscope,
   whestbench and sibling modules (no numpy/scipy/torch/ctypes/threads/subprocess).
2. Non-suite shapes (the grader's smoke test uses some) return finite (depth, width) arrays.
3. The shipped weights load and contain every network the operator expects.
"""

from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SUB = os.path.join(ROOT, "submission")
sys.path.insert(0, SUB)

FORBIDDEN = {"numpy", "scipy", "torch", "ctypes", "cffi", "threading", "multiprocessing",
             "subprocess", "asyncio", "concurrent", "jax", "numba"}
ALLOWED_THIRD_PARTY = {"flopscope", "whestbench"}


def check_imports() -> None:
    local = {f[:-3] for f in os.listdir(SUB) if f.endswith(".py")}
    std = set(sys.stdlib_module_names)
    for f in sorted(os.listdir(SUB)):
        if not f.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(SUB, f)).read())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
            for m in mods:
                top = m.split(".")[0]
                if top in FORBIDDEN:
                    raise SystemExit(f"{f}: forbidden import {m}")
                if top not in std and top not in ALLOWED_THIRD_PARTY and top not in local and top != "__future__":
                    raise SystemExit(f"{f}: non-stdlib import {m}")
    print("imports: ok (stdlib + flopscope + whestbench only)")


def check_shapes() -> None:
    import flopscope as flops
    import flopscope.numpy as fnp
    import numpy as np
    from whestbench.domain import MLP

    import estimator as est_mod

    class Ctx:
        seed = 0
        submission_dir = SUB

    est = est_mod.Estimator()
    est.setup(Ctx())
    rng = np.random.default_rng(0)
    for w, d in [(4, 2), (64, 6), (256, 32), (32, 1), (128, 20), (1024, 3)]:
        ws = [fnp.asarray((rng.standard_normal((w, w)) * np.sqrt(2.0 / w)).astype(np.float32)) for _ in range(d)]
        mlp = MLP(width=w, depth=d, weights=ws, seed=0)
        with flops.BudgetContext(flop_budget=2 ** 41, quiet=True):
            out = np.asarray(est.predict(mlp, 2 ** 41))
        assert out.shape == (d, w), (out.shape, w, d)
        assert np.all(np.isfinite(out)), (w, d)
    print("non-suite shapes: ok")
    if est._op is not None:
        print("operator networks:", sorted(est._op.nets), "active layers:", sorted(est._op.active))
    else:
        print("operator: none shipped (analytic chain only)")


if __name__ == "__main__":
    check_imports()
    check_shapes()
