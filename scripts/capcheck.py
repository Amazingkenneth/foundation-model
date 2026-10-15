"""Largest array the submission allocates during predict() (the grader rejects arrays over
100 MiB), plus C/B and residual time per call.

    python scripts/capcheck.py [--reps 2] [--config submission/fm_config.json]

Every new flopscope array passes through FlopscopeArray.__array_finalize__; the root buffer
of each is measured and the largest few are reported with the stack that made them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SUB = os.path.join(ROOT, "submission")
sys.path.insert(0, ROOT)
sys.path.insert(0, SUB)

CAP = 100 * 2 ** 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--config", default=None, help="fm_config.json to use instead of submission/'s")
    args = ap.parse_args()
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from flopscope._ndarray import FlopscopeArray
    from whestbench.domain import MLP

    import backend_fnp
    import estimator as est_mod
    from fm.mlps import make_weights

    seen = {}
    orig = getattr(FlopscopeArray, "__array_finalize__", None)

    def fin(self, obj):
        if orig is not None:
            orig(self, obj)
        r = self
        while isinstance(r, np.ndarray) and isinstance(r.base, np.ndarray):
            r = r.base
        nb = r.nbytes if isinstance(r, np.ndarray) else 0
        if nb > CAP * 0.5:
            k = (nb, tuple(r.shape))
            if k not in seen:
                seen[k] = "".join(traceback.format_stack(limit=8)[:-2])

    FlopscopeArray.__array_finalize__ = fin

    class Ctx:
        seed = 0
        submission_dir = SUB

    est = est_mod.Estimator()
    est.setup(Ctx())
    if args.config:
        cfg = json.load(open(args.config))
        est._levels.update(cfg.get("levels", {}))
        est._first.update(cfg.get("first_levels", {}))
        est._B.compact.update(cfg.get("compact", {}))
        backend_fnp.FUSE = bool(cfg.get("fuse", backend_fnp.FUSE))
        backend_fnp.STRASSEN_MIN = int(cfg.get("strassen_min", backend_fnp.STRASSEN_MIN))
    for rep in range(args.reps):
        ws = [fnp.asarray(w) for w in make_weights(1000 + rep)]
        with flops.BudgetContext(flop_budget=2 ** 41, quiet=True) as ctx:
            est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2 ** 41)
        print("call %d: C/B %.4f residual %.3f s" % (rep, ctx.flops_used / 2 ** 41, ctx.residual_wall_time_s))
    big = sorted(seen.items(), reverse=True)
    for (nb, sh), st in big[:4]:
        print("%.1f MiB %s%s" % (nb / 2 ** 20, sh, "  OVER THE CAP" if nb > CAP else ""))
        print(st)
    mx = big[0][0][0] if big else 0
    print("largest array: %.1f MiB (cap 100 MiB)" % (mx / 2 ** 20))
    sys.exit(1 if mx > CAP else 0)


if __name__ == "__main__":
    main()
