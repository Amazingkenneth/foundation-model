"""Evaluate the submission estimator under flopscope on local Monte Carlo ground truth.

Reports per MLP: metered FLOPs / budget, residual wall time, final-layer MSE against the
MC means, and the noise-corrected MSE (MSE minus the MC sampling variance of the targets,
an unbiased estimate of the MSE against the exact means).

    python scripts/eval_fnp.py --split val [--limit 8] [--no-operator] [--weights PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "submission"))

BUDGET = 2 ** 41


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-operator", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--levels", default=None, help="family,hub,form,inner,sym Strassen depths")
    args = ap.parse_args()
    if args.no_operator:
        os.environ["FM_NO_OPERATOR"] = "1"
    import flopscope as flops
    import flopscope.numpy as fnp
    from whestbench.domain import MLP

    import estimator as est_mod
    import backend_fnp
    from fm.mlps import load_split, make_weights

    if args.levels is not None:
        a = [int(x) for x in args.levels.split(",")]
        backend_fnp.LEVELS.update(family=a[0], hub=a[1], form=a[2], inner=a[3], sym=a[4])
    if args.weights:
        est_mod.WEIGHTS = os.path.abspath(args.weights)

    class Ctx:
        seed = 0
        submission_dir = os.path.join(ROOT, "submission")

    est = est_mod.Estimator()
    est.setup(Ctx())
    if args.weights:
        params, xmean, xstd, active, nets = est_mod.load_operator_params(args.weights)
        import fm_core
        est._op = fm_core.Operator(est._B, params, xmean, xstd, active, nets)
    print("operator:", "off" if est._op is None else sorted(est._op.nets), flush=True)
    data = load_split(args.split)
    if args.limit:
        data = data[:args.limit]
    res = []
    for d in data:
        ws = [fnp.asarray(w) for w in make_weights(d["seed"])]
        mlp = MLP(width=1024, depth=16, weights=ws, seed=d["seed"])
        t0 = time.time()
        with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
            pred = est.predict(mlp, BUDGET)
        wall = time.time() - t0
        p = np.asarray(pred, dtype=np.float64)
        err = p - d["means"]
        mse_l = (err ** 2).mean(axis=1)
        mse = mse_l[-1]
        corr = mse - d["noise_last"].mean()
        res.append((ctx.flops_used / BUDGET, ctx.residual_wall_time_s, mse, corr, mse_l))
        print(f"seed {d['seed']}: C/B {ctx.flops_used / BUDGET:.4f}  residual {ctx.residual_wall_time_s:.3f}s "
              f"wall {wall:.1f}s  mse_L {mse:.4e}  noise-corrected {corr:.4e}", flush=True)
    a = np.array([r[:4] for r in res])
    print(f"MEAN over {len(res)}: C/B {a[:, 0].mean():.4f}  residual {a[:, 1].mean():.3f}s (max {a[:, 1].max():.3f})  "
          f"mse_L {a[:, 2].mean():.4e}  noise-corrected {a[:, 3].mean():.4e} +- {a[:, 3].std() / max(1, len(res) - 1) ** 0.5:.1e}")
    print("per-layer raw mse:", " ".join(f"{v:.2e}" for v in np.mean([r[4] for r in res], axis=0)))


if __name__ == "__main__":
    main()
