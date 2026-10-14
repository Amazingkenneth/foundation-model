"""Parity of the torch training chain and the flopscope submission chain.

Runs both backends on the same MLP, without the operator and with a randomly initialized
operator (all ten networks, output layers scaled up so every correction is active), and
reports the maximum relative deviation of the per-layer means.

    python scripts/parity.py [--seed 1000] [--big]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "submission"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--scale", type=float, default=10.0, help="output-layer scale of the random operator")
    args = ap.parse_args()
    import torch
    import flopscope as flops
    import flopscope.numpy as fnp

    import fm_core
    from backend_fnp import FnpBackend
    import estimator as est_mod
    from fm.mlps import make_weights
    from fm.model import DEFAULT_ACTIVE, OperatorParams, make_chain, make_operator

    torch.set_num_threads(4)
    torch.manual_seed(0)
    w_np = make_weights(args.seed)
    tw = [torch.from_numpy(w) for w in w_np]
    fw = [fnp.asarray(w) for w in w_np]
    B, chain = make_chain()
    params = OperatorParams(seed=3)
    with torch.no_grad():
        for nm in params.nets:
            params.p[f"{nm}__W3"].copy_(torch.randn_like(params.p[f"{nm}__W3"]) * 1e-3 * args.scale)
        params.xstd.fill_(1.0)
    ok = True
    for use_op in (False, True):
        with torch.no_grad():
            op = make_operator(B, params, DEFAULT_ACTIVE) if use_op else None
            rows_t, _ = chain.run(tw, hooks=op)
        rt = np.stack([r.numpy() for r in rows_t]).astype(np.float64)
        FB = FnpBackend()
        fchain = fm_core.Chain(FB, fm_core.Consts(FB), 1024, 16)
        fop = None
        if use_op:
            path = os.path.join(tempfile.mkdtemp(), "w.npz")
            params.export(path, DEFAULT_ACTIVE)
            p, xm, xs, act, nets = est_mod.load_operator_params(path)
            fop = fm_core.Operator(FB, p, xm, xs, act, nets)
        with flops.BudgetContext(flop_budget=2 ** 41, quiet=True) as ctx:
            rows_f, _ = fchain.run(fw, hooks=fop)
        rf = np.stack([np.asarray(r) for r in rows_f]).astype(np.float64)
        rel = np.abs(rt - rf).max(axis=1) / np.abs(rt).max(axis=1)
        mse = ((rt - rf) ** 2).mean(axis=1)
        print(f"operator={'on' if use_op else 'off'}  C/B={ctx.flops_used / 2 ** 41:.4f}")
        print("  max rel dev per layer:", " ".join(f"{v:.1e}" for v in rel))
        print("  mse(torch, fnp) per layer:", " ".join(f"{v:.1e}" for v in mse))
        if rel.max() > 1e-4 or mse[-1] > 1e-10:
            ok = False
    # a random operator is only a meaningful parity probe while the chain stays in its
    # regime (a blown-up chain is chaotic in float32); report it
    if use_op and not np.all(np.isfinite(rt)):
        print("  random operator drove the chain non-finite: lower --scale")
    if use_op:
        # how much the random operator moved the output (the corrections are really applied)
        with torch.no_grad():
            r0, _ = chain.run(tw, hooks=None)
        r0 = np.stack([r.numpy() for r in r0]).astype(np.float64)
        print("  operator effect, mse(op, no-op) final layer: %.2e" % ((rt[-1] - r0[-1]) ** 2).mean())
    print("PARITY", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
