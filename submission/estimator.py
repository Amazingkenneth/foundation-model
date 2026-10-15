"""A learned foundation model for the ReLU step (ARC White-Box Estimation Challenge 2026).

Each layer is a fixed linear step (exact third-order transport of a factorized K=3 cumulant
chain) followed by the learned ReLU operator F_theta = F_an((u, v) + Delta_in) + Delta_out:
small MLPs shared by all neurons, pairs, layers and networks correct the incomplete inputs
of the analytic Hermite step and adjust its outputs. See README.md and fm_core.py.

The submission is pure stdlib + flopscope. The operator weights (fm_weights.npz) are loaded
in setup(); without them the estimator runs the analytic chain alone.
"""

from __future__ import annotations

import gc as _gc
import json
import os
import sys

import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import fm_core  # noqa: E402
import backend_fnp  # noqa: E402
from backend_fnp import F32, FnpBackend  # noqa: E402

SUITE = (1024, 16)
WEIGHTS = "fm_weights.npz"
CONFIG = "fm_config.json"   # optional: {"levels": {...}, "first_levels": {...}, "compact": {...}}


def load_operator_params(path):
    """npz -> (params dict, xmean, xstd, active layers, nets). Free (fnp.load is unbilled)."""
    z = fnp.load(str(path))
    keys = list(z.keys()) if hasattr(z, "keys") else list(z.files)
    params = {}
    for nm in fm_core.NETS:
        if f"{nm}.W1" in keys:
            params[nm] = [fnp.asarray(z[f"{nm}.{p}"], dtype=F32)
                          for p in ("W1", "b1", "W2", "b2", "W3", "b3")]
    xmean = fnp.asarray(z["xmean"], dtype=F32)
    xstd = fnp.asarray(z["xstd"], dtype=F32)
    active = [int(v) for v in fnp.asarray(z["active"]).tolist()]
    return params, xmean, xstd, active, sorted(params)


V5B_K34 = "k34.npz"
V5B_MP = "mp.npz"


def load_v5b_params(k34_path, mp_path):
    z = fnp.load(str(k34_path))
    k34 = {k: fnp.asarray(z[k], dtype=F32) for k in ("mu", "sd", "W0", "b0", "W1", "b1", "W2", "b2")}
    mp = None
    if os.path.exists(mp_path):
        z = fnp.load(str(mp_path))
        mp = {f"{a}_{w}{i}": fnp.asarray(z[f"{a}_{w}{i}"], dtype=F32)
              for a in "ABO" for w in "Wb" for i in range(3)}
    return k34, mp


def mean_propagation(mlp):
    """Fallback for non-suite shapes: per-neuron Gaussian mean/variance propagation."""
    n = mlp.width
    mu = fnp.zeros(n, dtype=F32)
    var = fnp.ones(n, dtype=F32)
    rows = []
    import flopscope as flops
    for w in mlp.weights:
        w32 = w if w.dtype == F32 else w.astype(F32)
        m = mu @ w32
        v = fnp.maximum(var @ (w32 * w32), 1e-12)
        s = fnp.sqrt(v)
        a = m / s
        phi = flops.stats.norm.pdf(a).astype(F32)
        Phi = flops.stats.norm.cdf(a).astype(F32)
        mu = s * phi + m * Phi
        e2 = (m * m + v) * Phi + m * s * phi
        var = fnp.maximum(e2 - mu * mu, 1e-12)
        rows.append(mu)
    return fnp.stack(rows, axis=0)


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._B = None
        self._consts = None
        self._op = None
        self._ncall = 0
        self._levels = dict(backend_fnp.LEVELS)
        self._first = dict(backend_fnp.FIRST_LEVELS)

    def setup(self, ctx: SetupContext) -> None:
        self._B = FnpBackend()
        self._consts = fm_core.Consts(self._B)
        sub = getattr(ctx, "submission_dir", None)
        for d in ([str(sub)] if sub is not None else []) + [_HERE]:
            p = os.path.join(d, CONFIG)
            if os.path.exists(p):
                with open(p) as f:
                    cfg = json.load(f)
                self._levels.update(cfg.get("levels", {}))
                self._first.update(cfg.get("first_levels", {}))
                self._B.compact.update(cfg.get("compact", {}))
                backend_fnp.FUSE = bool(cfg.get("fuse", backend_fnp.FUSE))
                break
        cand = [os.path.join(str(sub), WEIGHTS)] if sub is not None else []
        cand.append(os.path.join(_HERE, WEIGHTS))
        if os.environ.get("FM_NO_OPERATOR", "0") == "1":
            return
        for p in cand:
            if os.path.exists(p):
                params, xmean, xstd, active, nets = load_operator_params(p)
                self._op = fm_core.Operator(self._B, params, xmean, xstd, active, nets)
                return
        # the team's trained v5b input-correction operator (k34.npz + mp.npz)
        for d in ([str(sub)] if sub is not None else []) + [_HERE]:
            pk = os.path.join(d, V5B_K34)
            if os.path.exists(pk):
                k34, mp = load_v5b_params(pk, os.path.join(d, V5B_MP))
                self._op = fm_core.OperatorV5B(self._B, k34, mp)
                return

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        if self._B is None:
            self._B = FnpBackend()
            self._consts = fm_core.Consts(self._B)
        if (mlp.width, mlp.depth) != SUITE:
            return mean_propagation(mlp)
        self._B.levels = dict(self._first if self._ncall == 0 else self._levels)
        self._ncall += 1
        was = _gc.isenabled()
        _gc.disable()
        try:
            ws = [w if w.dtype == F32 else w.astype(F32) for w in mlp.weights]
            chain = fm_core.Chain(self._B, self._consts, mlp.width, mlp.depth)
            rows, _ = chain.run(ws, hooks=self._op)
            return fnp.stack(rows, axis=0)
        finally:
            if was:
                _gc.enable()
