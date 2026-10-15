"""Fast unit tests: Strassen families, the folded term program, torch/fnp agreement."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "submission"))

import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402

import fm_core  # noqa: E402
from backend_fnp import FnpBackend  # noqa: E402
from fm_tables import TERM_SPECS  # noqa: E402

BIG = 10 ** 14


@pytest.mark.parametrize("compact", [0, 1, 3])
def test_strassen_families(compact):
    rng = np.random.default_rng(compact)
    n = 256
    B = FnpBackend(levels=dict(family=3, hub=3, form=2, inner=2, sym=2))
    B.compact = {k: compact for k in B.compact}
    W = rng.standard_normal((n, n)).astype(np.float32)
    Y = rng.standard_normal((3, n, n)).astype(np.float32)
    X = rng.standard_normal((2, 2, n, n)).astype(np.float32)
    Z = rng.standard_normal((2, 2, 128, n)).astype(np.float32)
    Q = rng.standard_normal((n, 128)).astype(np.float32)
    F = rng.standard_normal((2, 2, 128, n)).astype(np.float32)
    S = rng.standard_normal((n, n)).astype(np.float32)
    with flops.BudgetContext(flop_budget=BIG, quiet=True):
        for _ in range(2):   # second pass runs on the cached views
            fam = np.asarray(B.family(fnp.asarray(W), fnp.asarray(Y), out="f"))
            hub = np.asarray(B.hub(fnp.asarray(X), fnp.asarray(Z), out="h", kind="inner"))
            frm = np.asarray(B.form(fnp.asarray(Q), fnp.asarray(F), out="fm"))
            sym = np.asarray(B.symprod(fnp.asarray(S), fnp.asarray(S.T.copy()), out="s"))
            hy = np.asarray(B.hub(fnp.asarray(X), fnp.asarray(X), out="hy", kind="hub"))

    def rel(a, b):
        return np.abs(a - b).max() / np.abs(b).max()

    assert rel(fam, W @ Y) < 1e-5
    assert rel(hub, np.einsum("bsij,bscj->ic", X, Z)) < 1e-5
    assert rel(frm, Q @ F) < 1e-5
    assert rel(sym, S @ S.T) < 1e-5
    assert rel(hy, np.einsum("bsij,bscj->ic", X, X)) < 1e-5


def _naive_terms(mode, Wall, vec, mats, n):
    """Direct evaluation of TERM_SPECS (d=2 outputs off-diagonal, d=1 outputs)."""
    missing = fm_core.MODE0_MISSING if mode == 0 else set()
    q = vec.get("q")
    sl = {"ones2": np.ones((n, n)), "c_off": mats["c_off"]}
    if mode == 1:
        wk4m = q[:, None] + q[None, :]
        np.fill_diagonal(wk4m, 0.0)
        sl.update(d21=mats["d21"], d21T=mats["d21"].T, wk431=mats["wk431"], wk4m=wk4m,
                  d3row=np.tile(vec["d3"][None, :], (n, 1)), d3col=np.tile(vec["d3"][:, None], (1, n)),
                  g4row=np.tile(vec["g4"][None, :], (n, 1)), g4col=np.tile(vec["g4"][:, None], (1, n)))
    out2 = np.zeros((3, n, n))
    for g, ip in enumerate(fm_core.D2_IPS):
        for a, b, wl, wr, coef in TERM_SPECS[ip]:
            if a in missing or b in missing:
                continue
            out2[g] += coef * sl[a] * sl[b] * np.outer(Wall[:, wl], Wall[:, wr])
    sl1 = {"ones1": np.ones(n)}
    if mode == 1:
        sl1.update(d3=vec["d3"], g4=vec["g4"])
    out1 = np.zeros((4, n))
    for g, ip in enumerate(fm_core.D1_IPS):
        for a, b, wl, wr, coef in TERM_SPECS[ip]:
            if a in missing or b in missing:
                continue
            out1[g] += coef * sl1[a] * sl1[b] * Wall[:, wl]
    for g in range(3):
        np.fill_diagonal(out2[g], 0.0)
    return out1, out2


@pytest.mark.parametrize("mode", [0, 1])
def test_term_program_matches_naive(mode):
    torch = pytest.importorskip("torch")
    from fm.backend_torch import TorchBackend
    n = 12
    rng = np.random.default_rng(1 + mode)
    B = TorchBackend()
    K = fm_core.Consts(B)

    class Dummy(fm_core.Chain):
        def __init__(self):
            self.B, self.K, self.n = B, K, n
            self.ones2 = B.ones((n, n))
            self.ones_n = B.ones((n,))

    ch = Dummy()
    Wall = rng.standard_normal((n, len(fm_core.WICK_PAIRS)))
    vec = {"1": np.ones(n), "d3": rng.standard_normal(n), "g4": rng.standard_normal(n),
           "q": rng.standard_normal(n)}
    mats = {k: rng.standard_normal((n, n)) for k in ("c_off", "d21", "wk431")}
    for m in mats.values():
        np.fill_diagonal(m, 0.0)
    if mode == 0:
        vec = {"1": vec["1"]}
        mats = {"c_off": mats["c_off"]}
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    B.const = lambda data: torch.tensor(data, dtype=torch.float64)
    K = fm_core.Consts(B)
    ch.K = K
    ch.ones2 = torch.ones((n, n), dtype=torch.float64)
    prog = K.progs[mode]
    WT = t(Wall).T
    tv = {k: t(v) for k, v in vec.items()}
    tm = {k: t(v) for k, v in mats.items()}
    if "d21" in tm:
        tm["d21T"] = tm["d21"].T
    pk1 = ch.term1(prog, WT, tv).numpy()
    pk2 = np.stack([p.numpy() for p in ch.term2(prog, WT, tv, tm)])
    for g in range(3):
        np.fill_diagonal(pk2[g], 0.0)
    r1, r2 = _naive_terms(mode, Wall, vec, mats, n)
    assert np.allclose(pk1, r1, rtol=1e-10, atol=1e-10)
    assert np.allclose(pk2, r2, rtol=1e-10, atol=1e-10)


def test_param_count():
    assert fm_core.n_params() == 114427
