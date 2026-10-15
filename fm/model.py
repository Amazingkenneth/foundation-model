"""torch side of the learned operator: parameters, initialization, export, chain wrapper."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SUB = os.path.join(ROOT, "submission")
if SUB not in sys.path:
    sys.path.insert(0, SUB)

import fm_core  # noqa: E402

from .backend_torch import TorchBackend  # noqa: E402

# Layers (0-based) where steps 1-7 run: the paper's "v5b (layers 9-15)" = 1-based 9..15.
DEFAULT_ACTIVE = list(range(8, 15))


class OperatorParams(torch.nn.Module):
    def __init__(self, nets=tuple(fm_core.NETS), seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.nets = list(nets)
        self.p = torch.nn.ParameterDict()
        H = fm_core.HID
        for nm in self.nets:
            i, o = fm_core.NETS[nm]
            self._add(nm, i, o, g)
        self.register_buffer("xmean", torch.zeros(14))
        self.register_buffer("xstd", torch.ones(14))

    # output-layer init: corrections start at exactly zero (F_theta = F_an at init); the
    # message nets produce features, so they start with a normal init; of each bilinear
    # pair (l, r) one side starts small and random so that the other gets a gradient
    OUT_INIT = {"msg1": (64 ** -0.5), "msg2": (64 ** -0.5), "pairL": 0.1, "covL": 0.1}

    def _add(self, nm, i, o, g, zero_out=None):
        H = fm_core.HID

        def lin(a, b, scale):
            return torch.nn.Parameter(torch.randn(a, b, generator=g) * scale)

        self.p[f"{nm}__W1"] = lin(i, H, (1.0 / i) ** 0.5)
        self.p[f"{nm}__b1"] = torch.nn.Parameter(torch.zeros(H))
        self.p[f"{nm}__W2"] = lin(H, H, (1.0 / H) ** 0.5)
        self.p[f"{nm}__b2"] = torch.nn.Parameter(torch.zeros(H))
        sc = self.OUT_INIT.get(nm, 0.0)
        self.p[f"{nm}__W3"] = torch.nn.Parameter(torch.randn(H, o, generator=g) * sc)
        self.p[f"{nm}__b3"] = torch.nn.Parameter(torch.zeros(o))

    def add_stage2(self, seed=1):
        g = torch.Generator().manual_seed(seed)
        for nm in fm_core.STAGE2:
            if nm not in self.nets:
                i, o = fm_core.NETS[nm]
                self._add(nm, i, o, g)
                self.nets.append(nm)

    def param_dict(self):
        return {nm: [self.p[f"{nm}__{k}"] for k in ("W1", "b1", "W2", "b2", "W3", "b3")]
                for nm in self.nets}

    def export(self, path, active):
        arrs = {"xmean": self.xmean.numpy().astype(np.float32),
                "xstd": self.xstd.numpy().astype(np.float32),
                "active": np.asarray(active, dtype=np.int64)}
        for nm in self.nets:
            for k in ("W1", "b1", "W2", "b2", "W3", "b3"):
                arrs[f"{nm}.{k}"] = self.p[f"{nm}__{k}"].detach().numpy().astype(np.float32)
        np.savez(path, **arrs)


def make_chain(n=1024, L=16):
    B = TorchBackend()
    return B, fm_core.Chain(B, fm_core.Consts(B), n, L)


def make_operator(B, params: OperatorParams, active=DEFAULT_ACTIVE):
    return fm_core.Operator(B, params.param_dict(), params.xmean, params.xstd, active, params.nets)


def torch_weights(seed):
    from .mlps import make_weights
    return [torch.from_numpy(w) for w in make_weights(seed)]


# ---- the team's v5b operator (fm_core.OperatorV5B): k34.npz (f_self) + mp.npz (messages) ----
V5B_K34_TRAIN = ("W0", "b0", "W1", "b1", "W2", "b2")
V5B_MP_KEYS = tuple(f"{a}_{w}{i}" for a in "ABO" for w in "Wb" for i in range(3))


class V5BParams(torch.nn.Module):
    """Trainable v5b weights, initialized from submission/k34.npz + mp.npz; the feature
    standardization (mu, sd) stays fixed."""

    def __init__(self, k34_path, mp_path=None):
        super().__init__()
        z = np.load(k34_path)
        self.register_buffer("mu", torch.from_numpy(z["mu"].astype(np.float32)))
        self.register_buffer("sd", torch.from_numpy(z["sd"].astype(np.float32)))
        self.k34 = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(torch.from_numpy(z[k].astype(np.float32))) for k in V5B_K34_TRAIN})
        self.mp = None
        if mp_path is not None and os.path.exists(mp_path):
            z = np.load(mp_path)
            self.mp = torch.nn.ParameterDict(
                {k: torch.nn.Parameter(torch.from_numpy(z[k].astype(np.float32))) for k in V5B_MP_KEYS})

    def operator(self, B):
        k34 = dict(self.k34.items(), mu=self.mu, sd=self.sd)
        return fm_core.OperatorV5B(B, k34, None if self.mp is None else dict(self.mp.items()))

    def export(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        k = {n: v.detach().numpy().astype(np.float32) for n, v in self.k34.items()}
        k.update(mu=self.mu.numpy(), sd=self.sd.numpy())
        np.savez(os.path.join(out_dir, "k34.npz"), **k)
        if self.mp is not None:
            np.savez(os.path.join(out_dir, "mp.npz"),
                     **{n: v.detach().numpy().astype(np.float32) for n, v in self.mp.items()})
