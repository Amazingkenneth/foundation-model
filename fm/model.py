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
            self._add(nm, i, o, g, zero_out=False)
        self.register_buffer("xmean", torch.zeros(14))
        self.register_buffer("xstd", torch.ones(14))

    def _add(self, nm, i, o, g, zero_out):
        H = fm_core.HID

        def lin(a, b, scale):
            return torch.nn.Parameter(torch.randn(a, b, generator=g) * scale)

        self.p[f"{nm}__W1"] = lin(i, H, (1.0 / i) ** 0.5)
        self.p[f"{nm}__b1"] = torch.nn.Parameter(torch.zeros(H))
        self.p[f"{nm}__W2"] = lin(H, H, (1.0 / H) ** 0.5)
        self.p[f"{nm}__b2"] = torch.nn.Parameter(torch.zeros(H))
        # output layers start small: every correction starts as (nearly) the analytic step
        self.p[f"{nm}__W3"] = torch.nn.Parameter(torch.zeros(H, o) if zero_out
                                                 else torch.randn(H, o, generator=g) * 1e-3)
        self.p[f"{nm}__b3"] = torch.nn.Parameter(torch.zeros(o))

    def add_stage2(self, seed=1):
        g = torch.Generator().manual_seed(seed)
        for nm in fm_core.STAGE2:
            if nm not in self.nets:
                i, o = fm_core.NETS[nm]
                self._add(nm, i, o, g, zero_out=True)
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
