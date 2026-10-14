"""torch backend of submission/fm_core.py: dense products, functional ops, autograd.

Same arithmetic as the flopscope backend (which only adds pooled out= buffers and
Strassen-Winograd products); ``out`` names are accepted and ignored.
"""

from __future__ import annotations

import math

import torch

F32 = torch.float32
_SQ2PI = 1.0 / math.sqrt(2.0 * math.pi)


class TorchBackend:
    name = "torch"

    def __init__(self, device="cpu"):
        self.device = device

    def zeros(self, shape):
        return torch.zeros(shape, dtype=F32, device=self.device)

    def ones(self, shape):
        return torch.ones(shape, dtype=F32, device=self.device)

    def power(self, x, e):
        return torch.pow(x, e)

    def const(self, data):
        return torch.tensor(data, dtype=F32, device=self.device)

    def copy(self, x):
        return x.clone()

    def reshape(self, x, shape):
        return x.reshape(shape)

    def swap(self, x):
        return x.transpose(-1, -2)

    def swapaxes01(self, x):
        return x.transpose(0, 1)

    def mul(self, a, b, out=None):
        return a * b

    def add(self, a, b, out=None):
        return a + b

    def maximum(self, x, c):
        return torch.clamp(x, min=c)

    def clip(self, x, lo, hi):
        return torch.clamp(x, lo, hi)

    def sqrt(self, x):
        return torch.sqrt(x)

    def abs(self, x, out=None):
        return torch.abs(x)

    def pdf(self, x):
        return torch.exp(-0.5 * x * x) * _SQ2PI

    def cdf(self, x):
        return torch.special.ndtr(x)

    def gelu(self, x):
        return x * torch.special.ndtr(x)

    def sum(self, x, axis, out=None):
        return x.sum(dim=axis)

    def mean(self, x, axis=None):
        return x.mean() if axis is None else x.mean(dim=axis)

    def diag(self, A):
        return torch.diagonal(A, dim1=-2, dim2=-1)

    def zero_diag(self, A, out=None):
        return A - torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1))

    def with_diag(self, A, v, out=None):
        return A - torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1)) + torch.diag_embed(v)

    def stack(self, xs, axis, out=None):
        return torch.stack(xs, dim=axis)

    def cat(self, xs, axis, out=None):
        return torch.cat(xs, dim=axis)

    def vstack1(self, terms, n, out=None):
        rows = []
        for a, b in terms:
            if a is None and b is None:
                rows.append(torch.ones(n, dtype=F32, device=self.device))
            elif a is None or b is None:
                rows.append(b if a is None else a)
            else:
                rows.append(a * b)
        return torch.stack(rows, 0)

    def vstack2(self, terms, n, out=None):
        rows = []
        for a, b in terms:
            v = (b if a is None else a) if (a is None or b is None) else a * b
            rows.append(v.expand(n, n))
        return torch.stack(rows, 0)

    def einsum(self, spec, *ops):
        if spec == "tij,ti,tj,tg->gij":
            AB, WL, WR, IND = ops
            X = AB * WL[:, :, None] * WR[:, None, :]
            return torch.einsum("tij,tg->gij", X, IND)
        return torch.einsum(spec, *ops)

    def qr(self, Y):
        return torch.linalg.qr(Y)[0]

    def mm(self, a, b, out=None):
        return torch.matmul(a, b)

    def family(self, W, Ys, out=None):
        return torch.matmul(W, Ys)

    def form(self, Q, F, out=None):
        return torch.matmul(Q, F)

    def rows(self, arr, lo, hi):
        return arr[lo:hi]

    def pair_slots(self, name, k, n):
        return None, None, None

    def thinhub(self, X, Y):
        return torch.einsum("kiq,kcq->ic", X, Y)

    def hub(self, X, Y, out=None, kind="hub"):
        return torch.einsum("bsij,bscj->ic", X, Y)

    def symprod(self, X, Y, out=None):
        return X @ Y

    def stack_params(self, plist, dout, key=None):
        G = len(plist)
        W1 = torch.stack([p[0] for p in plist], 0)
        b1 = torch.stack([p[1] for p in plist], 0)[:, None, :]
        W2 = torch.stack([p[2] for p in plist], 0)
        b2 = torch.stack([p[3] for p in plist], 0)[:, None, :]
        W3s, b3s = [], []
        for p in plist:
            o = p[4].shape[1]
            if o < dout:
                W3s.append(torch.nn.functional.pad(p[4], (0, dout - o)))
                b3s.append(torch.nn.functional.pad(p[5], (0, dout - o)))
            else:
                W3s.append(p[4])
                b3s.append(p[5])
        W3 = torch.stack(W3s, 0)
        b3 = torch.stack(b3s, 0)[:, None, :]
        return W1, b1, W2, b2, W3, b3
