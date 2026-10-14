"""flopscope backend of fm_core: pooled result buffers and Strassen-Winograd products.

Residual (non-flopscope) wall time is capped at 400 ms per MLP, and it is spent mostly on
result allocation and on Python-side view creation, not on arithmetic. So:
  - every large result is written with out= into a persistent named buffer (``Pool``);
    buffers grow along the leading axis to the largest batch seen and persist across MLPs;
  - every view of a pooled buffer (row slices, quadrants, Strassen slots) is created once
    and cached, so from the second MLP on an op costs only its own dispatch;
  - operands that are not pooled (the MLP weight, freshly computed factors) are staged into
    pooled buffers with one copy before entering a Strassen family.
The n^3 families run as batched Strassen-Winograd recursions written as ordinary flopscope
ops: seven half-size products and 22 block combinations per level, each combination one op
over the whole batch. ``LEVELS`` sets the recursion depth per family. All families share
one set of scratch buffers (they run one after another), which bounds the memory.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp

F32 = fnp.float32
STRASSEN_MIN = 32
# recursion depth per family (0 = dense): transport family, young hub contraction,
# shared-basis formings, shared-basis contractions, covariance sandwich
LEVELS = dict(family=2, hub=2, form=1, inner=1, sym=0)
# the first predict() of a process also pays buffer allocation and view creation on the
# residual clock: it runs shallower
FIRST_LEVELS = dict(family=1, hub=1, form=0, inner=0, sym=0)
# how many of the top recursion levels use the compact (coefficient-matmul) combinations:
# 6 ops per level but ~7x the block-addition FLOPs, so they pay at shallow levels only
COMPACT = dict(family=2, hub=2, form=1, inner=1, sym=1)


def _alloc(shape):
    # fnp.empty, not zeros: a zeros() buffer of square trailing shape is tagged symmetric
    # and matmul refuses to write a non-symmetric result into it. Every pooled buffer is
    # written before it is read.
    return fnp.empty(shape, dtype=F32)


# per-source stacks (up to L-1 = 15 sources) are allocated at full size up front
CAP_NAMES = {"legs", "lap", "ap", "pp", "zl", "mp", "t", "xt", "yt", "lpadd", "pyt", "x3"}


class Pool:
    """Persistent named buffers and cached views.

    Buffers of rank >= 3 are keyed by (name, trailing shape) and sliced along the leading
    axis; plain-named stacks are allocated with ``cap`` rows at once (the largest source
    count), so they never grow and their views stay valid. Every view handed out, and every
    view derived from one, is cached and tagged with the buffer it lives in; if a buffer
    has to grow, only the views into that buffer are dropped."""

    def __init__(self, cap=15):
        self.cap = cap
        self.bufs = {}
        self.views = {}     # (name, shape) -> view
        self.vkeys = {}     # buffer key -> [view keys]
        self.base = {}      # id(view) -> buffer key
        self.dv = {}        # (id(arr), key) -> (arr, derived)
        self.dkeys = {}     # buffer key -> [derived keys]

    def _register(self, obj, bk):
        if isinstance(obj, (tuple, list)):
            for o in obj:
                self._register(o, bk)
        elif isinstance(obj, dict):
            for o in obj.values():
                self._register(o, bk)
        else:
            self.base[id(obj)] = bk

    def _invalidate(self, bk):
        for vk in self.vkeys.pop(bk, ()):
            v = self.views.pop(vk, None)
            if v is not None:
                self.base.pop(id(v), None)
        for dk in self.dkeys.pop(bk, ()):
            self.dv.pop(dk, None)

    def get(self, name, shape):
        if shape.__class__ is not tuple:
            shape = tuple(shape)
        v = self.views.get((name, shape))
        if v is not None:
            return v
        shape = tuple(int(x) for x in shape)
        v = self.views.get((name, shape))
        if v is not None:
            return v
        if len(shape) >= 3:
            bk = (name, shape[1:])
            b = self.bufs.get(bk)
            if b is None or b.shape[0] < shape[0]:
                rows = shape[0]
                if name in CAP_NAMES:
                    rows = max(rows, self.cap)
                if b is not None:
                    self._invalidate(bk)
                b = _alloc((rows,) + shape[1:])
                self.bufs[bk] = b
            v = b[:shape[0]]
        else:
            bk = (name, shape)
            v = self.bufs.get(bk)
            if v is None:
                v = _alloc(shape)
                self.bufs[bk] = v
        self.views[(name, shape)] = v
        self.vkeys.setdefault(bk, []).append((name, shape))
        self.base[id(v)] = bk
        return v

    def get5(self, name, shape5):
        """Pooled (b, 7, P, h, q) buffer and its (b, 7P, h, q) view (reshape is billed:
        made once)."""
        k5 = ("5", name, shape5)
        v = self.views.get(k5)
        if v is not None:
            return v
        b5 = self.get(("b5", name), shape5)
        b4 = fnp.reshape(b5, (shape5[0], shape5[1] * shape5[2], shape5[3], shape5[4]))
        v = (b5, b4)
        bk = self.base[id(b5)]
        self.views[k5] = v
        self.vkeys.setdefault(bk, []).append(k5)
        self.base[id(b4)] = bk
        return v

    def derived(self, arr, key, fn):
        """Cached view derived from a pooled array (keyed by the array object)."""
        k = (id(arr), key)
        e = self.dv.get(k)
        if e is not None and e[0] is arr:
            return e[1]
        v = fn(arr)
        self.dv[k] = (arr, v)
        bk = self.base.get(id(arr))
        if bk is not None:
            self.dkeys.setdefault(bk, []).append(k)
            self._register(v, bk)
        return v

    def stage(self, name, arr):
        buf = self.get(name, arr.shape)
        if buf is not arr:
            fnp.copyto(buf, arr)
        return buf


def _quads(X):
    m, kd = X.shape[-2], X.shape[-1]
    h, q = m // 2, kd // 2
    return (X[..., :h, :q], X[..., :h, q:], X[..., h:, :q], X[..., h:, q:])


def _slots(B5):
    return tuple(B5[:, i] for i in range(7))


# Strassen-Winograd coefficients over the quadrant order (11, 12, 21, 22)
_CL = [[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]]
_CR = [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
_CA = [[1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, 1, 0]]
# hub right operand: blocks of Y^T read from the un-transposed Y, whose natural quadrant
# order (00, 01, 10, 11) is (B11, B21, B12, B22) of Y^T
_CRH = [[r[0], r[2], r[1], r[3]] for r in _CR]


def _blocks6(a):
    """(b, P, m, k) -> view (b, 2, 2, P, m/2, k/2) of its 2 x 2 blocks."""
    b, P, m, k = a.shape
    return fnp.transpose(fnp.reshape(a, (b, P, 2, m // 2, 2, k // 2)), (0, 2, 4, 1, 3, 5))


class Strassen:
    """mm: X (bx, P, m, kd) @ Y (by, P, kd, w) -> out (by, P, m, w), bx in {1, by};
    hub: X (b, P, m, kd), Y (b, P, w, kd) -> out (P, m, w) = sum_b X_b Y_b^T.

    Each level is six ops whatever the batch: one copy gathers the four quadrants of each
    operand into a contiguous (b, 4, P, h, q) stack, one small matmul with the 7 x 4
    Strassen-Winograd coefficient matrix forms the seven combinations as the (b, 7P, h, q)
    batch of the next level, and the seven products are assembled into the four output
    quadrants by one 4 x 7 coefficient matmul and one copy. All arguments must be pooled
    (cached) views."""

    def __init__(self, pool):
        self.pool = pool
        self.CL = fnp.asarray(_CL, dtype=F32)
        self.CR = fnp.asarray(_CR, dtype=F32)
        self.CRH = fnp.asarray(_CRH, dtype=F32)
        self.CA = fnp.asarray(_CA, dtype=F32)

    @staticmethod
    def ok(m, kd, w, lev):
        d = 2 ** lev
        return (lev > 0 and m % d == 0 and kd % d == 0 and w % d == 0
                and min(m, kd, w) // d >= STRASSEN_MIN)

    def level(self, m, kd, w, lev):
        while lev > 0 and not self.ok(m, kd, w, lev):
            lev -= 1
        return lev

    def _combine(self, A, coef, role, tag):
        """A (b, P, r, c) pooled -> (b, 7P, r/2, c/2) pooled view of the 7 combinations."""
        pl = self.pool
        b, P, r, c = A.shape
        h, q = r // 2, c // 2
        Q = pl.get((tag, "Q" + role, P, h, q), (b, 4, P, h, q))
        fnp.copyto(pl.derived(Q, "6", lambda a: fnp.reshape(a, (b, 2, 2, P, h, q))),
                   pl.derived(A, "b6", _blocks6))
        S5, S4 = pl.get5((tag, role, P, h, q), (b, 7, P, h, q))
        fnp.matmul(coef, pl.derived(Q, "3", lambda a: fnp.reshape(a, (b, 4, P * h * q))),
                   out=pl.derived(S5, "3", lambda a: fnp.reshape(a, (b, 7, P * h * q))))
        return S4

    def _assemble(self, M5, out, tag):
        """M5 (b, 7, P, h, w) products -> out (b, P, 2h, 2w)."""
        pl = self.pool
        b, _, P, h, w = M5.shape
        C = pl.get((tag, "QR", P, h, w), (b, 4, P, h, w))   # = the R gather buffer (free now)
        fnp.matmul(self.CA, pl.derived(M5, "3", lambda a: fnp.reshape(a, (b, 7, P * h * w))),
                   out=pl.derived(C, "3", lambda a: fnp.reshape(a, (b, 4, P * h * w))))
        fnp.copyto(pl.derived(out, "b6", _blocks6),
                   pl.derived(C, "6", lambda a: fnp.reshape(a, (b, 2, 2, P, h, w))))

    # ---- grouped block-addition form: 15 ops per level, ~1.5x the plain-addition FLOPs ----
    # The seven products are stored in slot order (M5, M6, M2, M3, M0, M1, M4), which makes
    # every pair of combinations a strided slice of the gathered quadrant stack Q = (11, 12,
    # 21, 22): one op writes two slots.
    def _combos_grouped(self, A, kind, role, tag, hubT=False):
        pl = self.pool
        b, P, r, c = A.shape
        h, q = r // 2, c // 2
        Q = pl.get((tag, "G" + role, P, h, q), (b, 4, P, h, q))
        fnp.copyto(pl.derived(Q, "6", lambda a: fnp.reshape(a, (b, 2, 2, P, h, q))),
                   pl.derived(A, "b6", _blocks6))
        S5, S4 = pl.get5((tag, role, P, h, q), (b, 7, P, h, q))
        if hubT:
            # Y^T blocks read from Y: natural order (00, 01, 10, 11) = (B11, B21, B12, B22)
            g = pl.derived(Q, "gT", lambda a: dict(
                b11=a[:, 0], b12=a[:, 2], b21=a[:, 1], b22=a[:, 3],
                b12_21=a[:, 2:0:-1], b11_22=a[:, 0::3], b22_11=a[:, 3::-3],
                b11_21=a[:, 0:2], b12_22=a[:, 2:4]))
        else:
            g = pl.derived(Q, "g", lambda a: dict(
                b11=a[:, 0], b12=a[:, 1], b21=a[:, 2], b22=a[:, 3],
                b21_12=a[:, 2:0:-1], b11_22=a[:, 0::3], b22_11=a[:, 3::-3],
                b11_21=a[:, 0::2], b12_22=a[:, 1::2], b12_21=a[:, 1:3]))
        sl = pl.derived(S5, "g", lambda a: (a[:, 0:2], a[:, 2:4], a[:, 4:6], a[:, 6], a[:, 4], a[:, 5:7]))
        if kind == "L":
            # slots (L5, L6) = (21, 12) - (11, 22); (L2, L3) = (11, 22); (L0, L1) = (11, 21) + 22;
            # L4 = 11 + 12
            if hubT:
                raise ValueError("left operand is never transposed")
            fnp.subtract(g["b21_12"], g["b11_22"], out=sl[0])
            fnp.copyto(sl[1], g["b11_22"])
            fnp.add(g["b11_21"], g["b22"][:, None], out=sl[2])
            fnp.add(g["b11"], g["b12"], out=sl[3])
        else:
            # slots (R5, R6) = (11, 21) + (12, 22); (R2, R3) = (12, 21) - (22, 11);
            # R0 = 11 + 22; (R1, R4) = (11, 22)
            fnp.add(g["b11_21"], g["b12_22"], out=sl[0])
            fnp.subtract(g["b12_21"], g["b22_11"], out=sl[1])
            fnp.add(g["b11"], g["b22"], out=sl[4])
            fnp.copyto(sl[5], g["b11_22"])
        return S4

    def _assemble_grouped(self, M5, out, tag):
        """Products in slot order T = (M5, M6, M2, M3, M0, M1, M4):
        C11 = T4 + T1 + T3 - T6, C22 = T4 + T0 + T2 - T5, (C12, C21) = (T2, T5) + (T6, T3)."""
        pl = self.pool
        b, _, P, h, w = M5.shape
        C = pl.get((tag, "GR", P, h, w), (b, 4, P, h, w))   # = the R gather buffer (free now)
        t = pl.derived(M5, "ga", lambda a: dict(
            t4=a[:, 4:5], t10=a[:, 1::-1], t32=a[:, 3:1:-1], t65=a[:, 6:4:-1],
            t25=a[:, 2::3], t63=a[:, 6:2:-3]))
        c = pl.derived(C, "ga", lambda a: (a[:, 0::3], a[:, 1:3]))
        fnp.add(t["t4"], t["t10"], out=c[0])
        fnp.add(c[0], t["t32"], out=c[0])
        fnp.subtract(c[0], t["t65"], out=c[0])
        fnp.add(t["t25"], t["t63"], out=c[1])
        fnp.copyto(pl.derived(out, "b6", _blocks6),
                   pl.derived(C, "6", lambda a: fnp.reshape(a, (b, 2, 2, P, h, w))))

    # ---- block-addition form (22 ops per level, 7 FLOPs per element of each operand) ----
    def _combos_add(self, A, kind, role, tag, Qfun=None):
        pl = self.pool
        b, P, r, c = A.shape
        h, q = r // 2, c // 2
        S5, S4 = pl.get5((tag, role, P, h, q), (b, 7, P, h, q))
        S = pl.derived(S5, "s", lambda a: tuple(a[:, i] for i in range(7)))
        A11, A12, A21, A22 = pl.derived(A, "q" if Qfun is None else "qt", Qfun or _quads)
        if kind == "L":
            fnp.add(A11, A22, out=S[0]); fnp.add(A21, A22, out=S[1])
            fnp.copyto(S[2], A11); fnp.copyto(S[3], A22)
            fnp.add(A11, A12, out=S[4]); fnp.subtract(A21, A11, out=S[5])
            fnp.subtract(A12, A22, out=S[6])
        else:
            fnp.add(A11, A22, out=S[0]); fnp.copyto(S[1], A11)
            fnp.subtract(A12, A22, out=S[2]); fnp.subtract(A21, A11, out=S[3])
            fnp.copyto(S[4], A22); fnp.add(A11, A12, out=S[5])
            fnp.add(A21, A22, out=S[6])
        return S4

    def _assemble_add(self, M5, out):
        pl = self.pool
        M = pl.derived(M5, "s", lambda a: tuple(a[:, i] for i in range(7)))
        C11, C12, C21, C22 = pl.derived(out, "q", _quads)
        fnp.add(M[0], M[3], out=C11); fnp.subtract(C11, M[4], out=C11)
        fnp.add(C11, M[6], out=C11)
        fnp.add(M[2], M[4], out=C12)
        fnp.add(M[1], M[3], out=C21)
        fnp.subtract(M[0], M[1], out=C22); fnp.add(C22, M[2], out=C22)
        fnp.add(C22, M[5], out=C22)

    def mm(self, X, Y, out, lev, tag="s", compact=0):
        bx, P, m, kd = X.shape
        by, w = Y.shape[0], Y.shape[3]
        if not self.ok(m, kd, w, lev):
            fnp.matmul(X, Y, out=out)
            return out
        h, v = m // 2, w // 2
        if compact > 0:
            L4 = self._combine(X, self.CL, "L", tag)
            R4 = self._combine(Y, self.CR, "R", tag)
        else:
            L4 = self._combos_grouped(X, "L", "L", tag)
            R4 = self._combos_grouped(Y, "R", "R", tag)
        M5, M4 = self.pool.get5((tag, "M" if compact > 0 else "MG", P, h, v), (by, 7, P, h, v))
        self.mm(L4, R4, M4, lev - 1, tag, compact - 1)
        if compact > 0:
            self._assemble(M5, out, tag)
        else:
            self._assemble_grouped(M5, out, tag)
        return out

    def hub(self, X, Y, out, lev, tag="s", compact=0):
        b, P, m, kd = X.shape
        w = Y.shape[2]
        pl = self.pool
        if not self.ok(m, kd, w, lev):
            T = pl.get((tag, "T", P, m, w), (b, P, m, w))
            fnp.matmul(X, pl.derived(Y, "t", lambda a: fnp.swapaxes(a, -1, -2)), out=T)
            fnp.sum(T, axis=0, out=out)
            return out
        h, v = m // 2, w // 2
        q = kd // 2
        if compact > 0:
            L4 = self._combine(X, self.CL, "L", tag)
            R4 = self._combine(Y, self.CRH, "R", tag)
        else:
            L4 = self._combos_grouped(X, "L", "L", tag)
            R4 = self._combos_grouped(Y, "R", "R", tag, hubT=True)
        M5, M4 = pl.get5((tag, "M" if compact > 0 else "MG", P, h, v), (1, 7, P, h, v))
        self.hub(L4, R4, pl.derived(M4, "0", lambda a: a[0]), lev - 1, tag, compact - 1)
        o4 = pl.derived(out, "1", lambda a: a[None])
        if compact > 0:
            self._assemble(M5, o4, tag)
        else:
            self._assemble_grouped(M5, o4, tag)
        return out


class FnpBackend:
    name = "fnp"

    def __init__(self, levels=None):
        self.pool = Pool()
        self.smm = Strassen(self.pool)
        self.levels = dict(LEVELS if levels is None else levels)
        self.compact = dict(COMPACT)
        self._oc = {}
        self.pcache = {}

    # ---- creation / conversion ----
    def zeros(self, shape):
        return fnp.zeros(shape, dtype=F32)

    def ones(self, shape):
        # untagged (fnp.ones of a square shape carries a symmetry tag whose validation on
        # every downstream op costs ~40 ms of dispatch per (n, n) operand)
        b = fnp.empty(shape, dtype=F32)
        fnp.copyto(b, 1.0)
        return b

    def power(self, x, e):
        return fnp.power(x, e)

    def const(self, data):
        return fnp.asarray(data, dtype=F32)

    def copy(self, x):
        return fnp.copy(x)

    def reshape(self, x, shape):
        return fnp.reshape(x, shape)

    def swap(self, x):
        return x.T if x.ndim == 2 else fnp.swapaxes(x, -1, -2)

    def swapaxes01(self, x):
        return fnp.swapaxes(x, 0, 1)

    # ---- elementwise ----
    @staticmethod
    def _bs(sa, sb):
        nd = max(len(sa), len(sb))
        sa = (1,) * (nd - len(sa)) + tuple(sa)
        sb = (1,) * (nd - len(sb)) + tuple(sb)
        return tuple(max(x, y) for x, y in zip(sa, sb))

    def _bshape(self, a, b):
        return self._bs(getattr(a, "shape", ()), getattr(b, "shape", ()))

    def _outbuf(self, out, a, b):
        if not isinstance(out, (str, tuple)):
            return out
        ka = (out, a.shape, getattr(b, "shape", ()))
        o = self._oc.get(ka)
        if o is None or self.pool.base.get(id(o)) is None:
            o = self.pool.get(out, self._bshape(a, b))
            self._oc[ka] = o
        return o

    def mul(self, a, b, out=None):
        if out is None:
            return a * b
        return fnp.multiply(a, b, out=self._outbuf(out, a, b))

    def add(self, a, b, out=None):
        if out is None:
            return a + b
        return fnp.add(a, b, out=self._outbuf(out, a, b))

    def maximum(self, x, c):
        return fnp.maximum(x, c)

    def clip(self, x, lo, hi):
        return fnp.clip(x, lo, hi)

    def sqrt(self, x):
        return fnp.sqrt(x)

    def abs(self, x, out=None):
        if out is None:
            return fnp.abs(x)
        return fnp.abs(x, out=self.pool.get(out, x.shape))

    def pdf(self, x):
        return flops.stats.norm.pdf(x).astype(F32)

    def cdf(self, x):
        return flops.stats.norm.cdf(x).astype(F32)

    def gelu(self, x):
        return x * self.cdf(x)

    # ---- reductions / structure ----
    def sum(self, x, axis, out=None):
        if out is None:
            return fnp.sum(x, axis=axis)
        ax = (axis,) if isinstance(axis, int) else tuple(axis)
        ax = tuple(a % x.ndim for a in ax)
        shape = tuple(s for i, s in enumerate(x.shape) if i not in ax)
        return fnp.sum(x, axis=axis, out=self.pool.get(out, shape))

    def mean(self, x, axis=None):
        return fnp.mean(x) if axis is None else fnp.mean(x, axis=axis)

    def diag(self, A):
        return fnp.diag(A)

    def zero_diag(self, A, out=None):
        """A with its diagonal zeroed, IN PLACE (callers pass values they own)."""
        fnp.fill_diagonal(A, 0.0)
        return A

    def with_diag(self, A, v, out=None):
        """A with diagonal v, in place."""
        fnp.fill_diagonal(A, v)
        return A

    def stack(self, xs, axis, out=None):
        if out is None:
            return fnp.stack(xs, axis=axis)
        s0 = tuple(xs[0].shape)
        shape = s0[:axis] + (len(xs),) + s0[axis:]
        return fnp.stack(xs, axis=axis, out=self.pool.get(out, shape))

    def cat(self, xs, axis, out=None):
        if out is None:
            return fnp.concatenate(xs, axis=axis)
        s0 = list(xs[0].shape)
        s0[axis] = sum(int(x.shape[axis]) for x in xs)
        return fnp.concatenate(xs, axis=axis, out=self.pool.get(out, s0))

    def qr(self, Y):
        return fnp.linalg.qr(Y)[0]

    # ---- products ----
    def mm(self, a, b, out=None):
        if out is None:
            return fnp.matmul(a, b)
        shape = self._bs(a.shape[:-2], b.shape[:-2]) + (a.shape[-2], b.shape[-1])
        return fnp.matmul(a, b, out=self.pool.get(out, shape))

    def rows(self, arr, lo, hi):
        """arr[lo:hi], as a cached view when arr is pooled (so Strassen can use it in place)."""
        if id(arr) in self.pool.base:
            return self.pool.derived(arr, ("rows", lo, hi), lambda a: a[lo:hi])
        return arr[lo:hi]

    def pair_slots(self, name, k, n):
        """(k, 2, n, n) pooled buffer and its two (k, n, n) slot views."""
        buf = self.pool.get(name, (k, 2, n, n))
        a, b = self.pool.derived(buf, "pair", lambda x: (x[:, 0], x[:, 1]))
        return buf, a, b

    def thinhub(self, X, Y):
        """sum_k X_k Y_k^T for thin X (k, n, q), Y (k, c, q) -> (n, c), one matmul."""
        k, n, q = X.shape
        Xf = fnp.reshape(fnp.swapaxes(X, 0, 1), (n, k * q))
        Yf = fnp.reshape(fnp.swapaxes(Y, 0, 1), (Y.shape[1], k * q))
        return fnp.matmul(Xf, fnp.swapaxes(Yf, 0, 1))

    def family(self, W, Ys, out):
        """W (n, n) @ Ys (b, n, n) -> (b, n, n)."""
        b, n, w = Ys.shape
        pl = self.pool
        Ws = pl.stage("fam_w", W)
        Ys = pl.stage("famin", Ys)
        o = pl.get(out, (b, n, w))
        lev = self.smm.level(W.shape[0], n, w, self.levels["family"])
        if lev == 0:
            return fnp.matmul(Ws, Ys, out=o)
        X4 = pl.derived(Ws, "4", lambda a: a[None, None])
        Y4 = pl.derived(Ys, "4", lambda a: a[:, None])
        O4 = pl.derived(o, "4", lambda a: a[:, None])
        self.smm.mm(X4, Y4, O4, lev, "s", self.compact["family"])
        return o

    def form(self, Q, F, out):
        """Q (n, r) @ F (b, 2, r, n) -> (b, 2, n, n)."""
        b, two, r, w = F.shape
        n = Q.shape[0]
        pl = self.pool
        o = pl.get(out, (b, two, n, w))
        lev = self.smm.level(n, r, w, self.levels["form"])
        if lev == 0:
            return fnp.matmul(Q, F, out=o)
        Qs = pl.stage(out + "_q", Q)
        Fs = pl.stage(out + "_f", F)
        X4 = pl.derived(Qs, "4", lambda a: a[None, None])
        F4 = pl.derived(Fs, "4", lambda a: fnp.reshape(a, (b * two, 1, r, w)))
        O4 = pl.derived(o, "4", lambda a: fnp.reshape(a, (b * two, 1, n, w)))
        self.smm.mm(X4, F4, O4, lev, "s", self.compact["form"])
        return o

    def hub(self, X, Y, out, kind="hub"):
        """X (b, 2, n, k), Y (b, 2, w, k) -> sum over (b, 2) of X_s Y_s^T, (n, w)."""
        b, two, m, kd = X.shape
        w = Y.shape[2]
        pl = self.pool
        o = pl.get(out, (1, m, w))
        lev = self.smm.level(m, kd, w, self.levels[kind])
        Xs = X if id(X) in pl.base else pl.stage(out + "_x", X)
        Ys = Y if id(Y) in pl.base else pl.stage(out + "_y", Y)
        X4 = pl.derived(Xs, "4", lambda a: fnp.reshape(a, (b * two, 1, m, kd)))
        Y4 = pl.derived(Ys, "4", lambda a: fnp.reshape(a, (b * two, 1, w, kd)))
        self.smm.hub(X4, Y4, o, lev, "s", self.compact[kind])
        return o[0]

    def symprod(self, X, Y, out):
        """X @ Y for a symmetric result: blocks 11, 12, 22 as one family, 21 = 12^T."""
        n = X.shape[0]
        pl = self.pool
        o = pl.get(out, (n, n))
        lev = self.smm.level(n // 2, n, n // 2, self.levels["sym"])
        if lev == 0 or n % 2:
            return fnp.matmul(X, Y, out=o)
        h = n // 2
        X3 = pl.get("sym_x", (3, 1, h, n))
        Y3 = pl.get("sym_y", (3, 1, n, h))
        O3 = pl.get("sym_o", (3, 1, h, h))
        fnp.copyto(X3[0, 0], X[:h]); fnp.copyto(X3[1, 0], X[:h]); fnp.copyto(X3[2, 0], X[h:])
        fnp.copyto(Y3[0, 0], Y[:, :h]); fnp.copyto(Y3[1, 0], Y[:, h:]); fnp.copyto(Y3[2, 0], Y[:, h:])
        self.smm.mm(X3, Y3, O3, lev, "s", self.compact["sym"])
        fnp.copyto(o[:h, :h], O3[0, 0]); fnp.copyto(o[:h, h:], O3[1, 0])
        fnp.copyto(o[h:, h:], O3[2, 0]); fnp.copyto(o[h:, :h], fnp.swapaxes(O3[1, 0], 0, 1))
        return o

    # ---- learned operator parameters ----
    def stack_params(self, plist, dout, key):
        e = self.pcache.get(key)
        if e is None:
            G = len(plist)
            W1 = fnp.stack([p[0] for p in plist], axis=0)
            b1 = fnp.reshape(fnp.stack([p[1] for p in plist], axis=0), (G, 1, -1))
            W2 = fnp.stack([p[2] for p in plist], axis=0)
            b2 = fnp.reshape(fnp.stack([p[3] for p in plist], axis=0), (G, 1, -1))
            W3s, b3s = [], []
            for p in plist:
                o = p[4].shape[1]
                if o < dout:
                    W3s.append(fnp.concatenate([p[4], fnp.zeros((p[4].shape[0], dout - o), dtype=F32)], axis=1))
                    b3s.append(fnp.concatenate([p[5], fnp.zeros((dout - o,), dtype=F32)], axis=0))
                else:
                    W3s.append(p[4])
                    b3s.append(p[5])
            W3 = fnp.stack(W3s, axis=0)
            b3 = fnp.reshape(fnp.stack(b3s, axis=0), (G, 1, dout))
            e = (W1, b1, W2, b2, W3, b3)
            self.pcache[key] = e
        return e
