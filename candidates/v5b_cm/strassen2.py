"""Batched Strassen for flopscope with low residual time (optc; successor of optb/strassen.py).

Residual wall time (Python outside flopscope ops) is what limits Strassen depth:
  * ~12 us per flopscope op, plus ~10 us per slice of a flopscope array (symmetry bookkeeping
    in __getitem__). The old kernel sliced ~33 views per level per call.
  * Here every internal buffer is pooled AND its views (7 combo planes, 4 quadrants) are made
    once and cached, so a level costs only its 22 ops (7 + 7 combos, 8 assembly adds).
  * The last level is "fused": its 7 products are formed one at a time (25 ops), so the
    deepest batched arrays are 7x smaller and 4 levels fit the 100 MiB per-array cap at
    batch 4 (n = 1024: largest array 86 MiB).
  * A shared left operand (the layer weight) can be prepared once per layer as a LeftTree:
    its combos at every level are reused by every call of that layer (7 ops/level saved).

mm(X, R, lev, out, ksum=False):
    X (Bx, m, k) array or LeftTree, R (B, k, w), Bx in {1, B}  ->  out (B, m, w)
    ksum=True: out (1, m, w) = sum_b X_b @ R_b (the D21 hub form; R may be a transposed view).
The caller provides `out` (results go straight into persistent buffers: no fresh arrays).
"""
import flopscope.numpy as fnp

MIN_BLOCK = 32
_POOL = {}
_VC = {}   # id(pooled array) -> (array, quads, planes)


def _quads(A):
    h, q = A.shape[-2] // 2, A.shape[-1] // 2
    return (A[..., :h, :q], A[..., :h, q:], A[..., h:, :q], A[..., h:, q:])


def buf(role, shape, dtype=None):
    key = (role, tuple(int(s) for s in shape))
    b = _POOL.get(key)
    if b is None:
        b = fnp.empty(key[1], dtype=dtype or fnp.float32)
        _POOL[key] = b
        planes = tuple(b[i] for i in range(b.shape[0])) if (b.shape[0] == 7 and b.ndim > 3) else None
        _VC[id(b)] = (b, _quads(b) if b.ndim >= 2 else None, planes)
    return b


def views(A):
    """(quads, planes) of A: cached for pooled / registered arrays, computed otherwise."""
    e = _VC.get(id(A))
    if e is not None and e[0] is A:
        return e[1], e[2]
    return _quads(A), None


def register(A):
    """Cache the views of a caller-owned persistent array (it is kept alive here)."""
    if _VC.get(id(A), (None,))[0] is not A:
        planes = tuple(A[i] for i in range(A.shape[0])) if (A.shape[0] == 7 and A.ndim > 3) else None
        _VC[id(A)] = (A, _quads(A), planes)
    return A


def _ok(m, k, w, mb):
    return m % 2 == 0 and k % 2 == 0 and w % 2 == 0 and min(m, k, w) // 2 >= mb


def levels(m, k, w, lev, mb=None):
    mb = MIN_BLOCK if mb is None else mb
    n = 0
    while n < lev and _ok(m >> n, k >> n, w >> n, mb):
        n += 1
    return n


def _combos_L(Q, P):
    X11, X12, X21, X22 = Q
    fnp.add(X11, X22, out=P[0]); fnp.add(X21, X22, out=P[1]); fnp.copyto(P[2], X11)
    fnp.copyto(P[3], X22); fnp.add(X11, X12, out=P[4]); fnp.subtract(X21, X11, out=P[5])
    fnp.subtract(X12, X22, out=P[6])


def _combos_R(Q, P):
    Y11, Y12, Y21, Y22 = Q
    fnp.add(Y11, Y22, out=P[0]); fnp.copyto(P[1], Y11); fnp.subtract(Y12, Y22, out=P[2])
    fnp.subtract(Y21, Y11, out=P[3]); fnp.copyto(P[4], Y22); fnp.add(Y11, Y12, out=P[5])
    fnp.add(Y21, Y22, out=P[6])


def _assemble(M, OQ):
    C11, C12, C21, C22 = OQ
    fnp.add(M[0], M[3], out=C11); fnp.subtract(C11, M[4], out=C11); fnp.add(C11, M[6], out=C11)
    fnp.add(M[2], M[4], out=C12)
    fnp.add(M[1], M[3], out=C21)
    fnp.subtract(M[0], M[1], out=C22); fnp.add(C22, M[2], out=C22); fnp.add(C22, M[5], out=C22)


class LeftTree:
    """Combos of a shared left operand X (1, m, k) at every level (plus the fused leaf's).
    t.c[d] = pooled (7,)*(d+1) + (1, m/2^(d+1), k/2^(d+1)) combos at depth d."""

    def __init__(self, X, lev, w, tag="T", mb=None):
        m, k = X.shape[-2], X.shape[-1]
        self.lev = levels(m, k, w, lev, mb)
        self.X = X
        self.c = []
        cur = X
        for d in range(self.lev):
            h, q = cur.shape[-2] // 2, cur.shape[-1] // 2
            Pc = buf((tag, d), (7,) + tuple(cur.shape[:-2]) + (h, q))
            _combos_L(views(cur)[0], views(Pc)[1])
            self.c.append(Pc)
            cur = Pc


def _leaf_fused(X, R, out, d, ksum, Lc):
    """Last level: 7 products one at a time. Lc = precomputed left combos (tree) or None."""
    RQ = views(R)[0]
    OQ = views(out)[0]
    Y11, Y12, Y21, Y22 = RQ
    C11, C12, C21, C22 = OQ
    if Lc is None:
        XQ = views(X)[0]
        X11, X12, X21, X22 = XQ
        Lb = buf(("FL", d), X11.shape)
        L = None
    else:
        L = views(Lc)[1]
    Rb = buf(("FR", d), Y11.shape)
    Mb = buf(("FM", d), C11.shape)
    if ksum:
        xs = L[0].shape if L is not None else X11.shape
        T = buf(("FT", d), tuple(max(a, b) for a, b in zip(xs[:-2], Y11.shape[:-2])) + (xs[-2], Y11.shape[-1]))

    def prod(a, b, dst):
        if ksum:
            fnp.matmul(a, b, out=T)
            fnp.sum(T, axis=-3, keepdims=True, out=dst)
        else:
            fnp.matmul(a, b, out=dst)

    def left(i, f):
        if L is not None:
            return L[i]
        f()
        return Lb

    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6 (Mi = 1-based)
    a = left(5, lambda: fnp.subtract(X21, X11, out=Lb)); fnp.add(Y11, Y12, out=Rb)
    prod(a, Rb, C22)                                                   # M6 -> C22
    a = left(1, lambda: fnp.add(X21, X22, out=Lb))
    prod(a, Y11, C21)                                                  # M2 -> C21
    fnp.subtract(C22, C21, out=C22)
    fnp.subtract(Y21, Y11, out=Rb)
    prod(L[3] if L is not None else X22, Rb, C11)                      # M4 -> C11
    fnp.add(C21, C11, out=C21)
    a = left(0, lambda: fnp.add(X11, X22, out=Lb)); fnp.add(Y11, Y22, out=Rb)
    prod(a, Rb, Mb)                                                    # M1
    fnp.add(C11, Mb, out=C11); fnp.add(C22, Mb, out=C22)
    fnp.subtract(Y12, Y22, out=Rb)
    prod(L[2] if L is not None else X11, Rb, C12)                      # M3 -> C12
    fnp.add(C22, C12, out=C22)
    a = left(4, lambda: fnp.add(X11, X12, out=Lb))
    prod(a, Y22, Mb)                                                   # M5
    fnp.subtract(C11, Mb, out=C11); fnp.add(C12, Mb, out=C12)
    a = left(6, lambda: fnp.subtract(X12, X22, out=Lb)); fnp.add(Y21, Y22, out=Rb)
    prod(a, Rb, Mb)                                                    # M7
    fnp.add(C11, Mb, out=C11)


def _rec(X, R, lev, d, out, ksum, tree):
    if lev == 0:
        if ksum:
            T = buf(("KT", d), tuple(max(a, b) for a, b in zip(X.shape[:-2], R.shape[:-2])) + (X.shape[-2], R.shape[-1]))
            fnp.matmul(X, R, out=T)
            fnp.sum(T, axis=-3, keepdims=True, out=out)
        else:
            fnp.matmul(X, R, out=out)
        return
    if lev == 1:
        _leaf_fused(X, R, out, d, ksum, tree.c[d] if tree is not None else None)
        return
    RQ = views(R)[0]
    h, q, v = out.shape[-2] // 2, R.shape[-2] // 2, R.shape[-1] // 2
    if tree is not None:
        Xc = tree.c[d]
    else:
        Xc = buf(("X", d), (7,) + tuple(X.shape[:-2]) + (h, q))
        _combos_L(views(X)[0], views(Xc)[1])
    Rc = buf(("R", d), (7,) + tuple(R.shape[:-2]) + (q, v))
    _combos_R(RQ, views(Rc)[1])
    M = buf(("M", d, ksum), (7,) + tuple(out.shape[:-2]) + (h, v))
    _rec(Xc, Rc, lev - 1, d + 1, M, ksum, tree)
    _assemble(views(M)[1], views(out)[0])


def mm(X, R, lev, out, ksum=False, mb=None):
    """See the module docstring. X: array (Bx, m, k) or LeftTree (its own level count rules)."""
    tree = X if isinstance(X, LeftTree) else None
    if tree is not None:
        lv = min(lev, tree.lev)
        Xa = tree.X
    else:
        Xa = X
        lv = levels(Xa.shape[-2], Xa.shape[-1], R.shape[-1], lev, mb)
    _rec(Xa, R, lv, 0, out, ksum, tree)
    return out
