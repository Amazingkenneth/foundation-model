"""Batched Strassen-Winograd for flopscope with low residual time (cn52w; successor of strassen2.py).

cn52 changes, both exact in exact arithmetic:
  * the Winograd form of the 7 products (15 additions per level instead of 18; the batched
    levels save one output addition, the fused leaf three);


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


_FLAT = {}  # (role, dtype) -> flat backing buffer shared by every shape of that role


def buf(role, shape, dtype=None):
    """Pooled scratch of a role. Every shape of one role is a view of ONE flat buffer: a role
    is scratch inside a single call, so shapes that differ only by a column count (cn: narrowed
    families) or a batch chunk reuse the same memory instead of adding a pool each."""
    key = (role, tuple(int(s) for s in shape))
    b = _POOL.get(key)
    if b is None:
        dt = dtype or fnp.float32
        size = 1
        for s in key[1]:
            size *= s
        fk = (role, str(dt))
        fl = _FLAT.get(fk)
        if fl is None or fl.shape[0] < size:
            fl = fnp.empty((size,), dtype=dt)
            _FLAT[fk] = fl
            for kk in [kk for kk in _POOL if kk[0] == role]:   # views of the old backing
                _VC.pop(id(_POOL[kk]), None)
                del _POOL[kk]
        b = fnp.reshape(fl[:size], key[1])
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
    """Winograd left combos [A11, A12, S4, A22, S1, S2, S3]."""
    X11, X12, X21, X22 = Q
    fnp.add(X21, X22, out=P[4])          # S1 = A21 + A22
    fnp.subtract(P[4], X11, out=P[5])    # S2 = S1 - A11
    fnp.subtract(X11, X21, out=P[6])     # S3 = A11 - A21
    fnp.subtract(X12, P[5], out=P[2])    # S4 = A12 - S2
    fnp.copyto(P[0], X11); fnp.copyto(P[1], X12); fnp.copyto(P[3], X22)


def _combos_R(Q, P):
    """Winograd right combos [B11, B21, B22, T4, T1, T2, T3]."""
    Y11, Y12, Y21, Y22 = Q
    fnp.subtract(Y12, Y11, out=P[4])     # T1 = B12 - B11
    fnp.subtract(Y22, P[4], out=P[5])    # T2 = B22 - T1
    fnp.subtract(Y22, Y12, out=P[6])     # T3 = B22 - B12
    fnp.subtract(P[5], Y21, out=P[3])    # T4 = T2 - B21
    fnp.copyto(P[0], Y11); fnp.copyto(P[1], Y21); fnp.copyto(P[2], Y22)


def _assemble(M, OQ):
    """C from the Winograd products M[i] = L_i R_i (7 additions)."""
    C11, C12, C21, C22 = OQ
    fnp.add(M[0], M[1], out=C11)         # C11 = P1 + P2
    fnp.add(M[0], M[5], out=C12)         # U2 = P1 + P6
    fnp.add(C12, M[6], out=C21)          # U3 = U2 + P7
    fnp.add(C12, M[4], out=C12)          # U4 = U2 + P5
    fnp.add(C12, M[2], out=C12)          # C12 = U4 + P3
    fnp.add(C21, M[4], out=C22)          # C22 = U3 + P5
    fnp.subtract(C21, M[3], out=C21)     # C21 = U3 - P4


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
    """Last level: the 7 Winograd products one at a time (15 additions). Lc = precomputed left
    combos (tree, Winograd order [A11, A12, S4, A22, S1, S2, S3]) or None."""
    Y11, Y12, Y21, Y22 = views(R)[0]
    C11, C12, C21, C22 = views(out)[0]
    if Lc is None:
        X11, X12, X21, X22 = views(X)[0]
        S1 = buf(("FL", d), X11.shape)
        S2 = buf(("FL2", d), X11.shape)
        S3 = buf(("FL3", d), X11.shape)
        fnp.add(X21, X22, out=S1)
        fnp.subtract(S1, X11, out=S2)
        fnp.subtract(X11, X21, out=S3)
        A11, A12, A22, S4 = X11, X12, X22, None
    else:
        A11, A12, S4, A22, S1, S2, S3 = views(Lc)[1]
    T1 = buf(("FR", d), Y11.shape)
    T2 = buf(("FR2", d), Y11.shape)
    T3 = buf(("FR3", d), Y11.shape)
    Mb = buf(("FM", d), C11.shape)
    if ksum:
        xs = A11.shape
        T = buf(("FT", d), tuple(max(a, b) for a, b in zip(xs[:-2], Y11.shape[:-2])) + (xs[-2], Y11.shape[-1]))

    def prod(a, b, dst):
        if ksum:
            fnp.matmul(a, b, out=T)
            fnp.sum(T, axis=-3, keepdims=True, out=dst)
        else:
            fnp.matmul(a, b, out=dst)

    fnp.subtract(Y12, Y11, out=T1)
    fnp.subtract(Y22, T1, out=T2)
    fnp.subtract(Y22, Y12, out=T3)
    prod(A11, Y11, Mb)                                                 # P1
    prod(A12, Y21, C11); fnp.add(C11, Mb, out=C11)                     # C11 = P1 + P2
    prod(S2, T2, C12); fnp.add(C12, Mb, out=C12)                       # U2 = P1 + P6
    prod(S3, T3, C21); fnp.add(C21, C12, out=C21)                      # U3 = U2 + P7
    prod(S1, T1, Mb); fnp.add(C12, Mb, out=C12)                        # U4 = U2 + P5
    fnp.add(C21, Mb, out=C22)                                          # C22 = U3 + P5
    if S4 is None:
        fnp.subtract(X12, S2, out=S2)                                  # S4 = A12 - S2
        S4 = S2
    prod(S4, Y22, Mb); fnp.add(C12, Mb, out=C12)                       # C12 = U4 + P3
    fnp.subtract(T2, Y21, out=T2)                                      # T4 = T2 - B21
    prod(A22, T2, Mb); fnp.subtract(C21, Mb, out=C21)                  # C21 = U3 - P4


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
