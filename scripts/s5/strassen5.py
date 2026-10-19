"""Batched Strassen-Winograd for flopscope, quadrant-major below the top level (successor of strassen4.py).

strassen4 keeps every combo plane in the standard (h, q) layout, so every level reads and writes
strided quadrant views; at the deep levels their rows are 32-64 floats and numpy runs the adds at
2-2.6x the cost of contiguous ones. Here the top level splits the operands' quadrants into the
recursive block layout (..., (2, 2) * r, m >> r, k >> r) (fnp.reshape: one FLOP per element of the
operands, the transpose is free), and every plane below is stored in that layout: each quadrant at
each level is a contiguous sub-block. The arithmetic is strassen4's, operation by operation.

mm(X, R, lev, out, ksum=False):
    X (Bx, m, k) array or LeftTree, R (B, k, w), Bx in {1, B}  ->  out (B, m, w)
    ksum=True: out (1, m, w) = sum_b X_b @ R_b (the D21 hub form; R may be a transposed view).
"""
import flopscope.numpy as fnp

MIN_BLOCK = 32
_POOL = {}
_VC = {}   # id(pooled array) -> (array, standard quads, planes)
_VR = {}   # (id(array), r) -> (array, rec-layout quads)


def _quads(A):
    h, q = A.shape[-2] // 2, A.shape[-1] // 2
    return (A[..., :h, :q], A[..., :h, q:], A[..., h:, :q], A[..., h:, q:])


def _split(A, r):
    """standard-layout (..., m, k) view -> recursive-layout view (..., (2, 2) * r, m >> r, k >> r)."""
    if r == 0:
        return A
    nb = A.ndim - 2
    m, k = A.shape[-2], A.shape[-1]
    S = fnp.reshape(A, tuple(A.shape[:-2]) + (2,) * r + (m >> r,) + (2,) * r + (k >> r,))
    perm = list(range(nb))
    for t in range(r):
        perm += [nb + t, nb + r + 1 + t]
    perm += [nb + r, nb + 2 * r + 1]
    return fnp.transpose(S, tuple(perm))


def _rq(A, r):
    tail = (slice(None),) * (2 * r)
    return tuple(A[(Ellipsis, i, j) + tail] for i in (0, 1) for j in (0, 1))


_SLABS = {}   # (role, dtype) -> max-shape backing arrays of that role
_CAP = 134_000_000   # bytes per slab, under the grader's ~134.9 MB per-array cap


def _nbytes(shape, dt):
    size = 4 if dt == fnp.float32 else 8
    for s in shape:
        size *= s
    return size


def buf(role, shape, dtype=None):
    """Pooled scratch of a role: a free slice of a max-shape slab of that role (see strassen4)."""
    key = (role, tuple(int(s) for s in shape))
    b = _POOL.get(key)
    if b is None:
        dt = dtype or fnp.float32
        shp = key[1]
        sl = _SLABS.setdefault((role, str(dt)), [])
        S = next((S for S in sl if S.ndim == len(shp) and all(a <= c for a, c in zip(shp, S.shape))), None)
        if S is None:
            i = next((i for i, T in enumerate(sl) if T.ndim == len(shp) and _nbytes(
                tuple(max(a, c) for a, c in zip(shp, T.shape)), dt) <= _CAP), None)
            if i is None:
                S = fnp.empty(shp, dtype=dt)
                sl.append(S)
            else:
                S = fnp.empty(tuple(max(a, c) for a, c in zip(shp, sl[i].shape)), dtype=dt)
                sl[i] = S
                for kk in [kk for kk in _POOL if kk[0] == role]:   # views of the old slab
                    old = _POOL[kk]
                    _VC.pop(id(old), None)
                    for rk in [rk for rk in _VR if rk[0] == id(old)]:
                        del _VR[rk]
                    del _POOL[kk]
        b = S[tuple(slice(0, a) for a in shp)]
        _POOL[key] = b
        planes = tuple(b[i] for i in range(b.shape[0])) if (b.shape[0] == 7 and b.ndim > 3) else None
        _VC[id(b)] = (b, _quads(b) if b.ndim >= 2 else None, planes)
    return b


def views(A):
    """(standard quads, planes) of A: cached for pooled arrays, computed otherwise."""
    e = _VC.get(id(A))
    if e is not None and e[0] is A:
        return e[1], e[2]
    return _quads(A), None


def planes(A):
    e = _VC.get(id(A))
    if e is not None and e[0] is A and e[2] is not None:
        return e[2]
    return tuple(A[i] for i in range(A.shape[0]))


def rviews(A, r):
    """recursive-layout quads of A (r >= 1 levels below): cached for pooled arrays."""
    key = (id(A), r)
    e = _VR.get(key)
    if e is not None and e[0] is A:
        return e[1]
    q = _rq(A, r)
    if id(A) in _VC and _VC[id(A)][0] is A:
        _VR[key] = (A, q)
    return q


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
    """Combos of a shared left operand X (1, m, k) at every level (plus the fused leaf's), per level
    count lv: combos(lv)[d] = pooled (7,)*(d+1) + (1,) + ((2, 2) * (lv-1-d)) + leaf block at depth d
    (lv = 1: the standard-layout (h, q) planes, as strassen4)."""

    def __init__(self, X, lev, w, tag="T", mb=None):
        m, k = X.shape[-2], X.shape[-1]
        self.lev = levels(m, k, w, lev, mb)
        self.X = X
        self.tag = tag
        self._c = {}
        if self.lev >= 1:
            self.combos(self.lev)

    def combos(self, lv):
        c = self._c.get(lv)
        if c is not None:
            return c
        c = []
        X = self.X
        if lv == 1:
            h, q = X.shape[-2] // 2, X.shape[-1] // 2
            Pc = buf((self.tag, 0, 1), (7,) + tuple(X.shape[:-2]) + (h, q))
            _combos_L(views(X)[0], planes(Pc))
            c.append(Pc)
        else:
            XQ = tuple(_split(qq, lv - 1) for qq in _quads(X))
            cur = buf((self.tag, 0, lv), (7,) + tuple(XQ[0].shape))
            _combos_L(XQ, planes(cur))
            c.append(cur)
            for d in range(1, lv):
                QQ = rviews(cur, lv - d)
                Pc = buf((self.tag, d, lv), (7,) + tuple(QQ[0].shape))
                _combos_L(QQ, planes(Pc))
                c.append(Pc)
                cur = Pc
        self._c[lv] = c
        return c


def _leaf_fused(X, R, out, d, ksum, Lc, rec):
    """Last level: the 7 Winograd products one at a time (15 additions). Lc = precomputed left
    combos (Winograd order [A11, A12, S4, A22, S1, S2, S3]) or None. rec: X, R, out are in the
    recursive layout with one level left (their quadrants are contiguous blocks)."""
    Y11, Y12, Y21, Y22 = rviews(R, 1) if rec else views(R)[0]
    C11, C12, C21, C22 = rviews(out, 1) if rec else views(out)[0]
    if Lc is None:
        X11, X12, X21, X22 = rviews(X, 1) if rec else views(X)[0]
        S1 = buf(("FL", d), X11.shape)
        S2 = buf(("FL2", d), X11.shape)
        S3 = buf(("FL3", d), X11.shape)
        fnp.add(X21, X22, out=S1)
        fnp.subtract(S1, X11, out=S2)
        fnp.subtract(X11, X21, out=S3)
        A11, A12, A22, S4 = X11, X12, X22, None
    else:
        A11, A12, S4, A22, S1, S2, S3 = planes(Lc)
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


def _rec_r(X, R, r, d, out, ksum, tc):
    """X, R, out in the recursive layout with r >= 1 levels left; tc = tree combos or None."""
    if r == 1:
        _leaf_fused(X, R, out, d, ksum, tc[d] if tc is not None else None, True)
        return
    RQ = rviews(R, r)
    if tc is not None:
        Xc = tc[d]
    else:
        XQ = rviews(X, r)
        Xc = buf(("X", d), (7,) + tuple(XQ[0].shape))
        _combos_L(XQ, planes(Xc))
    Rc = buf(("R", d), (7,) + tuple(RQ[0].shape))
    _combos_R(RQ, planes(Rc))
    OQ = rviews(out, r)
    M = buf(("M", d, ksum), (7,) + tuple(OQ[0].shape))
    _rec_r(Xc, Rc, r - 1, d + 1, M, ksum, tc)
    _assemble(planes(M), OQ)


def mm(X, R, lev, out, ksum=False, mb=None):
    """See the module docstring. X: array (Bx, m, k) or LeftTree (its own level count rules)."""
    tree = X if isinstance(X, LeftTree) else None
    if tree is not None:
        lv = min(lev, tree.lev)
        Xa = tree.X
    else:
        Xa = X
        lv = levels(Xa.shape[-2], Xa.shape[-1], R.shape[-1], lev, mb)
    if lv == 0:
        if ksum:
            T = buf(("KT", 0), tuple(max(a, b) for a, b in zip(Xa.shape[:-2], R.shape[:-2])) + (Xa.shape[-2], R.shape[-1]))
            fnp.matmul(Xa, R, out=T)
            fnp.sum(T, axis=-3, keepdims=True, out=out)
        else:
            fnp.matmul(Xa, R, out=out)
        return out
    tc = tree.combos(lv) if tree is not None else None
    if lv == 1:
        _leaf_fused(Xa, R, out, 0, ksum, tc[0] if tc is not None else None, False)
        return out
    r = lv - 1
    if tc is not None:
        Xc = tc[0]
    else:
        XQ = tuple(_split(qq, r) for qq in _quads(Xa))
        Xc = buf(("X", 0), (7,) + tuple(XQ[0].shape))
        _combos_L(XQ, planes(Xc))
    RQ = tuple(_split(qq, r) for qq in _quads(R))
    Rc = buf(("R", 0), (7,) + tuple(RQ[0].shape))
    _combos_R(RQ, planes(Rc))
    OQ = tuple(_split(qq, r) for qq in _quads(out))
    M = buf(("M", 0, ksum), (7,) + tuple(OQ[0].shape))
    _rec_r(Xc, Rc, r, 1, M, ksum, tc)
    _assemble(planes(M), OQ)
    return out
