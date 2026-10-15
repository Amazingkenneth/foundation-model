"""A learned foundation model for the ReLU step, on a factorized K=3 cumulant chain.

One code path, two backends: the same functions run on flopscope arrays in the submission
(``backend_fnp.FnpBackend``: pooled buffers, Strassen-Winograd products) and on torch tensors
during training (``fm/backend_torch.py``: dense products, autograd). Every op goes through
the backend object ``B`` or plain operators (+, -, *, /, @, slicing), which both array types
support; results are always used through return values, never by relying on mutation.

Layer step (Algorithm 1). State entering layer l: post-ReLU mean mu and covariance C of
layer l-1, the third cumulant as a sum of hub sources (one born per layer; dense (A, P) legs
for young sources, a shared transported basis for sources older than AGE_OLD transports and
a nested sub-basis beyond AGE_OLD2), thin transported legs and the post-ReLU kappa4 diagonal
core g. The linear step is the exact third-order transport; the ReLU step is
F_theta = F_an((u, v) + Delta_in) + Delta_out, where F_an is the analytic Hermite/Wick update
(term table in fm_tables) and the Delta are the learned corrections of ``Operator``.

Notation: z = W h with W = w32.T (w32 is the stored (n_in, n_out) matrix), s = sqrt(var(z)),
D3 = kappa3_iii, D21 = kappa3_iic (the (2,1) slice), dG = regenerated kappa4 diagonal core,
wk431 = (m/2) * kappa4 (3,1) slice with kappa_iiij = lambda C_off (+ learned correction).
"""

from __future__ import annotations

import math

from fm_tables import (ALPHA_DEG, LAM, METRIC_C, REF_R, SIG_EMAX, SIG_EMIN, TERM_SPECS,
                       WICK_PAIRS, WICK_UNIFIED)

R_RES = 16      # rank of the thin residual leg of the M sub-block
R_FB = 16       # rank of the D21 feedback thin legs
AGE_OLD = 4     # sources older than this many transports live in the shared basis
R_OLD = 384     # its rank
AGE_OLD2 = 7    # nested tier age gate
R_OLD2 = 224    # nested tier rank
QPASS = 1       # subspace passes of the n-space range finder
QPASS2 = 2      # passes of the r1-space range finder

I11 = WICK_PAIRS.index((1, 1))
I21 = WICK_PAIRS.index((2, 1))
I12 = WICK_PAIRS.index((1, 2))
I31 = WICK_PAIRS.index((3, 1))
D2_IPS = [(1, 1), (2, 1), (2, 2)]
D1_IPS = [(1,), (2,), (3,), (4,)]
MODE0_MISSING = {"d21", "d21T", "d3col", "d3row", "d3", "wk4m", "g4col", "g4row", "g4",
                 "wk431"}


def wick_tables():
    np_ = len(WICK_PAIRS)
    C1 = [[0.0] * np_ for _ in range(ALPHA_DEG + 1)]
    C2 = [[0.0] * np_ for _ in range(ALPHA_DEG + 1)]
    SELC = [[0.0] * np_ for _ in range(SIG_EMAX - SIG_EMIN + 1)]
    for j, (const, e, p1, p2) in enumerate(WICK_UNIFIED):
        SELC[e - SIG_EMIN][j] = const
        for i, c in enumerate(p1):
            C1[i][j] = c
        for i, c in enumerate(p2):
            C2[i][j] = c
    return C1, C2, SELC


MATRIX_SLICES = ("c_off", "d21", "d21T", "wk431")
VECTOR_SLICES = {"d3row": ("r", "d3"), "d3col": ("l", "d3"), "g4row": ("r", "g4"),
                 "g4col": ("l", "g4")}


def term_program(mode: int):
    """Nonlinear-term program of one availability mode (mode 0: no K3/K4 slices yet).

    d=2 outputs: pK[g](i, j) = sum_t coef_t S_t(i, j) w_l(t)_i w_r(t)_j. Slices that are row
    or column vectors (d3row/col, g4row/col) are folded into the Wick vectors, and the rank-2
    slice wk4m = q_i + q_j (q = m dG / 6) splits into two folded terms (the d=2 outputs are
    zero-diagonal, so its diagonal never matters). What remains is a handful of matrix
    classes M_c (1, C_off, D21, products, ...): pK[g] = sum_c M_c * (WL_cg WR_cg^T) with thin
    WL_cg, WR_cg -- one batched thin product and one weighted sum instead of a loop over
    terms. d=1 outputs: pK[g](i) = sum_t coef_t v_t(i) w_l(t)_i."""
    missing = MODE0_MISSING if mode == 0 else set()
    np_ = len(WICK_PAIRS)
    exp = []   # (g, cls, wl, wr, coef, lscale, rscale)
    for g, ip in enumerate(D2_IPS):
        for a, b, wl, wr, coef in TERM_SPECS[ip]:
            if a in missing or b in missing:
                continue
            variants = [([], [], [])]
            for sl in (a, b):
                if sl == "ones2":
                    continue
                if sl in MATRIX_SLICES:
                    for v in variants:
                        v[0].append(sl)
                elif sl in VECTOR_SLICES:
                    side, nm = VECTOR_SLICES[sl]
                    for v in variants:
                        (v[1] if side == "l" else v[2]).append(nm)
                elif sl == "wk4m":
                    variants = ([(m, l + ["q"], r) for m, l, r in variants]
                                + [(m, l, r + ["q"]) for m, l, r in variants])
                else:
                    raise ValueError(sl)
            for m, l, r in variants:
                exp.append((g, tuple(sorted(m)), wl, wr, coef,
                            "*".join(sorted(l)) or "1", "*".join(sorted(r)) or "1"))
    classes = sorted({e[1] for e in exp})
    vnames = sorted({e[5] for e in exp} | {e[6] for e in exp})
    G = len(D2_IPS)
    cnt = {}
    for e in exp:
        cnt[(e[1], e[0])] = cnt.get((e[1], e[0]), 0) + 1
    rmax = max(cnt.values())
    T = len(exp)
    SL2 = [[0.0] * np_ for _ in range(T)]
    SR2 = [[0.0] * np_ for _ in range(T)]
    SELL = [[0.0] * len(vnames) for _ in range(T)]
    SELR = [[0.0] * len(vnames) for _ in range(T)]
    PL = [[0.0] * T for _ in range(len(classes) * G * rmax)]
    PR = [[0.0] * T for _ in range(len(classes) * G * rmax)]
    fill = {}
    for t, (g, cls, wl, wr, coef, ls, rs) in enumerate(exp):
        SL2[t][wl] = coef
        SR2[t][wr] = 1.0
        SELL[t][vnames.index(ls)] = 1.0
        SELR[t][vnames.index(rs)] = 1.0
        c = classes.index(cls)
        slot = fill.get((c, g), 0)
        fill[(c, g)] = slot + 1
        row = (c * G + g) * rmax + slot
        PL[row][t] = 1.0
        PR[row][t] = 1.0
    # d = 1
    d1 = []
    for g, ip in enumerate(D1_IPS):
        for a, b, wl, wr, coef in TERM_SPECS[ip]:
            if a in missing or b in missing:
                continue
            nm = "*".join(sorted(x for x in (a, b) if x != "ones1")) or "1"
            d1.append((g, wl, coef, nm))
    v1names = sorted({e[3] for e in d1})
    SL1 = [[0.0] * np_ for _ in d1]
    SEL1 = [[0.0] * len(v1names) for _ in d1]
    IND1T = [[0.0] * len(d1) for _ in D1_IPS]
    for t, (g, wl, coef, nm) in enumerate(d1):
        SL1[t][wl] = coef
        SEL1[t][v1names.index(nm)] = 1.0
        IND1T[g][t] = 1.0
    return dict(classes=classes, vnames=vnames, rmax=rmax, G=G, SL2=SL2, SR2=SR2, SELL=SELL,
                SELR=SELR, PL=PL, PR=PR, v1names=v1names, SL1=SL1, SEL1=SEL1, IND1T=IND1T)


def statics(n: int) -> dict:
    # r=1 matrix-core harmonic projection of the post-ReLU kappa4 (closed forms in n)
    return dict(cA=6.0 / (n + 4.0), cI=-3.0 / ((n + 2.0) * (n + 4.0)))


class Consts:
    """Per-backend constant arrays (built once per estimator)."""

    def __init__(self, B):
        C1, C2, SELC = wick_tables()
        self.C1, self.C2, self.SELC = B.const(C1), B.const(C2), B.const(SELC)
        self.APOW_E = B.const([[float(i) for i in range(ALPHA_DEG + 1)]])
        self.P14_E = B.const([[0.0, 1.0, 2.0, 3.0, 4.0]])
        # rows: monomial X_a * Y_b at index 4 a + b, X = (1, p1, p1^2, p1^3, p1^4, pk2),
        # Y = (1, pk2, pk3, pk4); columns: K2, K3, K4
        kc = [[0.0] * 3 for _ in range(24)]
        kc[4 * 0 + 1][0] = 1.0      # pk2
        kc[4 * 2 + 0][0] = -1.0     # -p1^2
        kc[4 * 0 + 2][1] = 1.0      # pk3
        kc[4 * 3 + 0][1] = 2.0      # 2 p1^3
        kc[4 * 1 + 1][1] = -3.0     # -3 p1 pk2
        kc[4 * 0 + 3][2] = 1.0      # pk4
        kc[4 * 5 + 1][2] = -3.0     # -3 pk2^2
        kc[4 * 2 + 1][2] = 12.0     # 12 p1^2 pk2
        kc[4 * 4 + 0][2] = -6.0     # -6 p1^4
        kc[4 * 1 + 2][2] = -4.0     # -4 p1 pk3
        self.KCOEF = B.const(kc)
        self.SB_E = B.const([[float(e) for e in range(SIG_EMIN, SIG_EMAX + 1)]])
        self.progs = {}
        for mode in (0, 1):
            p = term_program(mode)
            self.progs[mode] = dict(
                {k: p[k] for k in ("classes", "vnames", "rmax", "G", "v1names")},
                **{k: B.const(p[k]) for k in ("SL2", "SR2", "SELL", "SELR", "PL", "PR",
                                              "SL1", "SEL1", "IND1T")})


# ---------------------------------------------------------------------------------------
class Chain:
    """Factorized K=3 cumulant chain with hooks for the learned ReLU-step corrections.

    ``run`` executes layers [start, stop) from ``state`` (None = network input) and returns
    the list of predicted post-ReLU means of those layers and the state after them.
    """

    def __init__(self, B, consts: Consts, n: int, L: int):
        self.B, self.K, self.n, self.L = B, consts, n, L
        self.st_ = statics(n)
        self.r = min(R_RES, n)
        self.rfb = min(R_FB, n)
        self.confine = R_OLD < n
        self.nest = self.confine and R_OLD2 < R_OLD
        self.ones2 = B.ones((n, n))
        self.ones_n = B.ones((n,))

    def init_state(self):
        B, n = self.B, self.n
        return dict(mu=B.zeros((n,)), C=None, k=0, ka=0, kb=0, APhat=None, Qc=None,
                    FAP=None, FAP2=None, U=None, Sg=None, S2=None, Z=None, Lst=None,
                    R1T=None, R2T=None, w2b=[], s=[], e=[], c1=[], c2=[], dA=[], dP=[],
                    g_prev=None, var_prev=None, lam_prev=0.0, w1_prev=None, K3v_prev=None,
                    newborn=None,
                    # carried for the learned operator's features (v5b layout): previous
                    # layer's pre-activation (alpha, var, corrected D3), its corrected kappa4
                    # diagonal (m dG), its post-ReLU K(4,), and per-source birth statistics
                    alpha_prev=None, varpre_prev=None, D3c_prev=None, g4c_prev=B.zeros((n,)),
                    K4v_prev=None, g3s=[], cvs=[])

    def run(self, ws, hooks=None, state=None, start=0, stop=None):
        stop = self.L if stop is None else stop
        st = self.init_state() if state is None else dict(state)
        rows = []
        for li in range(start, stop):
            row, st = self.layer(li, ws[li], st, hooks)
            rows.append(row)
        return rows, st

    # -----------------------------------------------------------------------------------
    def layer(self, li, w32, st, hooks):
        B, n, L, r, rfb = self.B, self.n, self.L, self.r, self.rfb
        last = li == L - 1
        W = B.swap(w32)
        mu = st["mu"] @ w32                      # = W mu
        k_old = st["k"]
        ka, kb = st["ka"], st["kb"]
        nb = st["newborn"]
        C_pre = None
        var = None
        # ---------------- linear step: exact third-order transport -----------------
        if li == 0:
            if last:
                var = B.maximum(B.sum(w32 * w32, 0), 1e-10)
            else:
                C_pre = B.symprod(W, w32, out="cpre")      # C = I: C_pre = w32^T w32
            A_all = P_all = None
            k = 0
        else:
            w1p = st["w1_prev"]
            WD = B.mul(W, w1p[None, :], out="wd")
            APh, PPh = st["APhat"] or (None, None)          # (ny, n, n) w1-scaled young A, P legs
            Qc, FAP, FAP2, U, Sg, S2 = st["Qc"], st["FAP"], st["FAP2"], st["U"], st["Sg"], st["S2"]
            ka_t = ka
            if self.confine and not last and k_old > 0:
                ka_t = max(ka, min(k_old, li - AGE_OLD))
            if ka_t > ka:
                # ---- one source joins the shared basis (range finder on the leg Gram) ----
                j = ka
                Aj = APh[0]
                Pj = PPh[0]
                dAj = st["dA"][j][:, None]
                dPj = st["dP"][j][:, None]
                Om = B.copy(w32[:, :R_OLD])
                Qp = w1p[:, None] * Qc if ka > 0 else None
                # the joiner's two legs as one batch: every n x n x r product below is one
                # shared-left Strassen family (B.lmm) or one hub (B.hub2)
                AjPj = B.stack([Aj, Pj], 0, out="join_ap")              # (2, n, n)
                dJ = B.stack([st["dA"][j], st["dP"][j]], 0)[:, None, :]  # (2, 1, n)
                for _ in range(QPASS):
                    # Yq = Aj dA Aj^T Om + Pj dP Pj^T Om (+ the old basis' share)
                    T = B.mul(B.lmm(B.swap(Om), AjPj, out="join_t"), dJ, out="join_td")
                    Yq = B.hub2(AjPj[0:1], AjPj[1:2], T[0:1], T[1:2], out="join_y", kind="join")
                    if ka > 0:
                        Sfull = Sg if kb == 0 else Sg + U @ (S2 @ B.swap(U))
                        Yq = Yq + Qp @ (Sfull @ (B.swap(Qp) @ Om))
                    Qn = B.qr(Yq)
                    Om = Qn
                QnT = B.swap(Qn)
                FAPj = B.copy(B.lmm(QnT, AjPj, out="join_f"))[None]     # (1, 2, r1, n)
                if ka > 0:
                    Tq = QnT @ Qp
                    if ka > kb:
                        FAP = B.cat([B.form(Tq, FAP, out="rot1"), FAPj], 0)
                        Sg = Tq @ (Sg @ B.swap(Tq))
                    else:
                        FAP, Sg = FAPj, None
                    if kb > 0:
                        U = Tq @ U
                else:
                    FAP = FAPj
                FAj, FPj = FAPj[0, 0], FAPj[0, 1]
                Sj = FAj @ (dAj * B.swap(FAj)) + FPj @ (dPj * B.swap(FPj))
                Sg = Sj if Sg is None else Sg + Sj
                kb_t = kb
                if self.nest:
                    kb_t = max(kb, min(ka_t - 1, li - AGE_OLD2))
                if kb_t > kb:
                    # ---- the oldest tier-1 source moves into the nested sub-basis ----
                    FA1, FP1 = FAP[0, 0], FAP[0, 1]
                    dAb = st["dA"][kb][:, None]
                    dPb = st["dP"][kb][:, None]
                    S_s = FA1 @ (dAb * B.swap(FA1)) + FP1 @ (dPb * B.swap(FP1))
                    G2 = S_s if kb == 0 else S_s + U @ (S2 @ B.swap(U))
                    Om2 = B.copy(w32[:R_OLD, :R_OLD2])
                    for _ in range(QPASS2):
                        Un = B.qr(G2 @ Om2)
                        Om2 = Un
                    UnT = B.swap(Un)
                    FAP2j = B.stack([UnT @ FA1, UnT @ FP1], 0)[None]
                    if kb > 0:
                        FAP2 = B.cat([B.form(UnT @ U, FAP2, out="rot2"), FAP2j], 0)
                    else:
                        FAP2 = FAP2j
                    S2 = UnT @ (G2 @ Un)
                    U = Un
                    FAP = FAP[1:]
                    Sg = Sg - S_s
                    kb = kb_t
                Qc = B.lmm(W, Qn[None], out="qc%d" % (li % 2))[0]
                APh, PPh = APh[1:], PPh[1:]
                ka = ka_t
            elif ka > 0:
                Qc = B.lmm(WD, Qc[None], out="qc%d" % (li % 2))[0]
            # A and P legs are kept as two (k, n, n) stacks (every array stays under the
            # grader's 100 MiB per-array cap; one (k, 2, n, n) stack would not at k = 15)
            pa, pp = [], []
            if kb > 0:
                QU = Qc @ U
                f2 = B.form(QU, FAP2, out="form2")              # (kb, 2, n, n)
                pa.append(f2[:, 0]); pp.append(f2[:, 1])
            if ka > kb:
                f1 = B.form(Qc, FAP, out="form1")               # (ka-kb, 2, n, n)
                pa.append(f1[:, 0]); pp.append(f1[:, 1])
            # ---- transport family: young legs, the newborn A leg and C ride one product ----
            ny = k_old - ka
            fam_in = [nb["a_b"][None], st["C"][None]]
            if ny > 0:
                fam_in = [APh, PPh] + fam_in
            fam = B.family(W, B.cat(fam_in, 0, out="famin"), out="famout")
            if ny > 0:
                pa.append(fam[:ny]); pp.append(fam[ny:2 * ny])
            WC = fam[2 * ny + 1]
            if last:
                var = B.maximum(B.sum(B.mul(WC, W, out="t1"), 1), 1e-10)   # diag(W C W^T)
            else:
                C_pre = B.symprod(WC, w32, out="cpre")
            pa.append(fam[2 * ny:2 * ny + 1])                   # newborn: A = W a_b, P = W
            pp.append(W[None])
            A_all = B.cat(pa, 0, out="legsA")
            P_all = B.cat(pp, 0, out="legsP")
            k = k_old + 1
            # thin legs: transported + the newborn columns
            # thin legs [Z_res (r) | u | y | F1 (rfb) | F2 (rfb)]: transported + newborn
            zn = (W @ nb["Rr"])[None]
            Z = B.cat([WD @ st["Z"], zn], 0) if k_old > 0 else zn
            st["s"] = st["s"] + [nb["s"]]
            st["e"] = st["e"] + [nb["e"]]
            st.update(Qc=Qc, FAP=FAP, FAP2=FAP2, U=U, Sg=Sg, S2=S2, Z=Z)
        st["ka"], st["kb"], st["k"] = ka, kb, k

        # ---------------- pre-activation slices -----------------
        C_off = None
        if not last:
            var = B.maximum(B.diag(C_pre), 1e-10)
            C_off = B.zero_diag(C_pre, out="coff")
        s_pre = B.sqrt(var)
        D3 = D21 = dG = None
        lam = st["lam_prev"]
        ctx = None
        dG_birth = None
        D3_birth = None
        if k > 0:
            D3, D21 = self.dslices(st, A_all, P_all, need_d21=not last)
            WW = B.mul(W, W, out="ww")
            t_g = WW @ st["g_prev"]
            t_v = var - WW @ st["var_prev"]
            dG0 = t_g + t_v * lam
            rr = B.mean(dG0) / B.mean(var)
            lam = lam * B.clip(rr * (1.0 / REF_R[min(li - 1, len(REF_R) - 1)]), 0.5, 2.0)
            dG = t_g + t_v * lam
            kap31 = None
            dG_birth = dG
            D3c = D3
            if hooks is not None:
                ctx = dict(li=li, L=L, last=last, W=W, w32=w32, WW=WW, mu=mu, var=var, s=s_pre,
                           D3=D3, D21=D21, C_off=C_off, dG=dG, t_g=t_g, t_v=t_v, lam=lam,
                           g_prev=st["g_prev"], var_prev=st["var_prev"], K3v_prev=st["K3v_prev"],
                           A_all=A_all, P_all=P_all, w1_prev=st["w1_prev"], alpha_prev=st["alpha_prev"],
                           varpre_prev=st["varpre_prev"], D3c_prev=st["D3c_prev"],
                           g4c_prev=st["g4c_prev"], K4v_prev=st["K4v_prev"], g3s=st["g3s"],
                           cvs=st["cvs"])
                corr = hooks.pre(ctx)
                if corr is not None:
                    D3c, dG, kap31 = corr
                    # which quantities the new source's birth reads: the paper operator feeds
                    # the corrected kappa4 diagonal to the K4 -> K3 feed and keeps the raw D3
                    # in the old-source scaling; the v5b operator does the opposite
                    if getattr(hooks, "birth_corrected_dG", True):
                        dG_birth = dG
                    if getattr(hooks, "d3w_corrected", False):
                        D3_birth = D3c
            st["g4c_prev"] = dG * METRIC_C
            st["D3c_prev_next"] = D3c
        # ---------------- analytic ReLU step F_an -----------------
        Wall, alpha_w, phi_w, Phi_w = self.wick(mu, var)
        WT = B.swap(Wall)
        mode = 0 if k == 0 else 1
        prog = self.K.progs[mode]
        vec = {"1": self.ones_n}
        if mode == 1:
            g4 = dG * METRIC_C
            vec.update(d3=D3c, g4=g4)
        PK1 = self.term1(prog, WT, vec)
        pk1v = PK1[0]
        if last:
            if hooks is not None and ctx is not None:
                pk1v = hooks.last(ctx, pk1v)
            return pk1v, st
        pk2v, pk3v, pk4v = PK1[1], PK1[2], PK1[3]
        mats = {"c_off": C_off}
        if mode == 1:
            wk431 = C_off * (0.5 * METRIC_C * lam)
            if kap31 is not None:
                wk431 = wk431 + kap31 * (0.5 * METRIC_C)
            mats.update(d21=D21, d21T=B.swap(D21), wk431=wk431)
            vec["q"] = dG * (METRIC_C / 6.0)
        PK2 = self.term2(prog, WT, vec, mats)
        pk11 = B.zero_diag((PK2[0] + B.swap(PK2[0])) * 0.5, out="pk11")
        pk21 = B.zero_diag(PK2[1], out="pk21")
        pk22 = B.zero_diag((PK2[2] + B.swap(PK2[2])) * 0.5, out="pk22")

        # ---------------- wick scalings of the old content, birth of the new source ----
        w1 = Wall[:, I11]
        w2 = Wall[:, I21]
        w1c = w1[:, None]
        if mode == 1:
            D3_w = (D3 if D3_birth is None else D3_birth) * w1 * w1 * w1
            D21_w = (w1c * w1c) * D21 * w1[None, :]
        a_b = B.mul(w1c, C_off, out="ab")
        if mode == 1:
            w3 = Wall[:, I31]
            Omf = B.copy(w32[:, :rfb])
            Qf = B.qr(D21 @ Omf)
            Qf = B.qr(D21 @ (B.swap(D21) @ Qf))
            Bf = B.swap(Qf) @ D21
            F1_b = w2[:, None] * Qf
            R1T_b = B.swap(Bf) * 1.5
            F2_b = w1c * B.swap(Bf)
            R2T_b = Qf * (w3 * 0.5)[:, None]
            Xt_b = F1_b @ B.swap(R1T_b)
            Yt_b = F2_b @ B.swap(R2T_b)
            X1_b = a_b * 3.0 + Xt_b
            Y1_b = a_b * w2[None, :] + Yt_b
            xd = B.diag(Xt_b)
            yd = B.diag(Yt_b)
            D21_new = (xd[:, None] * B.swap(Y1_b) + X1_b * Y1_b + yd[:, None] * B.swap(X1_b)) * (1.0 / 3.0)
            D3_new = xd * yd
            # kappa4 -> kappa3 feed of the regenerated core
            w1sq = w1 * w1
            y_b = w2 * (0.25 * METRIC_C)
            dgw = w1sq * dG_birth
            u_b = y_b * dG_birth
            c1_b = w1 * lam
            D21_new = (D21_new + dgw[:, None] * y_b[None, :] * (1.0 / 3.0)
                       + (y_b * (2.0 / 3.0))[:, None] * (a_b * c1_b[None, :])
                       + w1sq[:, None] * (u_b * (1.0 / 3.0))[None, :])
            D3_new = D3_new + dgw * y_b + u_b * w1sq
            rep21 = B.zero_diag(D21_w + D21_new, out="rep21")
        else:
            D21_new = a_b * a_b * w2[None, :]
            rep21 = B.zero_diag(D21_new, out="rep21")
            zn_ = B.zeros((n,))
            F1_b = B.zeros((n, rfb))
            F2_b = R1T_b = R2T_b = F1_b
            y_b = u_b = c1_b = dgw = w1sq = zn_
        # ---------------- power cumulants -> cumulants -----------------
        p1 = pk1v
        K1v = p1
        # K2 = pk2 - p1^2,  K3 = pk3 + 2 p1^3 - 3 p1 pk2,
        # K4 = pk4 - 3 pk2^2 + 12 p1^2 pk2 - 6 p1^4 - 4 p1 pk3:
        # one outer product of (1, p1, .., p1^4, pk2) with (1, pk2, pk3, pk4) and one matmul
        Xm = B.cat([B.power(p1[:, None], self.K.P14_E), pk2v[:, None]], 1)       # (n, 6)
        Ym = B.swap(B.cat([self.ones_n[None, :], PK1[1:4]], 0))                 # (n, 4)
        K234 = B.reshape(Xm[:, :, None] * Ym[:, None, :], (n, 24)) @ self.K.KCOEF  # (n, 3)
        K2v, K3v, K4v = K234[:, 0], K234[:, 1], K234[:, 2]
        q2 = p1 * 2.0
        q2c, q2r = q2[:, None], q2[None, :]
        K11 = pk11
        # K21 = pk21 - 2 p1_i pk11;  K22 = pk22 + 4 p1_i p1_j pk11 - 2 p1_j pk21 - 2 p1_i pk21^T - 2 pk11^2
        K21 = B.zero_diag(pk21 - q2c * pk11, out="k21")
        K22 = B.zero_diag((q2c * pk11 - pk21) * q2r - q2c * B.swap(pk21) - pk11 * pk11 * 2.0 + pk22,
                          out="k22")
        k22row = B.sum(K22, 1)
        g_new = (K4v + k22row) * self.st_["cA"] + (B.sum(K4v, 0) + B.sum(k22row, 0)) * self.st_["cI"]
        S3c = K3v - D3_w - D3_new if mode == 1 else K3v
        e_b = Wall[:, I12] - 2.0 * p1 * w1
        # ---------------- learned output corrections (Delta_out) -----------------
        if hooks is not None and ctx is not None:
            out = hooks.post(ctx, dict(K1v=K1v, K2v=K2v, K11=K11, g=g_new, S3c=S3c, w2=w2, e=e_b))
            if out is not None:
                K1v, K2v, K11, g_new, S3c, w2b, e_b = out
            else:
                w2b = w2
        else:
            w2b = w2
        mu_out = K1v
        C = B.with_diag(B.add(K11, B.swap(K11), out="c_sym") * 0.5, K2v, out="c")
        # ---------------- M leg of the new source: S21 = S_sep + Rres (rank r) ---------
        S21 = B.zero_diag(K21 - rep21, out="s21")
        S_sep = B.zero_diag(e_b[:, None] * C_off * w1[None, :], out="ssep")
        Rres = S21 - S_sep
        Om = B.copy(w32[:, :r])
        Q = B.qr(Rres @ Om)
        Q = B.qr(Rres @ (B.swap(Rres) @ Q))
        Bm = B.swap(Q) @ Rres
        Rr = B.cat([B.swap(Bm) * 3.0, u_b[:, None], y_b[:, None], F1_b, F2_b], 1)
        Lr = B.cat([Q, w1sq[:, None], B.zeros((n, 1))], 1)
        st["Lst"] = Lr[None] if st["Lst"] is None else B.cat([st["Lst"], Lr[None]], 0)
        st["R1T"] = R1T_b[None] if st["R1T"] is None else B.cat([st["R1T"], R1T_b[None]], 0)
        st["R2T"] = R2T_b[None] if st["R2T"] is None else B.cat([st["R2T"], R2T_b[None]], 0)
        st["w2b"] = st["w2b"] + [w2b]
        st["dA"] = st["dA"] + [9.0 + w2b * w2b + 9.0 * e_b * e_b]
        st["dP"] = st["dP"] + [1.0 + S3c * S3c]
        st["c1"] = st["c1"] + [c1_b]
        st["c2"] = st["c2"] + [dgw]
        # young legs pre-scaled by w1 for the next transport family
        if k > ka:
            st["APhat"] = (B.mul(A_all[ka:], w1c[None], out=("aphatA", li % 2)),
                           B.mul(P_all[ka:], w1c[None], out=("aphatP", li % 2)))
        else:
            st["APhat"] = None
        st["g3s"] = st["g3s"] + [alpha_w * phi_w * (-1.0) / var]
        st["cvs"] = st["cvs"] + [Phi_w * var]
        st.update(alpha_prev=alpha_w, varpre_prev=var, K4v_prev=K4v,
                  D3c_prev=st.pop("D3c_prev_next", None) if k > 0 else B.zeros((n,)))
        st.update(mu=mu_out, C=C, g_prev=g_new, var_prev=K2v, K3v_prev=K3v,
                  lam_prev=float(LAM[min(li, len(LAM) - 1)]), w1_prev=w1,
                  newborn=dict(a_b=a_b, Rr=Rr, s=S3c, e=e_b))
        return mu_out, st

    # -----------------------------------------------------------------------------------
    def _vecs(self, names, vec):
        out = []
        for nm in names:
            v = None
            for f in nm.split("*"):
                v = vec[f] if v is None else v * vec[f]
            out.append(v)
        return self.B.stack(out, 0)

    def term1(self, prog, WT, vec):
        V = self._vecs(prog["v1names"], vec)
        return prog["IND1T"] @ ((prog["SL1"] @ WT) * (prog["SEL1"] @ V))      # (4, n)

    def term2(self, prog, WT, vec, mats):
        B, n = self.B, self.n
        V = self._vecs(prog["vnames"], vec)
        WL = (prog["SL2"] @ WT) * (prog["SELL"] @ V)
        WR = (prog["SR2"] @ WT) * (prog["SELR"] @ V)
        C, G, r = len(prog["classes"]), prog["G"], prog["rmax"]
        WLp = B.swap(B.reshape(prog["PL"] @ WL, (C, G, r, n)))
        WRp = B.reshape(prog["PR"] @ WR, (C, G, r, n))
        Ms = []
        for cls in prog["classes"]:
            if len(cls) == 0:
                Ms.append(self.ones2)
            elif len(cls) == 1:
                Ms.append(mats[cls[0]])
            else:
                Ms.append(B.mul(mats[cls[0]], mats[cls[1]], out=("term_m", cls)))
        Mst = B.stack(Ms, 0, out="term_ms")
        # one output group at a time: (C, n, n) products stay under the 100 MiB array cap
        out = []
        for g in range(G):
            LR = B.mm(WLp[:, g], WRp[:, g], out="term_lr")                     # (C, n, n)
            out.append(B.sum(B.mul(LR, Mst, out="term_lr"), 0, out=("term_pk2", g)))
        return out

    def wick(self, mu, var):
        """All ReLU Wick coefficients w(k, p) per neuron, (n, len(WICK_PAIRS))."""
        B, K = self.B, self.K
        sigma = B.sqrt(var)
        alpha = mu / sigma
        phi = B.pdf(alpha)
        Phi = B.cdf(alpha)
        APOW = B.power(alpha[:, None], K.APOW_E)        # alpha^0..5
        SB = B.power(sigma[:, None], K.SB_E)            # sigma^-6..4
        Wall = ((APOW @ K.C1) * phi[:, None] + (APOW @ K.C2) * Phi[:, None]) * (SB @ K.SELC)
        return Wall, alpha, phi, Phi

    # -----------------------------------------------------------------------------------
    def dslices(self, st, A, P, need_d21):
        """(3,) and (2,1) slices of the transported third cumulant. Per source b with legs
        A, P (and M = P d(s) + 3 A d(e) + Z L^T, X1 = 3A + Xt, Y1 = A d(w2) + Yt):
          D21 = sum_b LA_b A_b^T + LP_b P_b^T + thin terms,  D3 = row sums of Hadamard products.
        Old sources contract through their shared-basis factors."""
        B, r, rfb = self.B, self.r, self.rfb
        k, ka, kb = st["k"], st["ka"], st["kb"]
        W2B = B.stack(st["w2b"], 0)[:, None, :]
        Sb = B.stack(st["s"], 0)[:, None, :]
        Eb = B.stack(st["e"], 0)[:, None, :]
        Z, Lst = st["Z"], st["Lst"]
        Zr = Z[:, :, :r + 2]                     # residual leg columns (+ u, y)
        AP = B.mul(A, P, out="ap")
        PP = B.mul(P, P, out="pp")
        MP = B.mul(B.mm(Zr, B.swap(Lst), out="zl"), P, out="mp")
        MP = B.add(MP, B.mul(PP, Sb, out="t"), out="mp")
        MP = B.add(MP, B.mul(AP, Eb * 3.0, out="t"), out="mp")
        F1 = Z[:, :, r + 2:r + 2 + rfb]
        F2 = Z[:, :, r + 2 + rfb:]
        Xt = B.mm(F1, B.swap(st["R1T"]), out="xt")
        Yt = B.mm(F2, B.swap(st["R2T"]), out="yt")
        # LPadd = A*Yt + Xt*A*w2/3 + Xt*Yt/3
        LPadd = B.mul(A, Yt, out="lpadd")
        LPadd = B.add(LPadd, B.mul(B.mul(Xt, A, out="t"), W2B * (1.0 / 3.0), out="t"), out="lpadd")
        LPadd = B.add(LPadd, B.mul(Xt, Yt * (1.0 / 3.0), out="t"), out="lpadd")
        # D3 = rowsum(P * (3 A*A*w2 + MP + 3 LPadd)) + sum_k R y
        D3s = B.mul(B.mul(A, A, out="t"), W2B * 3.0, out="t")
        D3s = B.add(D3s, MP, out="t")
        D3s = B.add(D3s, B.mul(LPadd, 3.0, out="x3"), out="t")
        D3 = B.sum(B.mul(D3s, P, out="t"), (0, 2))
        C1s = B.stack(st["c1"], 0)
        C2s = B.stack(st["c2"], 0)
        Yk = Z[:, :, r + 1]
        R = (B.mm(AP, C1s[:, :, None]) + B.mm(PP, C2s[:, :, None]))[:, :, 0]
        D3 = D3 + B.sum(R * Yk, 0)
        if not need_d21:
            return D3, None
        # left factors LA (right factor A) and LP (right factor P), interleaved (k, 2, n, n)
        # flopscope: LA / LP are written straight into the two slots of the pooled (k, 2, n, n)
        # buffer the hub contraction reads; torch: fresh tensors, stacked at the end
        la, lp = "la", "lp"
        LA = B.mul(AP, W2B * 2.0, out=la)
        LA = B.add(LA, B.mul(PP, Eb, out="t"), out=la)
        LA = B.add(LA, B.mul(B.mul(Xt, P, out="t"), W2B * (1.0 / 3.0), out="t"), out=la)
        PYt = B.mul(P, Yt, out="pyt")
        LA = B.add(LA, PYt, out=la)
        LP = B.mul(B.mul(A, A, out="t"), W2B, out=lp)
        LP = B.add(LP, B.mul(PP, Sb * (1.0 / 3.0), out="t"), out=lp)
        LP = B.add(LP, B.mul(MP, 2.0 / 3.0, out="t"), out=lp)
        LP = B.add(LP, LPadd, out=lp)
        X3 = B.add(B.mul(A, C1s[:, None, :], out="t"), B.mul(P, C2s[:, None, :], out="x3"), out="t")
        LP = B.add(LP, B.mul(X3, (Yk * (2.0 / 3.0))[:, :, None], out="t"), out=lp)
        # thin right-factor terms of the feedback legs
        GY = B.add(B.mul(B.mul(Xt, P, out="t"), 1.0 / 3.0, out="t"), AP, out="t")
        GYR = B.mm(GY, st["R2T"])
        GX = B.mul(B.add(B.mul(AP, W2B, out="t"), PYt, out="t"), 1.0 / 3.0, out="t")
        GXR = B.mm(GX, st["R1T"])
        # all thin right-factor terms in one product, column-aligned with Z:
        # (PP L / 3) Z_r^T + (GX R1^T) F1^T + (GY R2^T) F2^T
        D21 = B.thinhub(B.cat([B.mm(PP, Lst) * (1.0 / 3.0), GXR, GYR], 2), Z)
        D21 = D21 + (B.swap(R) @ Yk) * (1.0 / 3.0)
        if k > ka:
            D21 = D21 + B.hub2(LA[ka:], LP[ka:], A[ka:], P[ka:], out="hub")
        if ka > 0:
            inner = None
            if ka > kb:
                FAP = st["FAP"]
                inner = B.hub2(LA[kb:ka], LP[kb:ka], FAP[:, 0], FAP[:, 1], out="inner1", kind="inner")
            if kb > 0:
                FAP2 = st["FAP2"]
                lift = B.hub2(LA[:kb], LP[:kb], FAP2[:, 0], FAP2[:, 1], out="inner2",
                              kind="inner") @ B.swap(st["U"])
                inner = lift if inner is None else inner + lift
            D21 = D21 + B.lmm(inner, B.swap(st["Qc"])[None], out="d21q")[0]
        return D3, B.zero_diag(D21, out="d21")


# ---------------------------------------------------------------------------------------
# The learned operator F_theta (section 3 of the write-up)
# ---------------------------------------------------------------------------------------
NETS = {  # name: (in, out); every net is in -> 64 -> 64 -> out with GELU
    "self": (12, 2), "msg1": (14, 16), "msg2": (174, 16), "in": (334, 2),
    "pairL": (14, 36), "pairR": (14, 36), "out": (334, 6),
    "covL": (14, 36), "covR": (14, 36), "last": (14, 1),
}
HID = 64
STAGE1 = ("self", "msg1", "msg2", "in", "pairL", "pairR")
STAGE2 = ("out", "covL", "covR", "last")
GROUP14 = ("msg1", "pairL", "pairR", "covL", "covR")    # share input x_i (14), out padded to 36
GROUP334 = ("in", "out")                                  # share input x_msg (334), out padded to 6
N_EDGE = 10
N_PAIRCH = 9


def n_params() -> int:
    tot = 0
    for i, o in NETS.values():
        tot += i * HID + HID + HID * HID + HID + HID * o + o
    return tot


class Operator:
    """Hooks implementing Delta_in / Delta_out with the networks of Table 3.

    params: dict name -> [W1 (in,64), b1 (64,), W2 (64,64), b2, W3 (64,out), b3] (backend
    arrays); xmean / xstd: (14,) fixed feature standardization. ``active`` = 0-based layer
    indices l < L-1 where steps 1-7 run; the last layer always runs f_self and f_last when
    present. ``nets`` = the set of networks in use (stage 1 / stage 2)."""

    def __init__(self, B, params, xmean, xstd, active, nets):
        self.B, self.p = B, params
        self.xmean, self.xstd = xmean, xstd
        self.active = set(active)
        self.nets = set(nets)
        self.ctx_store = {}

    # ---- batched MLP evaluation: G nets on the same input, outputs padded to dout ----
    def mlp(self, names, x, dout):
        B = self.B
        if len(names) == 1:
            W1, b1, W2, b2, W3, b3 = self.p[names[0]]
            h = B.gelu(x @ W1 + b1)
            h = B.gelu(h @ W2 + b2)
            return (h @ W3 + b3)[None]
        W1, b1, W2, b2, W3, b3 = B.stack_params([self.p[nm] for nm in names], dout,
                                                key=("grp",) + tuple(names))
        h = B.gelu(x[None] @ W1 + b1)          # (G, n, 64)
        h = B.gelu(h @ W2 + b2)
        return h @ W3 + b3                     # (G, n, dout)

    def features(self, ctx):
        B = self.B
        s = ctx["s"]
        s2 = ctx["var"]
        s3 = s2 * s
        s4 = s2 * s2
        WW = ctx["WW"]
        # post-ReLU variances of near-dead neurons are tiny (or round to <= 0): regularize the
        # denominators of the scale-free ratios with 1% of the layer-mean variance
        vprev = B.maximum(ctx["var_prev"], 0.0) + B.mean(ctx["var_prev"]) * 0.01
        gprev = ctx["g_prev"]
        k3prev = ctx["K3v_prev"]
        rws = B.sum(WW, 1)
        W3 = B.mul(WW, ctx["W"], out="w3f")
        x = [ctx["mu"] / s, ctx["D3"] / s3, ctx["dG"] / s4,
             s * 0.0 + ctx["li"] / (ctx["L"] - 1.0),
             (B.mul(WW, WW, out="w4f") @ gprev) / s4,
             ctx["t_g"] / s4,
             (W3 @ k3prev) / s3,
             ctx["t_v"] * ctx["lam"] / s4,
             (WW @ (gprev / (vprev * vprev))) / rws,
             (WW @ (k3prev / (vprev * B.sqrt(vprev)))) / rws,
             (WW @ ctx["var_prev"]) / s2,
             s * 0.0 + ctx["lam"] * 100.0]
        return x

    def std(self, cols):
        B = self.B
        X = B.stack(cols, 1)
        return B.clip((X - self.xmean[None, :len(cols)]) / self.xstd[None, :len(cols)], -8.0, 8.0)

    def pre(self, ctx):
        B = self.B
        li, last = ctx["li"], ctx["last"]
        if not (last or li in self.active):
            return None
        s = ctx["s"]
        s3 = ctx["var"] * s
        s4 = ctx["var"] * ctx["var"]
        cols = self.features(ctx)
        D3, dG = ctx["D3"], ctx["dG"]
        if "self" in self.nets:
            o = self.mlp(["self"], self.std(cols), 2)[0]
            dG = dG + o[:, 0] * s4
            D3 = D3 + o[:, 1] * (0.1 * s3)
        cols = cols + [D3 / s3, dG / s4]
        x = self.std(cols)                                     # (n, 14)
        ctx["x"] = x
        if last:
            return D3, dG, None
        # ---- pair statistics, edge weights, messages ----
        n = x.shape[0]
        sr = 1.0 / s
        rho = B.mul(B.mul(ctx["C_off"], sr[:, None], out="rho"), sr[None, :], out="rho")
        t = B.mul(B.mul(ctx["D21"], (sr * sr)[:, None], out="tt"), sr[None, :], out="tt")
        tT = B.swap(t)
        rho2 = B.mul(rho, rho, out="rho2")
        E = B.stack([rho, rho2, B.mul(rho2, rho, out="e2"), B.abs(rho), t, tT,
                     B.mul(t, t, out="e6"), B.mul(tT, tT, out="e7"),
                     B.mul(rho, t, out="rhot"), B.mul(rho, tT, out="rhotT")], 0, out="edges")
        enorm = 1.0 / (B.mean(B.abs(B.reshape(E, (N_EDGE, n * n))), 1) * n)    # (10,)
        g14 = [nm for nm in GROUP14 if nm in self.nets]
        O14 = self.mlp(g14, x, 36)
        out14 = {nm: O14[i] for i, nm in enumerate(g14)}
        m1 = self.messages(E, enorm, out14["msg1"][:, :16])
        x174 = B.cat([x, m1], 1)
        m2 = self.messages(E, enorm, self.mlp(["msg2"], x174, 16)[0])
        xmsg = B.cat([x174, m2], 1)                            # (n, 334)
        g334 = [nm for nm in GROUP334 if nm in self.nets]
        O334 = self.mlp(g334, xmsg, 6)
        out334 = {nm: O334[i] for i, nm in enumerate(g334)}
        ctx["out334"] = out334
        ctx["out14"] = out14
        oi = out334["in"]
        dG = dG + oi[:, 0] * s4
        D3 = D3 + oi[:, 1] * s3
        # pair channels B_c(i, j)
        lam_s2 = ctx["lam"] * (sr * sr)
        k31n = B.mul(rho, lam_s2[None, :], out="k31n")    # kappa_iiij / (s_i s_j^3) = lam rho / s_j^2
        # the seven channels shared by f_pair and f_cov, stacked once: (7, n, n)
        common = B.stack([rho, rho2, t, tT, E[8], E[9], B.mul(t, tT, out="ttT")], 0, out="pairch")
        ctx["pair_common"] = common
        ctx["rho"] = rho
        fpair = self.pairfun(out14["pairL"], out14["pairR"], k31n, common, key="pair")
        kap31 = B.zero_diag(B.mul(B.mul(fpair, s[:, None], out="fp"), s3[None, :], out="fp"), out="fp")
        return D3, dG, kap31

    def messages(self, E, enorm, F):
        B = self.B
        M = B.mm(E, F) * enorm[:, None, None]                          # (10, n, 16)
        M = B.swapaxes01(M)
        return B.reshape(M, (M.shape[0], M.shape[1] * M.shape[2]))

    def pairfun(self, Lo, Ro, ch1, common, key):
        """sum_c B_c(i,j) <l_c(x_i), r_c(x_j)> with l, r in R^{9 x 4} per neuron and the
        channels B = (1, ch1, common[0..6])."""
        B = self.B
        n = Lo.shape[0]
        Lr = B.swapaxes01(B.reshape(Lo, (n, N_PAIRCH, 4)))   # (9, n, 4)
        Rr = B.swapaxes01(B.reshape(Ro, (n, N_PAIRCH, 4)))
        LR = B.mm(Lr, B.swap(Rr), out=key + "_lr")         # (9, n, n)
        acc = B.sum(B.mul(LR[2:], common, out=key + "_lc"), 0)
        return acc + LR[0] + LR[1] * ch1

    def post(self, ctx, q):
        B = self.B
        li = ctx["li"]
        if li not in self.active or "out" not in self.nets:
            return None
        o = ctx["out334"]["out"]
        s = ctx["s"]
        sig = B.sqrt(B.maximum(q["K2v"], 1e-12))
        sig2 = sig * sig
        K1v = q["K1v"] + o[:, 0] * (1e-3 * s)
        K2v = q["K2v"] * (1.0 + o[:, 1] * 1e-3)
        g = q["g"] + o[:, 5] * (1e-2 * sig2 * sig2)
        S3c = q["S3c"] + o[:, 2] * (1e-2 * sig2 * sig)
        w2 = q["w2"] * (1.0 + o[:, 3] * 1e-2)
        e = q["e"] * (1.0 + o[:, 4] * 1e-2)
        K11 = q["K11"]
        if "covL" in self.nets:
            fcov = self.pairfun(ctx["out14"]["covL"], ctx["out14"]["covR"], ctx["rho"],
                                ctx["pair_common"], key="cov")
            fsym = B.add(fcov, B.swap(fcov), out="fcs")
            dS = B.zero_diag(B.mul(B.mul(fsym, (sig * 5e-4)[:, None], out="fcs"), sig[None, :], out="fcs"), out="fcs")
            K11 = K11 + dS
        return K1v, K2v, K11, g, S3c, w2, e

    def last(self, ctx, pk1v):
        if "last" not in self.nets:
            return pk1v
        o = self.mlp(["last"], ctx["x"], 1)[0]
        return pk1v + o[:, 0] * (1e-3 * ctx["s"])


# ---------------------------------------------------------------------------------------
# The team's v5b operator (learned_relu_step v5b_mp_l8): input corrections only
# ---------------------------------------------------------------------------------------
MP_EDGES = 10
MP_PAIRCH = 9


class OperatorV5B:
    """Input-correction model of the submitted v5b (the PDF's "input-correction model,
    v5b (layers 9-15)", MSE 1.75e-8 at 25.3% budget), with its trained weights k34.npz
    (f_self) and mp.npz (f_msg / f_pair / f_in). Layout:

    * f_self (layers >= 2): 12 features -> 64 -> 64 -> 2 (GELU), standardized by (mu, sd):
      alpha, D3/s^3, g4/s^4 (g4 = m dG), own-kurtosis transport sum_a W_ia^4 K4_a / s^4, the
      off-diagonal variance share t_v / var, depth li/15, five Stein-type terms built from the
      newborn A leg v_ai ~ Cov(z'_a, z_i) (S1, M1, T1 with v^3 weights, S3, M3 with v^2), and
      the transported kappa4 "star" of the older sources. Outputs: g3 += 0.1 o_1, g4 += o_0.
    * message passing (layers >= mp_from, not the trimmed last layer): x = (raw features,
      g4', g3', 1) in R^15; net A (15 -> 192 -> 192 -> 88) gives the 16-dim first message
      and the 9 x 4 left / right pair factors, net B (175 -> 64 -> 64 -> 16) the second
      message, net O (335 -> 64 -> 64 -> 2) the residual on (g4', g3'); edges = 10 fixed
      functions of rho and t = D21 / (s_i^2 s_j), each scaled by n / sum |E|; the pair term
      adds sum_c B_c (L_c R_c^T) s_i s_j^3 to the (3,1) slice (B_0 = 1, B_1 = the chain's own
      (3,1) slice / (s_i s_j^3), then t t^T, rho, rho^2, t, t^T, rho t, rho t^T).
    * The new source's birth keeps the uncorrected kappa4 diagonal and scales the corrected
      D3 (birth_corrected_dG = False, d3w_corrected = True), exactly as v5b."""

    birth_corrected_dG = False
    d3w_corrected = True

    def __init__(self, B, k34, mp=None, mp_from=8):
        self.B = B
        self.k = k34          # dict: mu, sd, W0, b0, W1, b1, W2, b2
        self.mp = mp          # dict: {A,B,O}_{W,b}{0,1,2} or None
        self.mp_from = mp_from
        self.star = int(k34["mu"].shape[0]) == 12
        self.nets = ("v5b-k34",) + (("v5b-mp",) if mp is not None else ())
        self.active = set()

    def _mlp(self, x, p, pre):
        B = self.B
        h = B.gelu(x @ p[pre + "W0"] + p[pre + "b0"])
        h = B.gelu(h @ p[pre + "W1"] + p[pre + "b1"])
        return h @ p[pre + "W2"] + p[pre + "b2"]

    def _mlp_mp(self, x, nm):
        # bias of the first layer is the last row of W0 (x carries a ones column)
        B, p = self.B, self.mp
        h = B.gelu(x @ p[nm + "_W0"])
        h = B.gelu(h @ p[nm + "_W1"] + p[nm + "_b1"])
        return h @ p[nm + "_W2"] + p[nm + "_b2"]

    def features(self, ctx):
        B = self.B
        li = ctx["li"]
        var = ctx["var"]
        s1 = ctx["s"]
        s3 = var * s1
        s4 = var * var
        mu = ctx["mu"]
        WW = ctx["WW"]
        w32 = ctx["w32"]
        ap, vp, D3p = ctx["alpha_prev"], ctx["varpre_prev"], ctx["D3c_prev"]
        k4p = ctx["g4c_prev"]
        g4 = ctx["dG"] * METRIC_C
        own4 = (B.mul(WW, WW, out="w4f") @ ctx["K4v_prev"]) / s4
        ph = B.pdf(ap)
        Ph = B.cdf(ap)
        ivp = 1.0 / vp
        ivp2 = ivp * ivp
        A_new = ctx["A_all"][-1]
        # v_ai ~ Cov(z'_a, z_i): newborn A leg plus the diagonal (w1 var) part
        v = B.add(B.swap(A_new), (ctx["w1_prev"] * vp)[:, None] * w32, out="v5b_v")
        wv2 = B.mul(B.mul(w32, v, out="v5b_wv2"), v, out="v5b_wv2")
        wv3 = B.mul(wv2, v, out="v5b_wv3")
        q3 = ph / B.sqrt(vp)
        S1 = ((-ap * ph * ivp) @ wv3) / s4
        M1 = ((Ph * k4p * (ivp2 * ivp)) @ wv3) / s4
        T1 = ((q3 * D3p * ivp2 * 3.0) @ wv3) / s4
        S3 = (q3 @ wv2) / s3
        M3 = ((Ph * D3p * ivp2) @ wv2) / s3
        cols = [mu / s1, ctx["D3"] / s3, g4 / s4, own4, ctx["t_v"] / var,
                s1 * 0.0 + li / 15.0, S1, M1, T1, S3, M3]
        if self.star:
            cols.append(self._star(ctx) / s4)
        return B.stack(cols, 1), s1, s3, s4

    def _star(self, ctx):
        """sum over sources older than the newest of sum_j P_kij g3_kj (A + P d(cv_k))^3_ij."""
        B = self.B
        ko = ctx["A_all"].shape[0] - 1
        if ko < 1:
            return ctx["s"] * 0.0
        A, P = ctx["A_all"][:ko], ctx["P_all"][:ko]
        cv = B.stack(ctx["cvs"][:ko], 0)
        g3 = B.stack(ctx["g3s"][:ko], 0)
        V = B.add(B.mul(P, cv[:, None, :], out="t"), A, out="t")
        U = B.mul(B.mul(V, V, out="x3"), V, out="x3")
        U = B.mul(U, P, out="x3")
        return B.sum(B.mm(U, g3[:, :, None])[:, :, 0], 0)

    def pre(self, ctx):
        B = self.B
        li = ctx["li"]
        F, s1, s3, s4 = self.features(ctx)
        k = self.k
        o = self._mlp((F - k["mu"][None, :]) / k["sd"][None, :], k, "")
        y4 = F[:, 2] + o[:, 0]
        y3 = F[:, 1] + o[:, 1] * 0.1
        if self.mp is None or li < self.mp_from or ctx["last"]:
            return y3 * s3, (y4 * s4) * (1.0 / METRIC_C), None
        n = F.shape[0]
        x = B.cat([F, B.stack([y4, y3, s1 * 0.0 + 1.0], 1)], 1)          # (n, 15)
        is1 = 1.0 / s1
        is2 = is1 * is1
        rho = B.mul(B.mul(ctx["C_off"], is1[:, None], out="rho"), is1[None, :], out="rho")
        t = B.mul(B.mul(ctx["D21"], is2[:, None], out="tt"), is1[None, :], out="tt")
        tT = B.swap(t)
        rho2 = B.mul(rho, rho, out="rho2")
        rt = B.mul(rho, t, out="rhot")
        rtT = B.mul(rho, tT, out="rhotT")
        E = B.stack([rho, rho2, t, tT, rt, rtT, B.mul(rho2, rho, out="e2"), B.abs(rho, out="eabs"),
                     B.mul(t, t, out="e6"), B.mul(tT, tT, out="e7")], 0, out="edges")
        inv = float(n) / B.sum(B.sum(B.abs(E, out="edges_abs"), 2), 1)              # (10,)
        a = self._mlp_mp(x, "A")                                                # (n, 88)
        m1 = self._msg(E, inv, a[:, :16])
        m2 = self._msg(E, inv, self._mlp_mp(B.cat([x, m1], 1), "B"))
        oo = self._mlp_mp(B.cat([x, m1, m2], 1), "O")
        g4n = y4 + oo[:, 0]
        g3n = y3 + oo[:, 1]
        # pair term on the (3,1) slice
        chain31 = B.mul(B.mul(ctx["C_off"], (is1 * ctx["lam"])[:, None], out="k31n"),
                        (is2 * is1)[None, :], out="k31n")               # (lam C_off) / (s_i s_j^3)
        Bp = B.stack([chain31, B.mul(t, tT, out="ttT"), rho, rho2, t, tT, rt, rtT], 0, out="pairch")
        Lf = B.swapaxes01(B.reshape(a[:, 16:52], (n, MP_PAIRCH, 4)))           # (9, n, 4)
        Rf = B.swapaxes01(B.reshape(a[:, 52:88], (n, MP_PAIRCH, 4)))
        LR = B.mm(Lf, B.swap(Rf), out="pair_lr")                                # (9, n, n)
        Dl = B.sum(B.mul(LR[1:], Bp, out="pair_lc"), 0) + LR[0]
        kap31 = B.zero_diag(B.mul(B.mul(Dl, s1[:, None], out="fp"), s3[None, :], out="fp"), out="fp")
        return g3n * s3, (g4n * s4) * (1.0 / METRIC_C), kap31

    def _msg(self, E, inv, F):
        B = self.B
        M = B.mm(E, F) * inv[:, None, None]                                    # (10, n, 16)
        M = B.swapaxes01(M)
        return B.reshape(M, (M.shape[0], M.shape[1] * M.shape[2]))

    def post(self, ctx, q):
        return None

    def last(self, ctx, pk1v):
        return pk1v
