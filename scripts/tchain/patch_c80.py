import sys
p = sys.argv[1]
s = open(p).read()
def rep(old, new, cnt=1):
    global s
    c = s.count(old)
    assert c == cnt, (c, old[:80])
    s = s.replace(old, new)

rep("XC3, XMARG2 = 0.0, 1.0\n", """XC3, XMARG2 = 0.0, 1.0
# c80: the leaders' construction (18248) on our engine, behind flags.
XAUG = _os.environ.get("XAUG", "0") == "1"       # ARC current AUGMENT term table (augment=True) incl. the (3,1) output
XG = XAUG and _os.environ.get("XG", "0") == "1"  # carried kappa4 matrix core G <- W G W^T, produced by the r=1 projection
XEXT = float(_os.environ.get("XEXT", "0"))       # scale-mode extraction strength f (0 = off)
XTK34 = _os.environ.get("XTK34", "1") == "1"     # keep the learned k34 / MP steps in XG mode
XKNON = _os.environ.get("XKNON", "0") == "1"     # keep the cn71 closure knobs in XG mode
if XAUG:
    from aug_tables import WICK_PAIRS_A, WICK_UNIFIED_A, TERM_SPECS_A  # noqa: E402
    WICK_PAIRS, WICK_UNIFIED = WICK_PAIRS_A, WICK_UNIFIED_A
    TERM_SPECS = {k_: list(v_) for k_, v_ in TERM_SPECS_A.items()}
""")
rep("""_I = WICK_PAIRS.index
TERM_SPECS[(1, 1)].append(('wk431', 'ones2', _I((1, 1)), _I((3, 1)), 1.0 / 3.0))
TERM_SPECS[(2, 1)].append(('wk431', 'ones2', _I((1, 2)), _I((3, 1)), 1.0 / 6.0))
TERM_SPECS[(2, 1)].append(('wk431', 'ones2', _I((3, 2)), _I((1, 1)), 1.0 / 6.0))
TERM_SPECS[(2, 2)].append(('wk431', 'ones2', _I((1, 2)), _I((3, 2)), 1.0 / 3.0))
""", """_I = WICK_PAIRS.index
if not XAUG:
    TERM_SPECS[(1, 1)].append(('wk431', 'ones2', _I((1, 1)), _I((3, 1)), 1.0 / 3.0))
    TERM_SPECS[(2, 1)].append(('wk431', 'ones2', _I((1, 2)), _I((3, 1)), 1.0 / 6.0))
    TERM_SPECS[(2, 1)].append(('wk431', 'ones2', _I((3, 2)), _I((1, 1)), 1.0 / 6.0))
    TERM_SPECS[(2, 2)].append(('wk431', 'ones2', _I((1, 2)), _I((3, 2)), 1.0 / 3.0))
_MG5 = tuple(_I(p_) for p_ in ((5, 1), (5, 2)) if p_ in WICK_PAIRS)
_MG6 = tuple(_I(p_) for p_ in ((6, 1), (6, 2)) if p_ in WICK_PAIRS)
""")
rep("D2_IPS = [(1, 1), (2, 1), (2, 2)]\n", "D2_IPS = [(1, 1), (2, 1), (2, 2)] + ([(3, 1)] if XAUG else [])\n")
rep("""                 "wk431"}""", """                 "wk431", "wk431T"}""")
rep("""({"wk431"} if mode == 2 else set())""", """({"wk431", "wk431T"} if mode == 2 else set())""")
rep("""           if ('d3col' in (a, b) and wl in (16, 17)) or ('g4col' in (a, b) and wl in (18, 19))]""",
    """           if ('d3col' in (a, b) and wl in _MG5) or ('g4col' in (a, b) and wl in _MG6)]""")
rep("""           if ('d3row' in (a, b) and wr in (16, 17)) or ('g4row' in (a, b) and wr in (18, 19))]""",
    """           if ('d3row' in (a, b) and wr in _MG5) or ('g4row' in (a, b) and wr in _MG6)]""")
rep("""                cA=6.0 / (n + 4.0), cI=-3.0 / ((n + 2.0) * (n + 4.0)))""",
    """                cA=6.0 / (n + 4.0), cS=12.0 / (n + 4.0), cI=-3.0 / ((n + 2.0) * (n + 4.0)))""")
# carried core state
rep("""        g_prev = None  # post-ReLU kappa4 diagonal core of the previous layer
""", """        g_prev = None  # post-ReLU kappa4 diagonal core of the previous layer
        G_post = None  # c80: post-ReLU kappa4 matrix core of the previous layer (XG)
        G_po = None    # c80: off-diagonal pre-activation core of this layer (XG)
        ext = None     # c80: this layer's extraction (eps, m, v_b, C_b off)
""")
# kappa4 use side
rep("""                if regen:
                    # F68: exact transported diagonal of the regenerated core
                    # G = diag(g) + lam C_off:  dG = (W*W) g + lam (diag(C_pre)
                    # - (W*W) var).  NOT an einsum: same-object repeated operands
                    # crash flopscope (F45).
                    WW = W * W
                    if BETA != 0.0:""", """                if regen:
                    # F68: exact transported diagonal of the regenerated core
                    # G = diag(g) + lam C_off:  dG = (W*W) g + lam (diag(C_pre)
                    # - (W*W) var).  NOT an einsum: same-object repeated operands
                    # crash flopscope (F45).
                    WW = W * W
                    if XG:
                        # c80: carried core, G_pre = W G_post W^T (x@w: w^T G w)
                        t_v = var - WW @ var_prev
                        if trim:
                            dG = fnp.sum(w32 * (G_post @ w32), axis=0)
                        else:
                            G_pre = w32.T @ (G_post @ w32)
                            dG = fnp.diag(G_pre) * 1.0
                            G_po = _zero_diag(G_pre)
                            lam_prev = fnp.sum(G_po * C_off) / fnp.sum(C_off * C_off)   # lambda_eff
                    elif BETA != 0.0:""")
rep("""                    wk431 = None if trim else C_off * (0.5 * METRIC_C * lam_prev)
                    if learned:""", """                    if XG:
                        wk431 = None if trim else G_po * (0.5 * METRIC_C)
                    else:
                        wk431 = None if trim else C_off * (0.5 * METRIC_C * lam_prev)
                    if learned and (XTK34 or not XG):""")
rep("""                        _kb = "F" if trim else ("L" if li >= MP_LI_MIN else "E")
                        if KN["D3" + _kb] != 1.0:""", """                        _kb = "F" if trim else ("L" if li >= MP_LI_MIN else "E")
                        _knon = (not XG) or XKNON
                        if not _knon:
                            pass
                        elif KN["D3" + _kb] != 1.0:""")
rep("""                        if KN["G4" + _kb] != 1.0: g4row = g4row * KN["G4" + _kb]
                        if _kb != "F" and wk431 is not None and KN["31" + _kb] != 1.0: wk431 = wk431 * KN["31" + _kb]
                        if _kb != "F" and D21 is not None and KN["21" + _kb] != 1.0: D21 = D21 * KN["21" + _kb]""",
    """                        if _knon and KN["G4" + _kb] != 1.0: g4row = g4row * KN["G4" + _kb]
                        if _knon and _kb != "F" and wk431 is not None and KN["31" + _kb] != 1.0: wk431 = wk431 * KN["31" + _kb]
                        if _knon and _kb != "F" and D21 is not None and KN["21" + _kb] != 1.0: D21 = D21 * KN["21" + _kb]""")
rep("""                        if KN["S22"] != 0.0 and not trim:""", """                        if _knon and KN["S22"] != 0.0 and not trim:""")
# term program bases
rep("""            if XC3 != 0.0 and C_off is not None:
                b2["c3"] = C_off * C_off * C_off""", """            if (XC3 != 0.0 or XAUG) and C_off is not None:
                b2["c3"] = C_off * C_off * C_off""")
rep("""                    if regen and not NO_WK431:
                        b2["wk431"] = wk431""", """                    if regen and not NO_WK431:
                        b2["wk431"] = wk431
                        b2["wk431T"] = wk431.T""")
rep("""                pk22 = _zero_diag((pk22 + pk22.T) * 0.5)
""", """                pk22 = _zero_diag((pk22 + pk22.T) * 0.5)
                pk31 = _zero_diag(PK2[3]) if XAUG else None
""")
rep("""            K22 = pk22 + pk11 * ((c1 * r1) * 4.0 - pk11 * 2.0) - (S + S.T) * 2.0
""", """            K22 = pk22 + pk11 * ((c1 * r1) * 4.0 - pk11 * 2.0) - (S + S.T) * 2.0
            if XAUG:
                # c80: K(3,1) = pK31 - 3 m_i pK21 - 3 pK2_i pK11 + 6 m_i^2 pK11 (PK2K, augment table)
                K31 = (pk31 - (c1 * 3.0) * pk21 - fnp.reshape(p2 * 3.0, (-1, 1)) * pk11
                       + (c1 * c1 * 6.0) * pk11)
""")
# feed at birth with the exact core
rep("""                           + fnp.reshape(y_b * (2.0 / 3.0), (-1, 1)) * (a_b * fnp.reshape(c1_b, (1, -1)))""",
    """                           + fnp.reshape(y_b * (2.0 / 3.0), (-1, 1))
                           * ((w1col * G_po * fnp.reshape(w1, (1, -1))) if XG else (a_b * fnp.reshape(c1_b, (1, -1))))""")
# extraction: eps fit and covariance part
rep("""            S21 = _zero_diag(K21 - rep21)
""", """            S21 = _zero_diag(K21 - rep21)
            ext = None
            if XEXT > 0.0 and riders:
                # c80: coherent scale-mode extraction (18248 s3): eps from the post-ReLU (2,1) slice's projection
                # on S_ij = v_i m_j + 2 m_i C_ij, then C <- (C - eps m m^T)/(1+eps); the kappa3 part rides the
                # newborn's thin legs (below), the kappa4 part is removed from G_post.
                mc_ = fnp.reshape(K1v, (-1, 1))
                mr_ = fnp.reshape(K1v, (1, -1))
                Cpo_ = _zero_diag((K11 + K11.T) * 0.5)
                S_ = _zero_diag(fnp.reshape(K2v, (-1, 1)) * mr_ + (mc_ * 2.0) * Cpo_)
                eps_ = (fnp.sum(K21 * S_) / fnp.sum(S_ * S_)) * (0.5 * XEXT)
                ie_ = 1.0 / (1.0 + eps_)
                mm_ = mc_ * mr_
                fnp.subtract(C, mm_ * eps_, out=C)
                fnp.multiply(C, ie_, out=C)
                vb_ = fnp.diag(C) * 1.0
                Cbo_ = _zero_diag((Cpo_ - mm_ * eps_) * ie_)
                ext = (eps_, K1v, vb_, Cbo_)
                if _os.environ.get("XEXT_DBG", "0") == "1":
                    print("ext layer", li, "eps", float(eps_))
""")
# M-leg thin column for the diagonal-C part of the kappa3 mode
rep("""            Rr_full = fnp.concatenate([Bm.T * 3.0, fnp.reshape(u_b, (-1, 1)),
                                       fnp.reshape(y_b, (-1, 1))], axis=1)
            Lr_full = fnp.concatenate([Q, fnp.reshape(w1sq, (-1, 1)),
                                       fnp.zeros((n, 1), dtype=f32)], axis=1)""",
    """            Rr_full = fnp.concatenate([Bm.T * 3.0, fnp.reshape(u_b, (-1, 1)),
                                       fnp.reshape(y_b, (-1, 1))], axis=1)
            Lr_full = fnp.concatenate([Q, fnp.reshape(w1sq, (-1, 1)),
                                       fnp.zeros((n, 1), dtype=f32)], axis=1)
            if XEXT > 0.0 and riders:
                # c80: -2 eps Sym(diag(v_b) x m): M-type term u v^T with u = -6 eps m, v = v_b (last Z/L column)
                Rr_full = fnp.concatenate([Rr_full, fnp.reshape(ext[1] * (-6.0 * ext[0]), (-1, 1))], axis=1)
                Lr_full = fnp.concatenate([Lr_full, fnp.reshape(ext[2], (-1, 1))], axis=1)""")
# feedback thin column for the off-diagonal-C part
rep("""            Ff_b = fnp.concatenate([F1_b, F2_b], axis=1) if rfb > 0 else None""",
    """            if XEXT > 0.0 and riders and rfb > 0:
                # c80: -2 eps Sym(C_b x m), C_b off ~ d(w1) C_off d(w1) = a_b d(w1): Y1 += -2 eps m w1^T
                F2_b = fnp.concatenate([F2_b, fnp.reshape(ext[1] * (-2.0 * ext[0]), (-1, 1))], axis=1)
                R2T_b = fnp.concatenate([R2T_b, fnp.reshape(w1, (-1, 1))], axis=1)
            Ff_b = fnp.concatenate([F1_b, F2_b], axis=1) if rfb > 0 else None""")
# production of the carried core + kappa4 part of the extraction
rep("""                var_prev = K2v
                lam_prev = float(LAM[min(li, len(LAM) - 1)])""", """                var_prev = K2v if ext is None else ext[2]
                lam_prev = float(LAM[min(li, len(LAM) - 1)])
                if XG:
                    # c80: r=1 harmonic projection of the post-ReLU slices (no K211 hub content yet)
                    G_post = (K31 + K31.T) * (0.5 * float(st["cS"])) + fnp.diagflat(g_prev)
                    if ext is not None:
                        eps_, _m, vb_, Cbo_ = ext
                        t4_ = vb_ * vb_ * 12.0
                        r22_ = (vb_ * (fnp.sum(vb_) - vb_) + ((Cbo_ * Cbo_) @ ones_n) * 2.0) * 4.0
                        hd_ = (t4_ + r22_) * float(st["cA"]) + (fnp.sum(t4_) + fnp.sum(r22_)) * float(st["cI"])
                        hoff_ = (fnp.reshape(vb_, (-1, 1)) + fnp.reshape(vb_, (1, -1))) * Cbo_ * (6.0 * float(st["cS"]))
                        G_post = G_post - (hoff_ + fnp.diagflat(hd_)) * eps_""")
open(p, "w").write(s)
print("patched", p)
