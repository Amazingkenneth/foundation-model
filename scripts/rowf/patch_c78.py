"""patch_c78.py <estimator.py>: final-layer row restriction (ROWF) on the cn71 estimator, in place."""
import sys

p = sys.argv[1]
s = open(p).read()


def rep(old, new, cnt=1):
    global s
    c = s.count(old)
    assert c == cnt, (c, old[:120])
    s = s.replace(old, new)


# ---- flags ----
rep('''RW_TAU = float(_os.environ.get("RW_TAU", "0.01"))''',
    '''# c78: final-layer row restriction. The final layer consumes D3 only through the Edgeworth term of the mean, whose
# weight per unit skew is sigma |alpha| phi(alpha) (ROWMODE imp; band = min(Phi, 1 - Phi)). The young / tier transports,
# the thin legs and the D3 slices of the last layer run on the ROWF-complement of rows ranked highest (rounded up to
# 32 rows); dropped rows get D3 = 0 and no star feature. The newborn A leg stays full (the k34 Stein features read it).
ROWF = float(_os.environ.get("ROWF", "0"))
ROWMODE = _os.environ.get("ROWMODE", "imp")
RW_TAU = float(_os.environ.get("RW_TAU", "0.01"))''')

# ---- per-layer: rank the rows from W C before the transports ----
rep('''            skip_src = trim and NO_SRC_LAST
''', '''            skip_src = trim and NO_SRC_LAST
            rsel = (trim and riders and ROWF > 0.0 and n == 1024 and A_st is not None and newborn is not None\n                    and not skip_src)
''')
rep('''            if (A_st is not None or newborn is not None) and not skip_src:
                WD = W * fnp.reshape(w1_prev, (1, -1))''',
    '''            TW = None
            mr = n
            if rsel:
                # c78: W C first (the final variance ranks the rows); the newborn block below skips it
                TW = LeftTree(W[None], SLEV, n, "TV")
                _mmc(TW, STG[1:2], SLEV, STGo[1:2])
                _sv = fnp.sqrt(fnp.maximum(fnp.sum(STGo[1] * W, axis=1), 1e-10))
                _av = mu / _sv
                if ROWMODE == "band":
                    _Pv = flops.stats.norm.cdf(_av).astype(f32)
                    _imp = fnp.minimum(_Pv, 1.0 - _Pv)
                else:
                    _imp = _sv * fnp.abs(_av) * flops.stats.norm.pdf(_av).astype(f32)
                mr = min(n, -(-(n - int(ROWF * n)) // 32) * 32)
                _ordr = fnp.argsort(-_imp)
                rs_idx = _ordr[:mr]
                rs_inv = fnp.argsort(_ordr)
                Ws = fnp.take(W, rs_idx, axis=0)               # (mr, n) kept output rows
                del _sv, _av, _imp
            if (A_st is not None or newborn is not None) and not skip_src:
                WD = W * fnp.reshape(w1_prev, (1, -1))''')
rep('''                TW = (LeftTree(W[None], SLEV, n, "TV") if riders
                      else LeftTree(WDb, SLEV, n, "TW"))''',
    '''                TW = ((TW if TW is not None else LeftTree(W[None], SLEV, n, "TV")) if riders
                      else LeftTree(WDb, SLEV, n, "TW"))''')
rep('''                    TWc = LeftTree(Wc[None], SLEV, n, "TL", 24)
''', '''                    TWc = LeftTree(Wc[None], SLEV, n, "TL", 24)
                    if rsel:
                        Wcs = fnp.take(Wc, rs_idx, axis=0)                  # (mr, nl)
                        WDcs = Wcs * fnp.reshape(w1l, (1, -1))
                        TWcs = LeftTree(Wcs[None], SLEV, n, "TLs", 16)
                if rsel:
                    WDs = Ws * fnp.reshape(w1_prev, (1, -1))
                    TWs = LeftTree(Ws[None], SLEV, n, "TVs", 16) if al_prev is None else None
''')
# ---- tier-1 basis transport ----
rep('''                    Qc = (WDc @ fnp.take(Qc, al_prev, axis=0)) if al_prev is not None else WD @ Qc''',
    '''                    if rsel:
                        Qc = (WDcs @ fnp.take(Qc, al_prev, axis=0)) if al_prev is not None else WDs @ Qc
                    else:
                        Qc = (WDc @ fnp.take(Qc, al_prev, axis=0)) if al_prev is not None else WD @ Qc''')
rep('''                        _mmc2(TQ, FAo[:, :, :w1t], legs["A1"][kb:ka, :, :w1t],
                              FPo[:, :, :w1t], legs["P1"][kb:ka, :, :w1t], RLEV, T1B, RMB)
                        zb = _zb(ka - kb, n - w1t)
                        fnp.copyto(legs["A1"][kb:ka, :, w1t:], zb)
                        fnp.copyto(legs["P1"][kb:ka, :, w1t:], zb)
                    else:
                        _mmc2(TQ, FAo, legs["A1"][kb:ka], FPo, legs["P1"][kb:ka], RLEV, T1B, RMB)''',
    '''                        _mmc2(TQ, FAo[:, :, :w1t], legs["A1"][kb:ka, :mr, :w1t],
                              FPo[:, :, :w1t], legs["P1"][kb:ka, :mr, :w1t], RLEV, T1B, RMB)
                        zb = _zb(ka - kb, n - w1t)[:, :mr]
                        fnp.copyto(legs["A1"][kb:ka, :mr, w1t:], zb)
                        fnp.copyto(legs["P1"][kb:ka, :mr, w1t:], zb)
                    else:
                        _mmc2(TQ, FAo, legs["A1"][kb:ka, :mr], FPo, legs["P1"][kb:ka, :mr], RLEV, T1B, RMB)''')
rep('''                    _mmc2(TU, FA2, legs["A1"][:kb], FP2, legs["P1"][:kb], RLEV, 8, T2MB)''',
    '''                    _mmc2(TU, FA2, legs["A1"][:kb, :mr], FP2, legs["P1"][:kb, :mr], RLEV, 8, T2MB)''')
# ---- young legs ----
rep('''                        _mmc(TWc, GA, SLEV, legs["A1"][c0:c1, :, :wc])
                        fnp.take(P_st[c0:c1, :, :wc], al_prev, axis=1, out=GA)
                        _mmc(TWc, GA, SLEV, legs["P1"][c0:c1, :, :wc])
                        if wc < n:
                            zb = _zb(c1 - c0, n - wc)
                            fnp.copyto(legs["A1"][c0:c1, :, wc:], zb)
                            fnp.copyto(legs["P1"][c0:c1, :, wc:], zb)''',
    '''                        TWy = TWcs if rsel else TWc
                        _mmc(TWy, GA, SLEV, legs["A1"][c0:c1, :mr, :wc])
                        fnp.take(P_st[c0:c1, :, :wc], al_prev, axis=1, out=GA)
                        _mmc(TWy, GA, SLEV, legs["P1"][c0:c1, :mr, :wc])
                        if wc < n:
                            zb = _zb(c1 - c0, n - wc)[:, :mr]
                            fnp.copyto(legs["A1"][c0:c1, :mr, wc:], zb)
                            fnp.copyto(legs["P1"][c0:c1, :mr, wc:], zb)''')
rep('''                    _mmc(TW, A_st[ka:, :, :wt], SLEV, legs["A1"][ka:k, :, :wt])
                    _mmc(TW, P_st[ka:, :, :wt], SLEV, legs["P1"][ka:k, :, :wt])
                    zb = _zb(k - ka, n - wt)
                    fnp.copyto(legs["A1"][ka:k, :, wt:], zb)
                    fnp.copyto(legs["P1"][ka:k, :, wt:], zb)
                elif ka < k:
                    _mmc(TW, A_st[ka:], SLEV, legs["A1"][ka:k])
                    _mmc(TW, P_st[ka:], SLEV, legs["P1"][ka:k])''',
    '''                    TWy = TWs if rsel else TW
                    _mmc(TWy, A_st[ka:, :, :wt], SLEV, legs["A1"][ka:k, :mr, :wt])
                    _mmc(TWy, P_st[ka:, :, :wt], SLEV, legs["P1"][ka:k, :mr, :wt])
                    zb = _zb(k - ka, n - wt)[:, :mr]
                    fnp.copyto(legs["A1"][ka:k, :mr, wt:], zb)
                    fnp.copyto(legs["P1"][ka:k, :mr, wt:], zb)
                elif ka < k:
                    TWy = TWs if rsel else TW
                    _mmc(TWy, A_st[ka:], SLEV, legs["A1"][ka:k, :mr])
                    _mmc(TWy, P_st[ka:], SLEV, legs["P1"][ka:k, :mr])''')
# ---- thin legs ----
rep('''                    WDcb = fnp.reshape(WDc, (1, n, nl))''',
    '''                    WDcb = fnp.reshape(WDcs, (1, mr, nl)) if rsel else fnp.reshape(WDc, (1, n, nl))''')
rep('''                    Z_st = fnp.matmul(WDb, Z_st)
                    if Zf_st is not None:
                        Zf_st = fnp.matmul(WDb, Zf_st)''',
    '''                    WDb_ = fnp.reshape(WDs, (1, mr, n)) if rsel else WDb
                    Z_st = fnp.matmul(WDb_, Z_st)
                    if Zf_st is not None:
                        Zf_st = fnp.matmul(WDb_, Zf_st)''')
# ---- newborn ----
rep('''                    _mmc(TWc, GB, SLEV, legs["A0"][k:k + 1, :, :mbn])
                    if mbn < n:
                        fnp.copyto(legs["A0"][k:k + 1, :, mbn:], _zb(1, n - mbn))
                    _mmc(TW, STG[1:2], SLEV, STGo[1:2])
                elif riders:
                    _mmc(TW, STG, SLEV, STGo)                 # [W a_b, W C] in one call
                    fnp.copyto(legs["A0"][k], STGo[0])''',
    '''                    if rsel:
                        # c78: the full newborn leg (k34 Stein features), its kept rows into the stack slot
                        NB = _stage("nb", (1, n, n), f32)
                        _mmc(TWc, GB, SLEV, NB[:, :, :mbn])
                        if mbn < n:
                            fnp.copyto(NB[:, :, mbn:], _zb(1, n - mbn))
                        fnp.take(NB[0], rs_idx, axis=0, out=legs["A0"][k, :mr])
                    else:
                        _mmc(TWc, GB, SLEV, legs["A0"][k:k + 1, :, :mbn])
                        if mbn < n:
                            fnp.copyto(legs["A0"][k:k + 1, :, mbn:], _zb(1, n - mbn))
                        _mmc(TW, STG[1:2], SLEV, STGo[1:2])
                elif riders and rsel:
                    _mmc(TW, STG[0:1], SLEV, STGo[0:1])         # W C was formed above
                    NB = STGo[0:1]
                    fnp.take(STGo[0], rs_idx, axis=0, out=legs["A0"][k, :mr])
                elif riders:
                    _mmc(TW, STG, SLEV, STGo)                 # [W a_b, W C] in one call
                    fnp.copyto(legs["A0"][k], STGo[0])''')
rep('''                if ckeep is not None and perm_list[k] is not None:
                    fnp.take(W, perm_list[k], axis=1, out=legs["P0"][k])     # cn: live-first columns
                    fnp.multiply(legs["P0"][k], fnp.reshape(ckeep, (1, -1)), out=legs["P0"][k])
                elif ckeep is not None:
                    fnp.multiply(W, fnp.reshape(ckeep, (1, -1)), out=legs["P0"][k])
                else:
                    fnp.copyto(legs["P0"][k], W)''',
    '''                Wr = Ws if rsel else W
                if ckeep is not None and perm_list[k] is not None:
                    fnp.take(Wr, perm_list[k], axis=1, out=legs["P0"][k, :mr])     # cn: live-first columns
                    fnp.multiply(legs["P0"][k, :mr], fnp.reshape(ckeep, (1, -1)), out=legs["P0"][k, :mr])
                elif ckeep is not None:
                    fnp.multiply(Wr, fnp.reshape(ckeep, (1, -1)), out=legs["P0"][k, :mr])
                else:
                    fnp.copyto(legs["P0"][k, :mr], Wr)''')
rep('''                z_n = fnp.reshape(W @ Rr, (1, n, Rr.shape[1]))''',
    '''                z_n = fnp.reshape(Wr @ Rr, (1, mr, Rr.shape[1]))''')
rep('''                    zf_n = fnp.reshape(W @ Ff, (1, n, Ff.shape[1]))''',
    '''                    zf_n = fnp.reshape(Wr @ Ff, (1, mr, Ff.shape[1]))''')
# ---- D3 slices on the kept rows, scattered back ----
rep('''                D3, D21 = self._dslices(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,''',
    '''                D3, D21 = self._dslices(A_st[:, :mr] if rsel else A_st, P_st[:, :mr] if rsel else P_st,
                                        Z_st, L_st, w2b_list, s_list, e_list,''')
rep('''                if regen:
                    # F68: exact transported diagonal of the regenerated core''',
    '''                if rsel:
                    D3 = fnp.take(fnp.concatenate([D3, fnp.zeros(n - mr, dtype=f32)]), rs_inv)
                if regen:
                    # F68: exact transported diagonal of the regenerated core''')
rep('''                        star = (self._star(A_st, P_st, g3_list, cv_list, bufs, n)
                                if self._k34_star else None)''',
    '''                        star = (self._star(A_st[:, :mr] if rsel else A_st, P_st[:, :mr] if rsel else P_st,
                                           g3_list, cv_list, bufs, n)
                                if self._k34_star else None)
                        if rsel and star is not None:
                            star = fnp.take(fnp.concatenate([star, fnp.zeros(n - mr, dtype=f32)]), rs_inv)''')
rep('''                                                   A_st[A_st.shape[0] - 1], w1_prev, k4_prev, star,''',
    '''                                                   NB[0] if rsel else A_st[A_st.shape[0] - 1], w1_prev, k4_prev, star,''')
rep('''                    Qc1 = fnp.reshape(Qc, (1, n, Qc.shape[1]))''',
    '''                    Qc1 = fnp.reshape(Qc, (1, Qc.shape[0], Qc.shape[1]))   # c78: mr rows at the restricted last layer''')
# ---- row-count-aware scratch in _star and _dslices ----
rep('''        V = bufs["t"][:ko]
        U = bufs["u"][:ko]''',
    '''        m = A_st.shape[1]
        V = bufs["t"][:ko] if m == n else bufs["t"][:ko, :m]
        U = bufs["u"][:ko] if m == n else bufs["u"][:ko, :m]''')
rep('''        k = len(w2b_list)
        W2B = fnp.reshape(fnp.stack(w2b_list, axis=0), (k, 1, n))''',
    '''        k = len(w2b_list)
        m = A_st.shape[1]   # c78: kept rows of the final layer (= n elsewhere)
        _b = (lambda nm: bufs[nm][:k]) if m == n else (lambda nm: bufs[nm][:k, :m])
        W2B = fnp.reshape(fnp.stack(w2b_list, axis=0), (k, 1, n))''')
for nm in ("ap", "pp", "t", "mp", "la", "lp", "xt", "yt", "u"):
    old = 'bufs["%s"][:k]' % nm
    c = s.count(old)
    assert c >= 1, nm
    s = s.replace(old, '_b("%s")' % nm)
open(p, "w").write(s)
print("patched")
