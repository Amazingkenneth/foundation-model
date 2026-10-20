import sys
p = sys.argv[1]; s = open(p).read()
def rep(old, new, cnt=1):
    global s
    c = s.count(old); assert c == cnt, (c, old[:80]); s = s.replace(old, new)
rep('XEXTP = _os.environ.get("XEXT_PARTS", "c34")      # c80 debug: which parts of the extraction move (c, 3, 4)\n',
    'XEXTP = _os.environ.get("XEXT_PARTS", "c34")      # c80 debug: which parts of the extraction move (c, 3, 4)\n'
    'XK211 = XG and _os.environ.get("XK211", "1") == "1"   # c80: capped (2,1,1) production + newborn hub contraction in G\n'
    'if XK211:\n'
    '    from t211 import TERM_TABLE_211_A  # noqa: E402\n')
# helper for capped sums (module level, before class Estimator)
rep("class Estimator(BaseEstimator):\n", '''_AX = "abc"


def _cap_ein(ops, wks):
    """sum over a (a != b, a != c) of prod(ops) * prod(axis wick vectors) -> (b, c) (lean_k3_aug.eval_capped)."""
    def ein(sub):
        args, subs, seen = [], [], set()
        for t, idx in ops:
            new = "".join(sub.get(ch, ch) for ch in idx)
            if len(new) == 2 and new[0] == new[1]:
                t = fnp.diag(t) * 1.0
                new = new[0]
            if id(t) in seen:
                t = fnp.copy(t)          # same-object repeated operands crash flopscope (F45)
            seen.add(id(t))
            args.append(t)
            subs.append(new)
        for ax, w in wks.items():
            if id(w) in seen:
                w = fnp.copy(w)
            seen.add(id(w))
            args.append(w)
            subs.append(sub.get(ax, ax))
        return fnp.einsum(",".join(subs) + "->bc", *args)
    return ein({}) - ein({"a": "b"}) - ein({"a": "c"})


def _fetch211(vsum, vec, arrs):
    nzax = [i for i, x in enumerate(vec) if x > 0]
    pat = tuple(sorted((x for x in vec if x > 0), reverse=True))
    if vsum == 2:
        if pat == (2,):
            return arrs["var"], _AX[nzax[0]]
        return arrs["C_off"], _AX[nzax[0]] + _AX[nzax[1]]
    if vsum == 3:
        if arrs["D3"] is None or pat == (1, 1, 1):
            return None
        if pat == (3,):
            return arrs["D3"], _AX[nzax[0]]
        return arrs["D21"], _AX[vec.index(2)] + _AX[vec.index(1)]
    if vsum == 4:
        if arrs["g4row"] is None:
            return None
        if pat == (4,):
            return arrs["g4row"], _AX[nzax[0]]
        if pat == (2, 2):
            return arrs["wk4m"], _AX[nzax[0]] + _AX[nzax[1]]
        if pat == (3, 1):
            return arrs["wk431"], _AX[vec.index(3)] + _AX[vec.index(1)]
        ones_ax = [i for i, x in enumerate(vec) if x == 1]
        return arrs["wk4_211"], _AX[ones_ax[0]] + _AX[ones_ax[1]]
    raise AssertionError((vsum, vec))


class Estimator(BaseEstimator):
''')
# K211 computation after the extraction block
rep("""                ext = (eps_, K1v, vb_, Cbo_)
""", """                ext = (eps_, K1v, vb_, Cbo_)
            K211add = None
            if XK211 and riders:
                # c80: capped (2,1,1) pK production from the pre-activation slices (kappa3 (1,1,1) terms skipped)
                arrs_ = dict(var=var, C_off=C_off, D3=D3 if mode == 1 else None, D21=D21 if mode == 1 else None,
                             g4row=g4row if mode == 1 else None, wk4m=wk4m, wk431=wk431,
                             wk4_211=((G_po + fnp.diagflat(dG)) * (METRIC_C / 6.0)) if (mode == 1 and G_po is not None) else None)
                acc_ = None
                for _ip, facs_, cf_, wkp_ in TERM_TABLE_211_A:
                    ops_ = [_fetch211(vs_, vc_, arrs_) for vs_, vc_ in facs_]
                    if any(o_ is None for o_ in ops_):
                        continue
                    wks_ = {_AX[ax_]: W_all[:, WICK_PAIRS.index(kp_)] for ax_, kp_ in enumerate(wkp_)}
                    t_ = _cap_ein(ops_, wks_) * cf_
                    acc_ = t_ if acc_ is None else acc_ + t_
                pk211_ = _zero_diag((acc_ + acc_.T) * 0.5)
                K211c_ = pk211_ - _cap_ein([(pk11, "ac"), (pk11, "ab")], {}) * 2.0
                K211c_ = _zero_diag((K211c_ + K211c_.T) * 0.5)
                # newborn hub contraction pK(1,1,1) -> K211 (lean k211_from_sources, birth source, P = I), u = pK(1,)
                u_ = pk1v
                ur_ = fnp.reshape(u_, (1, -1)); uc_ = fnp.reshape(u_, (-1, 1))
                if rfb > 0 and mode == 1:
                    X1h_, Y1h_ = X1_b, Y1_b
                else:
                    X1h_, Y1h_ = a_b * 3.0, a_b * fnp.reshape(w2, (1, -1))
                cX_ = X1h_.T @ u_
                cY_ = Y1h_.T @ u_
                hb_ = fnp.reshape(cX_, (-1, 1)) * Y1h_.T + (Y1h_ * ur_) @ X1h_.T + X1h_ * fnp.reshape(cY_, (1, -1))
                if regen and mode == 1 and not NO_FEED:
                    X3h_ = w1col * G_po * fnp.reshape(w1, (1, -1)) + fnp.diagflat(dgw)
                    cX3_ = X3h_.T @ u_
                    hb_ = (hb_ + fnp.reshape(cX3_, (-1, 1)) * fnp.reshape(y_b, (1, -1))
                           + fnp.reshape(y_b, (-1, 1)) * fnp.reshape(X3h_ @ u_, (1, -1))
                           + X3h_ * fnp.sum(y_b * u_))
                    # extra triple M_t1 = u_b w1sq^T
                    Mt_ = fnp.reshape(u_b, (-1, 1)) * fnp.reshape(w1sq, (1, -1))
                    hb_ = hb_ + fnp.diagflat(Mt_.T @ u_) + uc_ * Mt_.T + Mt_ * ur_
                rep3_ = (D3_w if D3_w is not None else 0.0) + (D3_new if D3_new is not None else 0.0)
                Ar_ = 3.0 * rep21.T
                if D3_w is not None or D3_new is not None:
                    Ar_ = Ar_ + fnp.diagflat(rep3_)
                hb_ = hb_ - (fnp.diagflat(Ar_.T @ u_) + uc_ * Ar_.T + Ar_ * ur_)
                birth_ = (hb_ + hb_.T) * (0.5 / 3.0)
                K211add = K211c_ * float(st["cA"]) - birth_ * (2.0 * float(st["cA"]))
""")
rep("""                    G_post = (K31 + K31.T) * (0.5 * float(st["cS"])) + fnp.diagflat(g_prev)
""", """                    G_post = (K31 + K31.T) * (0.5 * float(st["cS"])) + fnp.diagflat(g_prev)
                    if K211add is not None:
                        G_post = G_post + K211add
""")
open(p, "w").write(s); print("ok")
