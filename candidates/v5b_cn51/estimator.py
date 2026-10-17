"""Option B / outcome (c): 504aldo's V25 factorized K=3 chain (MIT, see LICENSE_504aldo)
with a learned per-neuron kappa3 + kappa4 step (learned_relu_step/optb/k34.py, features5).

At every layer >= 2 a small MLP (11 -> 64 -> 64 -> 2, GELU) maps O(n^2) features of the
chain's own state -- alpha, D3/s^3, the regenerated kappa4/s^4, own-kurtosis transport, the
off-diagonal variance share, depth, and five Stein-type terms built from the newborn A leg
(v_ai ~ Cov(z'_a, z_i)) -- to corrections of the standardized kappa3 and kappa4 that the
Wick step then uses. PK2 runs in <= 18-plane chunks (remote 100 MiB array cap).
The young-source transport, the newborn leg and the dense D21 hubs use 3-level batched
Strassen (strassen.py), chunked so every array stays under 100 MiB.
Original V25 docstring follows.

K3-simple factored cumulant propagation + MEMORYLESS KAPPA4 REGENERATION (V17)
+ D21 FEEDBACK THIN LEGS (V18, 2026-09-02, F69)
+ FINAL-LAYER TRIM (V19, 2026-09-03)
+ RESIDUAL-TIME FUSION (V20, 2026-09-03)
+ AGE-GATED SHARED BASIS FOR OLD SOURCES (V21, 2026-09-03, F66 option C / F72)
+ PRE-ACTIVATION LAMBDA TABLE (V22, 2026-09-03, F68 prefit)
+ NESTED SECOND TIER FOR THE OLDEST SOURCES (V24, 2026-09-03, F72 tiers)
+ ADAPTIVE PER-MLP LAMBDA (V25, 2026-09-03).

V25: the regenerated kappa4 off-diagonal coefficient is no longer a frozen per-layer
table but lam_l = LAM[l] * ((mean(dG)/mean(var)) / REF_R[l])^BETA, evaluated online at the
layer from the transported kappa4 diagonal dG and the pre-activation variance (both already
computed there). Fitted offline on public dumps 0-7: the per-MLP LS lambdas track this
ratio with log-log slope ~1 at every layer and the unexplained spread drops from 3-10% to
1-2%. Kill-switch: V25_BETA=0 -> V24 values.

V24: sources older than AGE_OLD2 transports are re-confined to a rank-R_OLD2 sub-basis U
(r1 x r2) INSIDE the shared tier-1 basis Qc:  A_s = Qc U FA2_s.  Legs are formed from
QU = Qc U (r2/n units per leg), the dslice contraction of tier-2 sources is lifted by U^T
into the tier-1 inner product (one trailing Qc^T per layer, unchanged), U rotates with the
tier-1 factors at each rebuild, and the oldest tier-1 member moves into tier 2 in factor
space right after each rebuild (range finder on the r1 x r1 core, no n-space work). Lean
reference (nested tiers 4:384 + 7:256, dumps 0/1): 2.155e-8 vs 2.164e-8 = free.
Kill-switch: V24_AGE_OLD2=0 -> V22 op stream.

V21: sources older than AGE_OLD transports are confined to one shared rank-R_OLD basis
Q (n x r): A_s = Q FA_s, P_s = Q FP_s with static factors. Per old source-layer the
dense legs are formed from the factors (2 n^2 r) and the two dslice contractions go
through the factors (2 n^2 r) instead of 4 dense n^3 units. One source joins the old
group per layer; at a join the basis is rebuilt by a one-pass randomized range finder on
the weighted leg Gram of old + joiner (sketch = a slice of the layer weight), the old
factors are rotated into the new basis and the Gram core S (r x r) is updated. The lean
reference (g_regen_probe.py "...+agerank:3:384") measures +8.4% raw on dumps 0/1 for a
~25% cost cut. Kill-switch: V21_NO_CONFINE=1 -> exact V20 op stream.

V20: exact, bit-identical values; only the client-side op stream changes. The nonlin
term products are written with out= into one persistent (T, n, n) buffer instead of
~30 fresh (n,n) multiplies + a ~50-matrix fnp.stack per layer (measured 4 ms + 3 ms of
residual per layer), and the range-finder sketches are one contiguous copy of the
weight slice per layer instead of strided views (strided qr/matmul input costs ~0.8 ms
each). Kill-switch: none needed (values identical); V19 stays the reference.

V19: exact cost cut, zero accuracy change. The final layer emits only the mean, whose
(1,) nonlin terms need var = diag(C_pre), D3 and the K4 diagonal -- never C_off, the
(2,1) dslice D21, nor the (1,1)/(2,1)/(2,2) term program. V18 computed all of them at
the last layer and discarded them: the two dense dslice contractions (2k n^3 units,
k = 15 live sources) plus the full C_pre sandwich (2 units, only its diagonal is used).
V19 transports only diag(C_pre) = colsum(w32 * (C @ w32)) (1 unit) and runs the source
machinery in D3-only mode at the last layer (transports + n^2 contractions, no D21).
Expected C: 0.509xB -> ~0.48xB. Kill-switch: V19_FULL_LAST=1 -> exact V18 op stream.
Below this line the V18 docstring follows unchanged.


V18: the V1.6 "K3-dslice feedback into the new-block factors" (dropped since V1.6 as
DS-shadowed) is worth -10% raw on the regen base (F69) and is rank <= 16 in D21.
Birth factors of the B1 hub pair become
  X1 = 3 a_b + Xt,   Xt = 1.5 d(w2) D21            ~ F1 R1,  F1 = d(w2) Q,   R1 = 1.5 Bm
  Y1 = a_b d(w2) + Yt, Yt = 0.5 d(w1) D21^T d(w3)  ~ F2 R2,  F2 = d(w1) Bm^T, R2 = 0.5 Q^T d(w3)
with D21 ~ Q Bm from a rank-R_FB randomized range finder (same recipe as Rres). The
identity-born thin legs F1/F2 travel in their own transported stack Zf (like Z, but never
entering the M-leg einsums); in the dslice contraction the new terms either add to the fused left factors
(LA += Xt*P*w2/3 + P*Yt; LP += A*Yt + Xt*A*w2/3 + Xt*Yt/3; D3 += rowsum(P*3*LPadd)) or
contract against the thin right factors ((AP + Xt*P/3) R2^T F2^T and
(AP*w2 + P*Yt)/3 R1^T F1^T). Extra n^3: ZERO; extra ~16 n^2 r per source-layer.
Lean reference: g_regen_probe.py "regen:c+no31+feedc+usec+fitfull+rres:16+sb2+nowk22
+d21r:16": 2.195e-8 / 2.091e-8 on P2 dumps 0/1 (V17 port: 2.51 / 2.25e-8).
Kill-switch: V18_NO_FB=1 -> exact V17 behaviour (zero thin legs).
Below this line the V17 docstring follows unchanged.


V17 (2026-09-02, F68): the augmented-K3 kappa4 channel (F51/F64) without any
per-source kappa4 memory. The kappa4 matrix core G of the aug chain is, to
R^2 ~0.9-0.97, diag(g) + lambda_l * C_off with ONE frozen scalar per layer
(LAM table, fitted offline on public MLPs). Everything the channel needs then
rides on objects V16b already carries:
  - transported diagonal dG_pre = (W*W) g + lambda (diag(C_pre) - (W*W) var)  [n^2]
  - use side: wk4_row = m dG, wk4_mat = (m/6)(dG_i + dG_j) (== V16b's vec+ww form),
    plus the (3,1) slice wk431 = (m/2) lambda C_pre_off feeding 4 extra n^2 terms
  - K4->K3 feed: hub pair X3 = d(w1) G_pre d(w1) = lambda A d(w1) + P d(w1^2 dG)
    (column scalings of the A/P legs), Y3 = (m/4) w2 1^T (rank-1, folds into the
    P-group + one rank-1 outer product), M_t1 = (m/4)(w2*dG)(w1^2)^T (rank-1 M-type
    -> one extra thin Z/L column).  Extra n^3 cost: ZERO.
Lean reference: scratch/lean_k3_aug.py regen chain (g_regen_probe.py
"regen:c+no31+feedc+usec"): 2.196e-8 / 2.161e-8 on P2 dumps 0/1 vs 4.1e-8 for V16b.
Below this line the V16b docstring follows unchanged.


V16b (2026-09-01, F63): exact cost restructuring of the source machinery.
  - The M leg (B3 = Sym(M x P x P)) is never transported: M_b = diag(S3c)
    + 3*S21^T and S21 = S_sep + Rres with S_sep = diag(e) C_off diag(w1) the
    exact leading (2,1) Wick term (a column-scaled copy of a_b) and Rres a
    D21-driven residual that is numerically low-rank (rank-64 keeps final MSE
    at parity, F63). Hence M = P diag(s) + 3 A diag(e) + 3 Z L^T with Z = P Rr
    a thin (n,r) transported leg and L static.
  - All dslice contractions collapse into TWO fused einsums per layer (right
    factors A and P) plus O(n^2 r) thin terms: per source-layer 2 dense
    transports + 2 dense contractions (was 3 + 4).
  - Rres range finder: randomized (Omega = a slice of the layer weight), one
    power iteration, flopscope-billed QR; ~4 n^2 r FLOPs per birth.

Port of the reference kprop k_max=3 SIMPLE factored algorithm (companion paper
arXiv:2605.05179) with cost engineering measured in F42-F45, plus two
accuracy riders measured in F47/F49 (combined 1.44x lower final MSE):
  - vec+ww (augmented-lite v0): the radial K4 channel keeps a per-neuron
    diagonal-content vector instead of a single scalar; born from K(4,) and
    the K(2,2) column sums, transported one layer by the true (W*W) row
    action in place of one average-metric cup. ~n^2/layer, cost-neutral.
  - online mean correction: per-layer ridge-fitted delta = feats @ beta
    (13 per-neuron features already computed in the loop; beta fitted
    offline on the public teacher-forced trajectory, embedded below).
    Applied inside the loop so it propagates.
Base cost engineering:
  - shared-P hub blocks per source layer: B1 = Sym(X1 x P x Y1) with X1 = 3*abar,
    B3 = Sym(M x P x P); identity-born legs all evolve by the same map
    leg <- W @ (w1 * leg), so one P per source serves every sub-block, and the
    wick scaling is folded into the next layer's weight (WD).
  - V1.6 ablations (each measured to cost <~1% final MSE): the B2 hub sub-block
    and the K3-dslice feedback into new-block factors are DS-shadowed at birth
    (their diagonal-slice content migrates into the exact M chain automatically);
    the WK22 struct term is dropped from Y1; nonlin terms pruned at relative
    contribution >= 1e-5 (62 of 90 kept).
  - residual-time engineering: per-source matrices held as batched (k,n,n)
    stacks; all wick vectors built at once in the unified form
    w(k,p) = const*sigma^e*(P1(a)phi + P2(a)Phi); nonlin terms evaluated as one
    stacked einsum per arity; dslice hub contractions fused into 3/4-operand
    einsums with no (k,n,n) intermediates; final layer computes only the mean.

Cost: ~1.83e12 flopscope FLOPs per MLP = 0.833x the 2^41 budget (deterministic,
data-independent). Local mini-dump accuracy: mse_final ~3.2e-8 test-set mean
(V1.6 base alone: ~6.2e-8; exact K3-simple reference: 4.8e-8 at an over-budget
2.24x; cov-prop baseline: 4.4e-6).
Prediction: per-layer post-ReLU means, shape (depth, width). No randomness.
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from pathlib import Path
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from strassen2 import LeftTree, mm as _smm  # noqa: E402
MP_LI_MIN = 8  # optg: message-passing hooks only at layers li >= MP_LI_MIN

# optc Strassen kernel: square n x n families at SLEV levels in batch chunks of SB (largest
# array 86 MiB at 4 levels); n x 384 basis families at RLEV levels with leaf blocks >= RMB.
RFHUB = 1  # range-finder hub on Strassen too
T2MB = 20  # tier-2 rank 160, 3 Strassen levels
SLEV, SB, RLEV, RMB = 5, 2, 4, 20   # rank-320 families keep 4 levels
SBK = SB  # cn5: batch chunk (SB is shadowed by a local of _predict_core)
# tier-1 (4-level n x r) families: batch <= 6 keeps the depth-2 Strassen scratch (343, B, n/8, w/8)
# under the grader's ~134.9 MB per-array cap (B = 6: 134,873,088 bytes; B = 8: 179.8 MB fails)
T1B = 6


def _sd(A, Bm, lev=3, mb=48):
    """Dense 2-D A @ Bm on the Strassen kernel (the n x 384 Gram / basis products)."""
    out = fnp.empty((1, A.shape[0], Bm.shape[1]), dtype=fnp.float32)
    _smm(A[None], Bm[None], lev, out, mb=mb)
    return out[0]


def _mmc(X, R, lev, out, bmax=SB, mb=None):
    """out (B, m, w) = X @ R (X: LeftTree or (1 | B, m, k) array), in batch chunks of bmax."""
    B = R.shape[0]
    if B <= bmax:
        return _smm(X, R, lev, out, mb=mb)
    for c0 in range(0, B, bmax):
        c1 = min(B, c0 + bmax)
        Xc = X if (isinstance(X, LeftTree) or X.shape[0] == 1) else X[c0:c1]
        _smm(Xc, R[c0:c1], lev, out[c0:c1], mb=mb)
    return out


def _hub(X, Y, lev, bmax=SB, mb=None):
    """sum_k X_k @ Y_k^T for X (K, m, j), Y (K, c, j) -> fresh (m, c)."""
    K = X.shape[0]
    acc = fnp.empty((1, X.shape[1], Y.shape[1]), dtype=X.dtype)
    tmp = None
    for c0 in range(0, K, bmax):
        c1 = min(K, c0 + bmax)
        if c0 == 0:
            _smm(X[c0:c1], fnp.swapaxes(Y[c0:c1], -1, -2), lev, acc, ksum=True, mb=mb)
        else:
            if tmp is None:
                tmp = fnp.empty((1, X.shape[1], Y.shape[1]), dtype=X.dtype)
            _smm(X[c0:c1], fnp.swapaxes(Y[c0:c1], -1, -2), lev, tmp, ksum=True, mb=mb)
            fnp.add(acc, tmp, out=acc)
    return fnp.reshape(acc, (X.shape[1], Y.shape[1]))


_STG = {}


_STGF = {}


def _stage(tag, shape, dtype):
    """Staging scratch of a tag; every shape of one tag views one flat buffer (cn: the narrowed
    hub stages a different column count per layer)."""
    key = (tag, tuple(int(x) for x in shape))
    b = _STG.get(key)
    if b is None:
        size = 1
        for x in key[1]:
            size *= x
        fk = (tag, str(dtype))
        fl = _STGF.get(fk)
        if fl is None or fl.shape[0] < size:
            fl = fnp.empty((size,), dtype=dtype)
            _STGF[fk] = fl
            for kk in [kk for kk in _STG if kk[0] == tag]:
                del _STG[kk]
        b = fnp.reshape(fl[:size], key[1])
        _STG[key] = b
    return b


def _hub2(X1, Y1, X2, Y2, lev, bmax=SB, mb=None):
    """_hub(X1, Y1) + _hub(X2, Y2) with whole chunks direct and the remainders of both stacks
    staged into shared calls (one Strassen call costs ~90 flopscope ops of residual)."""
    m1, m2 = X1.shape[0], X2.shape[0]
    q1, q2 = (m1 // bmax) * bmax, (m2 // bmax) * bmax
    if m1 + m2 <= bmax:
        q1 = q2 = 0
    acc = None
    for X, Y, q in ((X1, Y1, q1), (X2, Y2, q2)):
        if q > 0:
            part = _hub(X[:q], Y[:q], lev, bmax, mb)
            acc = part if acc is None else acc + part
    r1, r2 = m1 - q1, m2 - q2
    if r1 + r2 > 0:
        Xs = _stage("hx", (r1 + r2,) + tuple(X1.shape[1:]), X1.dtype)
        Ys = _stage("hy", (r1 + r2,) + tuple(Y1.shape[1:]), Y1.dtype)
        if r1:
            fnp.copyto(Xs[:r1], X1[q1:]); fnp.copyto(Ys[:r1], Y1[q1:])
        if r2:
            fnp.copyto(Xs[r1:], X2[q2:]); fnp.copyto(Ys[r1:], Y2[q2:])
        part = _hub(Xs, Ys, lev, bmax, mb)
        acc = part if acc is None else acc + part
    return acc


def _mmc2(T, R1, out1, R2, out2, lev, bmax, mb=None):
    """_mmc(T, R1, out1) and _mmc(T, R2, out2) as one staged call when both fit one chunk."""
    m1, m2 = R1.shape[0], R2.shape[0]
    if m1 + m2 > bmax:
        _mmc(T, R1, lev, out1, bmax, mb)
        _mmc(T, R2, lev, out2, bmax, mb)
        return
    Rs = _stage("mr", (m1 + m2,) + tuple(R1.shape[1:]), R1.dtype)
    Os = _stage("mo", (m1 + m2,) + tuple(out1.shape[1:]), out1.dtype)
    fnp.copyto(Rs[:m1], R1); fnp.copyto(Rs[m1:], R2)
    _smm(T, Rs, lev, Os, mb=mb)
    fnp.copyto(out1, Os[:m1]); fnp.copyto(out2, Os[m1:])


def _cpre_sym(Y, w32, n, lev=3):
    """C_pre = Y w32 (Y = W C, W = w32^T) from the three distinct blocks of its 2x2 partition
    (11, 12, 22) as one batch-3 Strassen call; block 21 = 12^T. Returns a fresh (n, n)."""
    h = n // 2
    X3 = fnp.empty((3, h, n), dtype=Y.dtype)
    R3 = fnp.empty((3, n, h), dtype=Y.dtype)
    O3 = fnp.empty((3, h, h), dtype=Y.dtype)
    Y1, Y2 = Y[:h], Y[h:]
    W1, W2 = w32[:, :h], w32[:, h:]
    fnp.copyto(X3[0], Y1); fnp.copyto(X3[1], Y1); fnp.copyto(X3[2], Y2)
    fnp.copyto(R3[0], W1); fnp.copyto(R3[1], W2); fnp.copyto(R3[2], W2)
    _smm(X3, R3, lev, O3, mb=16)
    Cp = fnp.empty((n, n), dtype=Y.dtype)
    O12 = O3[1]
    fnp.copyto(Cp[:h, :h], O3[0]); fnp.copyto(Cp[:h, h:], O12)
    fnp.copyto(Cp[h:, :h], O12.T); fnp.copyto(Cp[h:, h:], O3[2])
    return Cp

# ---- embedded coefficient tables (generated by dump_k3_tables2.py) ----
# v2 K3-simple tables (dump_k3_tables2.py; pruned at rel>=1e-5)
WICK_PAIRS = [(0, 1),
 (0, 2),
 (0, 3),
 (0, 4),
 (1, 1),
 (1, 2),
 (2, 1),
 (2, 2),
 (3, 1),
 (3, 2),
 (3, 3),
 (3, 4),
 (4, 1),
 (4, 2),
 (4, 3),
 (4, 4),
 (5, 1),
 (5, 2),
 (6, 1),
 (6, 2),
 (7, 1)]

WICK_UNIFIED = [(1.0, 1, [1.0], [0.0, 1.0]),
 (1.0, 2, [0.0, 1.0], [1.0, 0.0, 1.0]),
 (1.0, 3, [2.0, 0.0, 1.0], [0.0, 3.0, 0.0, 1.0]),
 (1.0, 4, [0.0, 5.0, 0.0, 1.0], [3.0, 0.0, 6.0, 0.0, 1.0]),
 (1.0, 0, [], [1.0]),
 (1.0, 1, [2.0], [0.0, 2.0]),
 (1.0, -1, [1.0], []),
 (2.0, 0, [], [1.0]),
 (1.0, -2, [-0.0, -1.0], []),
 (2.0, -1, [1.0], []),
 (6.0, 0, [], [1.0]),
 (1.0, 1, [24.0], [0.0, 24.0]),
 (1.0, -3, [-1.0, 0.0, 1.0], []),
 (2.0, -2, [-0.0, -1.0], []),
 (6.0, -1, [1.0], []),
 (24.0, 0, [], [1.0]),
 (1.0, -4, [-0.0, 3.0, -0.0, -1.0], []),
 (2.0, -3, [-1.0, 0.0, 1.0], []),
 (1.0, -5, [3.0, 0.0, -6.0, 0.0, 1.0], []),
 (2.0, -4, [-0.0, 3.0, -0.0, -1.0], []),
 (1.0, -6, [-0.0, -15.0, -0.0, 10.0, -0.0, -1.0], [])]

TERM_SPECS = {(1,): [('ones1', 'ones1', 0, None, 1.0),
        ('d3', 'ones1', 8, None, 0.16666666666666666),
        ('g4', 'ones1', 12, None, 0.041666666666666664),
        ('d3', 'd3', 18, None, 0.013888888888888888),
        ('d3', 'g4', 20, None, 0.006944444444444444)],
 (1, 1): [('c_off', 'ones2', 4, 4, 1.0),
          ('d21T', 'ones2', 4, 6, 1.0),
          ('c_off', 'c_off', 6, 6, 0.5),
          ('wk4m', 'ones2', 6, 6, 0.25),
          ('d3row', 'c_off', 4, 12, 0.3333333333333333),
          ('c_off', 'd21T', 6, 8, 1.0),
          ('d3row', 'd21T', 4, 16, 0.16666666666666666),
          ('g4row', 'c_off', 4, 16, 0.08333333333333333),
          ('d3row', 'd21', 6, 12, 0.16666666666666666),
          ('d21T', 'd21T', 6, 12, 0.25),
          ('c_off', 'wk4m', 8, 8, 0.25),
          ('d21T', 'd21', 8, 8, 0.25),
          ('g4row', 'd21T', 4, 18, 0.041666666666666664),
          ('d3row', 'wk4m', 6, 16, 0.08333333333333333),
          ('g4row', 'd21', 6, 16, 0.041666666666666664),
          ('d21T', 'wk4m', 8, 12, 0.25),
          ('g4row', 'wk4m', 6, 18, 0.020833333333333332)],
 (2,): [('ones1', 'ones1', 1, None, 1.0),
        ('d3', 'ones1', 9, None, 0.16666666666666666),
        ('g4', 'ones1', 13, None, 0.041666666666666664)],
 (2, 1): [('c_off', 'ones2', 5, 4, 1.0),
          ('d21T', 'ones2', 5, 6, 0.5),
          ('d21', 'ones2', 7, 4, 0.5),
          ('c_off', 'c_off', 7, 6, 0.5),
          ('wk4m', 'ones2', 7, 6, 0.25),
          ('d3row', 'c_off', 5, 12, 0.16666666666666666),
          ('c_off', 'd21T', 7, 8, 0.5),
          ('c_off', 'd21', 9, 6, 0.5),
          ('c_off', 'd3col', 13, 4, 0.16666666666666666),
          ('d3row', 'd21T', 5, 16, 0.08333333333333333),
          ('g4row', 'c_off', 5, 16, 0.041666666666666664),
          ('d3row', 'd21', 7, 12, 0.08333333333333333),
          ('d21T', 'd21T', 7, 12, 0.125),
          ('c_off', 'wk4m', 9, 8, 0.25),
          ('d21T', 'd21', 9, 8, 0.25),
          ('c_off', 'g4col', 17, 4, 0.041666666666666664),
          ('d21', 'd3col', 17, 4, 0.08333333333333333),
          ('g4row', 'd21T', 5, 18, 0.020833333333333332),
          ('g4row', 'd21', 7, 16, 0.020833333333333332),
          ('d21', 'g4col', 19, 4, 0.020833333333333332)],
 (2, 2): [('c_off', 'ones2', 5, 5, 1.0),
          ('d21T', 'ones2', 5, 7, 1.0),
          ('c_off', 'c_off', 7, 7, 0.5),
          ('wk4m', 'ones2', 7, 7, 0.25),
          ('d3row', 'c_off', 5, 13, 0.3333333333333333),
          ('c_off', 'd21T', 7, 9, 1.0),
          ('d3row', 'd21T', 5, 17, 0.16666666666666666),
          ('g4row', 'c_off', 5, 17, 0.08333333333333333),
          ('d21T', 'd21T', 7, 13, 0.25),
          ('c_off', 'wk4m', 9, 9, 0.25),
          ('g4row', 'd21T', 5, 19, 0.041666666666666664)],
 (3,): [('ones1', 'ones1', 2, None, 1.0),
        ('d3', 'ones1', 10, None, 0.16666666666666666),
        ('g4', 'ones1', 14, None, 0.041666666666666664)],
 (4,): [('ones1', 'ones1', 3, None, 1.0),
        ('d3', 'ones1', 11, None, 0.16666666666666666),
        ('g4', 'ones1', 15, None, 0.041666666666666664)]}

PK2K_TABLE = {(1,): [(((1,),), 1.0)],
 (1, 1): [(((1, 1),), 1.0)],
 (2,): [(((1,), (1,)), -1.0), (((2,),), 1.0)],
 (2, 1): [(((1, 0), (1, 1)), -2.0), (((2, 1),), 1.0)],
 (2, 2): [(((0, 1), (1, 0), (1, 1)), 4.0),
          (((0, 1), (2, 1)), -2.0),
          (((1, 0), (1, 2)), -2.0),
          (((1, 1), (1, 1)), -2.0),
          (((2, 2),), 1.0)],
 (3,): [(((1,), (1,), (1,)), 2.0), (((1,), (2,)), -3.0), (((3,),), 1.0)],
 (4,): [(((1,), (1,), (1,), (1,)), -6.0),
        (((1,), (1,), (2,)), 12.0),
        (((1,), (3,)), -4.0),
        (((2,), (2,)), -3.0),
        (((4,),), 1.0)]}

# Online mean-correction coefficients, (depth, n_features) = (16, 13); ridge-fitted
# offline (lam=1e-3) on the teacher-forced trajectory of 5 public v2-phase2 MLPs
# (k3_aug_corr.py, F47/F49). Feature order:
# 1, mu_pre, var_pre, sigma, alpha, |alpha|, phi, Phi, pred, D3, D21n, K4, sigma*phi
CORR_BETA = [
    [0.0006411968414813632, 0.0, 0.00032431282848653057, -0.0003042144758593203, 0.0, 0.0, 0.0, 0.0, -0.0007625526069403323, 0.0, 0.0, 0.0, -0.0007625526055534857],
    [-0.0004483531031709554, -8.08970612775365e-08, 0.00024032689914639557, 0.0003234947612489975, 7.734051179371133e-06, -8.401877300643312e-06, 0.0026976072084027533, -2.9522256040202606e-06, -1.1604426266687203e-05, 9.457618761612473e-06, -3.3186036376400295e-05, -0.008465131129601192, -0.002371947086869632],
    [-0.00030645743691215543, 2.660655267636766e-06, 0.0006192175275623578, -0.0002468827461681828, -2.9900896046038434e-06, 6.841942903175705e-08, 0.0033128783967715193, 5.123025223409135e-06, -1.831086058750464e-06, -8.948552609048747e-06, -3.292520087183994e-05, 0.024826431355466128, -0.003290775627340336],
    [-0.00010464301263958924, 1.0735707741330386e-06, 0.0009154900192795534, -0.0006467314093895341, 3.6870994916648376e-06, 6.046342808060834e-06, 0.00341846204288833, -8.975061063301133e-06, -8.178979563111877e-06, 9.51453794000349e-05, -3.005588379050443e-05, 0.031163100958057806, -0.003816642326501922],
    [-0.0006681684602995246, -7.2789662985629386e-06, 2.591925967081249e-05, 0.0007797247634619618, 8.477664175402185e-06, 1.7025270836169033e-06, 0.003419282618548069, 7.535308544950828e-07, -5.733309150056555e-06, -2.9160257628201262e-05, -1.722583323732972e-05, 0.04076990884979579, -0.004248021529544134],
    [-0.0006734457317912116, 3.1290157938005224e-05, 0.00022889868689835653, 0.0007465462593369972, 6.851356914590185e-07, 2.1755554980020755e-05, 0.004058574189699476, -4.277653393515956e-06, -6.039384411257449e-05, -9.02015920342041e-05, -1.9819140691513135e-05, 0.041867576221848744, -0.005466361831539052],
    [-0.0006750844151648486, 6.180105939061905e-05, 0.0004389649313793868, 0.0006339652749835816, -6.694983507237225e-07, 4.690187390705702e-05, 0.003977857680309406, 4.383031188174939e-06, -0.00012887677523401884, 0.0002505563048388552, -2.5566781991513985e-05, 0.07694913516123503, -0.0057040529266279925],
    [-0.000972879718186106, 5.1156558236482596e-05, -0.0004913239875599442, 0.0017893987001835238, 1.90937415308986e-06, 3.802173926700858e-05, 0.0038578329372384434, 2.8522169318710464e-06, -0.00010738838919747425, -0.00015415811216305824, -2.5539604454967108e-05, 0.08311668913952446, -0.005918296132102073],
    [-0.0007850991486069435, 7.899688664131035e-05, -0.00022144314048615243, 0.001495955739849875, 1.8359978888302925e-07, 5.023393155879106e-05, 0.0034762600760059295, 3.215904175585605e-06, -0.00015314900768603073, -0.0004475649473906088, -3.028530737478287e-05, 0.06152086703424944, -0.00569825101925022],
    [-0.0005795122172445841, 0.00011465424961793733, 0.00039693472854898914, 0.0007970833957439235, 6.825630979361196e-06, 7.966232390898656e-05, 0.0032776762940374586, -1.387178929061937e-06, -0.0002567385150822406, 0.00028972790478315425, -2.7005665153789784e-05, 0.07590370226079919, -0.005546910408445549],
    [-0.0004949253254431177, 9.892837012135289e-05, 0.0005826341044064549, 0.0006128630824666212, 1.8258883411132417e-06, 6.024891049057027e-05, 0.003354817421260185, -6.690225809034046e-06, -0.00020554230085297872, 0.0002695214755196366, -2.5128600253275433e-05, 0.08226262500093764, -0.006092394095370619],
    [-0.0007584153570441029, 0.0001792621247156503, 0.0001931413288044634, 0.0014013628928244146, 5.054405581529621e-06, 0.00010157511263920278, 0.004187883074782575, -5.420295053799967e-06, -0.00037802160373858374, 0.00014440604770373433, -2.2764810929109396e-05, 0.07738752039037704, -0.008027695439553941],
    [-0.0008291520607940194, 0.0002115021197150293, -0.00012740149902796318, 0.0017326132691935436, -2.1624432383474912e-06, 0.0001087192480144056, 0.004533661385130805, 4.837349485145022e-07, -0.00041674041200178005, 0.00018467937018199515, -1.9126957306380207e-05, 0.07896437764069765, -0.008986219483970914],
    [-0.000658293372173689, 0.00018380103458278174, 0.00046374648579651084, 0.001216035726510531, 4.012828172272205e-06, 9.570320459811334e-05, 0.004325867834172417, -1.2894882645758776e-05, -0.00038451298928645256, 0.0005113367132840055, -2.241500653445156e-05, 0.07104629639605271, -0.00897436256704376],
    [-0.0009527557374376014, 0.00019049799075311402, -0.0009213478791293237, 0.0025553941675787975, 3.508060734949607e-07, 9.248186905581894e-05, 0.004641633979578725, -3.1682752945685096e-06, -0.00038201626600816054, 3.0155769164602614e-05, -2.3334690641924517e-05, 0.07182226701958777, -0.01001369215216409],
    [-0.0006908532933348147, 0.00018408588405074964, 0.0004249876754929348, 0.0013835702671030146, 5.170891351788878e-06, 9.144035619901057e-05, 0.0044972055980902715, -1.0115381018603137e-05, -0.00038927641644205714, 0.0002930281376816756, -2.8026123982155874e-05, 0.157561818094021, -0.010070547059124272],
]

# ---- end embedded tables ----

# Regenerated kappa4 off-diagonal coefficient per layer, fitted at the PRE-activation
# (transported G_pre_off = LAM[l] * C_pre_off after layer l's ReLU); LS fits on the dense
# aug chain, mean over public P2 dumps 0-7 (runs/lam_prefit_d0-7.log, V22). The older
# post-ReLU table (runs/g_regen_10.log) differs mainly at layer 0 (1.95e-3 -> 4.99e-3).
# Suite shape only (gated like CORR_BETA).
LAM = [4.9895e-03, 8.0876e-03, 9.8291e-03, 1.0549e-02, 1.0851e-02, 1.0828e-02, 1.0589e-02, 1.0048e-02, 9.6483e-03, 9.1720e-03, 8.7730e-03, 8.3938e-03, 8.0555e-03, 7.6770e-03, 7.2588e-03, 7.2588e-03]
METRIC_C = 2.0
# debug kill-switches (parity_v17.py): set from the environment before import
import os as _os
NO_FEED = _os.environ.get("V17_NO_FEED", "0") == "1"
NO_WK431 = _os.environ.get("V17_NO_WK431", "0") == "1"
NO_REGEN = _os.environ.get("V17_NO_REGEN", "0") == "1"
NO_FB = _os.environ.get("V18_NO_FB", "0") == "1"  # V18: D21 feedback thin legs off
FULL_LAST = _os.environ.get("V19_FULL_LAST", "0") == "1"  # V19: 1 -> V18 op stream
NO_CONFINE = _os.environ.get("V21_NO_CONFINE", "0") == "1"  # V21: 1 -> V20 op stream
QPASS = int(_os.environ.get("V21_QPASS", "1"))  # V21: subspace-iteration passes of the basis range finder
NO_SRC_LAST = _os.environ.get("V19_NO_SRC_LAST", "0") == "1"  # V19 probe: D3(last) := 0, skip source transports
# The V16b CORR_BETA mean rider was ridge-fitted on V16b's OWN trajectory and is
# anti-correlated with V16b's base error; on the regen base it HURTS (dump0: 3.57e-8
# with vs 2.51e-8 without). Off by default until refitted on the V17 trajectory.
NO_CORR = _os.environ.get("V17_NO_CORR", "1") == "1"
# V25: table scale 0.95 (8-dump scan 0.9/0.92/0.95/1.0 -> 2.2752/2.2727/2.2704/2.2782e-8)
LAM = [c * float(_os.environ.get("V17_LAM_SCALE", "0.95")) for c in LAM]
# V25: reference ratio mean(dG)/mean(var) at layer l+1 (8-dump mean, scratch/lam_obs_d0-7.npz)
# and the log-log exponent of the adaptive rule (pooled fit 1.08; 1.0 shipped).
REF_R = [6.58815e-03, 8.18414e-03, 8.53136e-03, 8.38859e-03, 8.10153e-03, 7.74287e-03,
         7.35951e-03, 6.91093e-03, 6.55752e-03, 6.17892e-03, 5.83953e-03, 5.56824e-03,
         5.32459e-03, 5.05165e-03, 4.77181e-03]
BETA = float(_os.environ.get("V25_BETA", "1.0"))
# cm: a newborn source gets no third-cumulant columns for neurons that were dead at its birth
# (Phi(alpha) < CM_TAU). Their newborn terms are Gaussian-closure Wick tails (w2 ~ phi(alpha),
# the residual skew S3c of a rarely active unit); zeroing them removes error, not content
# (11 + 22 local MLPs, paired).
CM_TAU = float(_os.environ.get("CM_TAU", "0.006"))  # cn50: columns dead at birth below Phi 0.006 (75 networks with rows 0.012: -1.46% +- 0.98% MSE, +0.67% C vs cn49b)
CN = _os.environ.get("CN", "1") == "1"
CN_MIN = 768   # narrowest column block: 5 Strassen levels with leaves >= 24
WARM = _os.environ.get("CN_WARM", "1") == "1"
FB_OS = int(_os.environ.get("CN_FBOS", "8"))  # cn49b: oversampling 8 (75 networks: -0.26% +- 0.13% vs 16, and -0.2% C)
KEEP = _os.environ.get("CN_KEEP", "alt")
# cn3: rows. After the wick the legs' rows of neurons with Phi < RW_TAU are scaled by w1 < RW_TAU,
# so the next layer's leg transports, newborn A leg, joiner and basis transport contract over
# the other neurons only (a multiple of 64 >= CN_MIN). W C keeps the full inner dimension.
RW_TAU = float(_os.environ.get("RW_TAU", "0.012"))  # cn50: rows below Phi 0.012   # cn11: rows at Phi < 0.01 (distance to the
# exact-row chain 1.0-1.2e-5 vs 0.75-0.93e-5 at 0.002, ~+0.3% MSE for -1.5% C)
# cn46: step-level fixes from the 75-network screen (docs/experiments_2026-10-17.md): the third Mehler
# pair term (1/6) a3 b3 C0^3 (Artyom_Mazur 18249, one of the two augmented pair diagrams a SIMPLE program lacks)
# and marg2 (keep the k = 2 marginal Edgeworth corrections of the pair coefficients only in the dead tail).
XC3, XMARG2 = 0.0, 1.0
DEBUG = []  # parity_v17.py: per-layer dict of diagnostics when V17_DEBUG=1

# pruned V16b table + the 4 (3,1)-slice use-side terms
_I = WICK_PAIRS.index
TERM_SPECS[(1, 1)].append(('wk431', 'ones2', _I((1, 1)), _I((3, 1)), 1.0 / 3.0))
TERM_SPECS[(2, 1)].append(('wk431', 'ones2', _I((1, 2)), _I((3, 1)), 1.0 / 6.0))
TERM_SPECS[(2, 1)].append(('wk431', 'ones2', _I((3, 2)), _I((1, 1)), 1.0 / 6.0))
TERM_SPECS[(2, 2)].append(('wk431', 'ones2', _I((1, 2)), _I((3, 2)), 1.0 / 3.0))
ALPHA_DEG = 5
SIG_EMIN, SIG_EMAX = -6, 4
D2_IPS = [(1, 1), (2, 1), (2, 2)]
D1_IPS = [(1,), (2,), (3,), (4,)]
D2_OUT = [(1, 1), (2, 1), (2, 2)]
D1_OUT = [(2,), (3,), (4,)]
MODE0_MISSING = {"d21", "d21T", "d3col", "d3row", "d3", "wk4m", "g4col", "g4row", "g4",
                 "wk431"}


def _statics(n: int) -> dict:
    c = 4 - 2 + n / 2.0 - 1.0
    P2 = (c - 2.0) / (16.0 * 2.0 * c * (1.0 - c) * (2.0 - c))
    return dict(P2=P2, k4_c4=24.0, k4_c22=24.0, wk4_c4=1.0, wk4_c22=1.0 / 3.0,
                cA=6.0 / (n + 4.0), cI=-3.0 / ((n + 2.0) * (n + 4.0)))


def _zero_diag(A):
    fnp.fill_diagonal(A, 0.0)
    return A


def _build_wick_consts():
    np_ = len(WICK_PAIRS)
    C1 = [[0.0] * np_ for _ in range(ALPHA_DEG + 1)]
    C2 = [[0.0] * np_ for _ in range(ALPHA_DEG + 1)]
    # sigma-power selection with the constant prefactor folded in
    SELC = [[0.0] * np_ for _ in range(SIG_EMAX - SIG_EMIN + 1)]
    for j, (const, e, p1, p2) in enumerate(WICK_UNIFIED):
        SELC[e - SIG_EMIN][j] = const
        for i, c in enumerate(p1):
            C1[i][j] = c
        for i, c in enumerate(p2):
            C2[i][j] = c
    return C1, C2, SELC


def _term_prog(mode: int):
    if XC3 != 0.0:
        for ip, wl, wr in (((1, 1), 8, 8), ((2, 1), 9, 8), ((2, 2), 9, 9)):
            if not any(t[0] == 'c3' for t in TERM_SPECS[ip]):
                TERM_SPECS[ip].append(('c3', 'ones2', wl, wr, XC3 / 6.0))
    """Fused-einsum program for the nonlin terms of one availability mode."""
    missing = MODE0_MISSING if mode == 0 else ({"wk431"} if mode == 2 else set())
    np_ = len(WICK_PAIRS)
    d2 = []  # (group, a, b, wl, wr, coef)
    d1 = []
    for g, ip in enumerate(D2_IPS):
        for a, b, wl, wr, coef in TERM_SPECS.get(ip, []):
            if a in missing or b in missing:
                continue
            d2.append((g, a, b, wl, wr, coef))
    for g, ip in enumerate(D1_IPS):
        for a, b, wl, wr, coef in TERM_SPECS.get(ip, []):
            if a in missing or b in missing:
                continue
            d1.append((g, a, b, wl, coef))
    # optc: product terms first; terms with a ones factor last, grouped by their matrix
    # (one einsum per matrix instead of one copy per term)
    def _ones(t):
        return t[1] in ("ones2", "ones1") or t[2] in ("ones2", "ones1")

    def _mat(t):
        return t[2] if t[1] in ("ones2", "ones1") else t[1]
    prod_t = [t for t in d2 if not _ones(t)]
    ones_t = sorted((t for t in d2 if _ones(t)), key=_mat)
    d2 = prod_t + ones_t
    OG = []
    for t, term in enumerate(ones_t, start=len(prod_t)):
        if OG and OG[-1][0] == _mat(term):
            OG[-1][2] = t + 1
        else:
            OG.append([_mat(term), t, t + 1])
    SL2 = [[0.0] * np_ for _ in d2]
    SR2 = [[0.0] * np_ for _ in d2]
    IND2 = [[0.0] * len(D2_IPS) for _ in d2]
    AB2 = []
    for t, (g, a, b, wl, wr, coef) in enumerate(d2):
        SL2[t][wl] = coef
        SR2[t][wr] = 1.0
        IND2[t][g] = 1.0
        AB2.append((a, b) if a <= b else (b, a))
    SL1 = [[0.0] * np_ for _ in d1]
    IND1 = [[0.0] * len(D1_IPS) for _ in d1]
    B1 = []
    for t, (g, a, b, wl, coef) in enumerate(d1):
        SL1[t][wl] = coef
        IND1[t][g] = 1.0
        B1.append((a, b) if a <= b else (b, a))
    # marg2: terms whose left (MGL) / right (MGR) coefficient is a k = 2 marginal correction (w(5) with D3,
    # w(6) with the kappa4 diagonal)
    MGL = [t for t, (g, a, b, wl, wr, coef) in enumerate(d2)
           if ('d3col' in (a, b) and wl in (16, 17)) or ('g4col' in (a, b) and wl in (18, 19))]
    MGR = [t for t, (g, a, b, wl, wr, coef) in enumerate(d2)
           if ('d3row' in (a, b) and wr in (16, 17)) or ('g4row' in (a, b) and wr in (18, 19))]
    return dict(MGL=MGL, MGR=MGR, SL2=SL2, SR2=SR2, IND2=IND2, AB2=AB2, SL1=SL1, IND1=IND1, B1=B1,
                NPROD=len(prod_t), OG=[tuple(g) for g in OG])


class Estimator(BaseEstimator):
    # cn9: truncation is only ~3% of the leaderboard MSE (cn3 -> cn7: -39% truncation error^2 for
    # -1.1% raw), so a cheaper old tier pays: the tier-1 basis is rebuilt by the range finder on
    # every other join only (kept and re-orthonormalised in between) and tier 2 starts one age
    # later at rank 160. Distance to the no-truncation chain ~ cn6's, C -1.7% vs cn7.
    # cn7: both age gates one transport later at the ranks of 504aldo's age law (rank that holds
    # accuracy ~3n/8 at age 4, n/4 at age 6, 7n/32 at age 7: 320 at age 5, ~192 at age 8). Each
    # source stays dense one transport longer and the tiers are sized for older, more
    # concentrated legs: 22% closer to the no-truncation chain together with the warm start, and
    # 1-2% cheaper than 384/224 at ages 4/7.
    AGE_OLD = int(_os.environ.get("V21_AGE_OLD", "5"))   # V21: sources with age > AGE_OLD are confined
    R_OLD = int(_os.environ.get("V21_R_OLD", "320"))     # V21: shared basis rank (lean ladder F72)
    AGE_OLD2 = int(_os.environ.get("V24_AGE_OLD2", "9"))  # V24: age gate of the nested tier (0 = off)
    R_OLD2 = int(_os.environ.get("V24_R_OLD2", "160"))    # V24: rank of the nested sub-basis U (8-dump frontier runs/v24_7_*_d0-7.log: 224 best, 128 cliff)
    QPASS2 = int(_os.environ.get("V24_QPASS2", "2"))      # V24: passes of the r1-space range finder
    R_FB = int(_os.environ.get('CN_RFB', '2'))    # cn35-39: feedback rank 2 (graded as cn21) + the cn26 hub fix; cn26: feedback rank 1 (rank 0 moves the chain 4e-5, rank 1 vs 2 < chaos); cn20/21: feedback rank 4 / 2 (D21 = one dominant singular pair + a flat bulk; the chain moves at chaos level); V18: rank of the D21 feedback thin legs (F69 lean ladder: 8/16/32 -> 2.17/2.14/2.14e-8)
    R_RES = int(_os.environ.get('CN_RRES', '4'))   # cn35-39: residual rank 16 / 8 / 4 / 2 / 1; rank of the S21 residual leg (V17 ladder on dumps 0/1: 16/32/64 -> 2.35/2.39/2.41e-8 at 0.492/0.502/0.518xB)

    def __init__(self) -> None:
        self._setup_rng = None
        self._wick_consts = _build_wick_consts()
        self._progs = {0: _term_prog(0), 1: _term_prog(1), 2: _term_prog(2)}
        self._i11 = WICK_PAIRS.index((1, 1))
        self._i21 = WICK_PAIRS.index((2, 1))
        self._i12 = WICK_PAIRS.index((1, 2))
        self._i31 = WICK_PAIRS.index((3, 1))
        self._k34 = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)
        d = getattr(ctx, "submission_dir", None)
        path = (Path(d) if d is not None else Path(__file__).resolve().parent) / "k34.npz"
        self._k34 = None
        self._k34_star = False
        if path.exists():
            z = fnp.load(str(path))  # pickle-free, 0 FLOPs
            self._k34 = {k: fnp.asarray(z[k], dtype=fnp.float32)
                         for k in ("mu", "sd", "W0", "b0", "W1", "b1", "W2", "b2")}
        self._k34_star = self._k34 is not None and self._k34["mu"].shape[0] == 12
        mpath = path.parent / "mp.npz"   # optg: message-passing corrector
        self._mp = None
        self._mpbuf = None
        self._mpones = None
        if self._k34_star and mpath.exists():
            z = fnp.load(str(mpath))
            self._mp = {k: fnp.asarray(z[k], dtype=fnp.float32) for k in
                        [f"{p}_{w}{i}" for p in "ABO" for w in "Wb" for i in range(3)]}

    # ------------------------------------------------------------------
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        # optc: objects alive before predict (the harness's request payload, modules, pools)
        # move to the permanent generation for its duration, so a full cyclic-GC pass
        # triggered inside predict scans only objects created here (outliers of ~+40 ms
        # every ~10 MLPs otherwise). Collection itself stays on; unfrozen on return.
        import gc as _gc
        _gc.freeze()
        try:
            return self._predict_core(mlp, budget)
        finally:
            _gc.unfreeze()

    def _predict_core(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        n = mlp.width
        f32 = fnp.float32
        st = _statics(n)
        metric2 = 2.0 ** 2
        L = len(mlp.weights)
        # vec+ww K4 transport and the fitted mean correction are calibrated for the
        # Phase-2 suite shape ONLY; any other shape (e.g. the grader smoke MLP)
        # runs the shape-generic V1.6 base path. The correction destabilizes small
        # widths (non-finite blowup observed at 16x64) -- never apply it off-suite.
        riders = (n == 1024 and L == len(CORR_BETA))

        C1l, C2l, SELCl = self._wick_consts
        C1 = fnp.asarray(C1l, dtype=f32)
        C2 = fnp.asarray(C2l, dtype=f32)
        SELC = fnp.asarray(SELCl, dtype=f32)
        FP = {}
        for mode, p in self._progs.items():
            FP[mode] = {k: fnp.asarray(v, dtype=f32)
                        for k, v in p.items() if k in ("SL2", "SR2", "IND2", "SL1", "IND1")}
        ones_n = fnp.ones(n, dtype=f32)
        ones_col = fnp.reshape(ones_n, (-1, 1))
        ones_row = ones_col.T
        ones2 = ones_col * ones_row
        beta_rows = [fnp.asarray(b, dtype=f32) for b in CORR_BETA]
        # V20: one persistent buffer for the d=2 term products (max terms over modes)
        t2max = max(len(p["AB2"]) for p in self._progs.values())
        TCH = 18  # PK2 term chunk: 18 n x n float32 planes = 72 MiB < the remote 100 MiB cap
        abbuf = fnp.empty((min(t2max, TCH), n, n), dtype=f32)
        abv = [abbuf[t] for t in range(abbuf.shape[0])]   # optc: plane views made once

        mu = fnp.zeros(n, dtype=f32)
        C = flops.as_symmetric(fnp.eye(n, dtype=f32), symmetry=(0, 1))
        A_st = P_st = Z_st = L_st = None
        newborn = None
        w2b_list = []
        STG = fnp.empty((2, n, n), dtype=f32) if riders else None    # optc: [a_b, C]
        STGo = fnp.empty((2, n, n), dtype=f32) if riders else None   # [W a_b, W C]
        s_list = []
        e_list = []
        c1_list = []   # per source: lambda_b * w1_b   (X3 = A*c1 + P*c2, column scalings)
        c2_list = []   # per source: w1_b^2 * dG_pre_b
        g3_list = []   # opts: per source, E[relu'''] = -alpha phi / var at birth (kappa4 star legs)
        cv_list = []   # opts: per source, Cov(h, z') = Phi var at birth
        y_list = []    # per source: (m/4) w2_b       (Y3 = y 1^T)
        g_prev = None  # post-ReLU kappa4 diagonal core of the previous layer
        var_prev = None
        lam_prev = 0.0
        regen = riders and not NO_REGEN  # memoryless kappa4 channel: suite shape only
        bufs = None
        legs = None
        r = min(int(self.R_RES), n)  # smoke shapes can be narrower than the rank
        rfb = 0 if NO_FB else min(int(self.R_FB), n)  # V18 feedback rank
        # V21: shared-basis state for old sources (suite-width only: r must be < n)
        r_old = int(self.R_OLD)
        confine = (not NO_CONFINE) and r_old < n
        age_old = int(self.AGE_OLD)
        ka = 0           # number of confined (old) sources = leading slots of the stacks
        Qc = None        # (n, r) current basis of the old legs (pre-wick of this layer)
        FAo = FPo = None   # (ka, r, n) static factors: A_s = Qc FA_s, P_s = Qc FP_s
        Sg = None        # (r, r) weighted Gram core of the TIER-1 legs in the factor basis
        # V24: nested tier 2 (slots [0:kb] of the stacks, the oldest sources)
        r2 = int(self.R_OLD2)
        age_old2 = int(self.AGE_OLD2)
        nest = confine and age_old2 > 0 and r2 < r_old
        kb = 0           # number of tier-2 sources (leading slots, subset of the ka confined)
        U = None         # (r1, r2) shared sub-basis of tier 2 inside Qc's factor space
        FA2 = FP2 = None   # (kb, r2, n) static factors: A_s = Qc U FA2_s
        S2 = None        # (r2, r2) weighted Gram core of tier 2 in the U basis
        QU = None        # (n, r2) = Qc U, formed once per layer
        dA_list = []     # per source: hub-column Gram weights of the A-type legs
        dP_list = []     # per source: hub-column Gram weights of the P-type legs
        R1T_st = R2T_st = None  # V18: static right factors of the thin legs, (k, n, rfb)
        Zf_st = None  # V18: transported thin legs [F1 | F2], (k, n, 2 rfb); separate from Z
                      # so the M-leg einsums (Z L^T, PPL Z^T) never see them
        K4_sigma = None
        K4_vec = None
        rows = []

        w1_prev = None  # wick w(1) of the previous layer, folded into WD
        al_prev = None  # cn3: rows kept for the next transports (index array) or None = all
        ckeep = None    # cm: column mask of the newborn (1 = live at birth)
        # cn: each source stores its columns live-first (perm_list, an exact relabelling of the
        # birth neurons) and has ms_list live columns rounded up to a multiple of 32; the young
        # transports and the young D21 hub run on the first max(ms) columns only.
        perm_list = []
        ms_list = []
        cperm = None    # cn: newborn's column permutation (set at its birth, used at the next linear)
        _ZB = [None]    # cn: zero block for the dead column tails of transported / formed legs

        def _zb(cnt, w):
            if _ZB[0] is None:
                _ZB[0] = fnp.zeros((L, n, n - CN_MIN), dtype=f32)
            return _ZB[0][:cnt, :, :w]
        learned = self._k34 is not None and riders and regen and BETA != 0.0
        K4p = None                       # previous layer's post-ReLU kappa4 diagonal K(4,)
        pre = None                       # previous layer's pre-activation (alpha, var, D3)
        k4_prev = fnp.zeros(n, dtype=f32)  # previous layer's kappa4 (z Gaussian at layer 1)

        for li, w in enumerate(mlp.weights):
            last = li == L - 1
            trim = last and not FULL_LAST  # V19: mean-only final layer
            skip_src = trim and NO_SRC_LAST
            w32 = w if w.dtype == fnp.float32 else w.astype(f32)
            W = w32.T
            # ---- linear ----
            mu = W @ mu
            if riders:
                C_pre = None   # optc: assembled after the W family (needs W C), below
            elif trim:
                # only var = diag(C_pre) is consumed: C_pre[a,a] = sum_i w32[i,a] (C w32)[i,a]
                C_pre = None
                var = fnp.maximum(fnp.sum(w32 * (C @ w32), axis=0), 1e-10)
            else:
                C_pre = fnp.einsum("ij,ia,jb->ab", C, w32, w32)
            # Source stacks evolve by W @ diag(w1_prev); the newborn (added after
            # last layer's wick) evolves by the raw W. Fold the wick into WD so
            # the stacks never need a separate (k,n,n) scaling pass.
            # Dense legs live in ping-pong slot buffers (F53: no fresh (k,n,n)
            # results, no concatenates): transport k slots into the spare buffer,
            # write the newborn into slot k, swap.
            if (A_st is not None or newborn is not None) and not skip_src:
                WD = W * fnp.reshape(w1_prev, (1, -1))
                WDb = fnp.reshape(WD, (1, n, n))
                # optc: with riders the young legs are pre-scaled by w1 at the wick stage, so
                # their transport (and [a_b, C]) multiplies by the raw W
                TW = (LeftTree(fnp.reshape(W, (1, n, n)), SLEV, n, "TV") if riders
                      else LeftTree(WDb, SLEV, n, "TW"))
                TWD = TW
                if al_prev is not None:
                    nl = al_prev.shape[0]
                    w32c = fnp.take(w32, al_prev, axis=0)          # (nl, n): kept inputs
                    Wc = w32c.T                                     # (n, nl)
                    w1l = fnp.take(w1_prev, al_prev)
                    WDc = Wc * fnp.reshape(w1l, (1, -1))
                    TWc = LeftTree(fnp.reshape(Wc, (1, n, nl)), SLEV, n, "TL", 24)
            if A_st is not None and not skip_src:
                k = A_st.shape[0]
                # V21: one source per layer crosses the age gate; join it to the shared
                # basis (post-wick legs, like the lean chain), skip the join at the last
                # layer (its legs are used once more, densely, for D3 only).
                ka_t = ka
                if confine and not last:
                    ka_t = max(ka, min(k, li - age_old))
                if ka_t > ka:
                    j = ka  # the joiner (exactly one per layer)
                    w1c = fnp.reshape(w1_prev, (-1, 1))
                    Aj = A_st[j] if riders else w1c * A_st[j]
                    Pj = P_st[j] if riders else w1c * P_st[j]
                    dAj = fnp.reshape(dA_list[j], (-1, 1))
                    dPj = fnp.reshape(dP_list[j], (-1, 1))
                    if al_prev is not None:
                        Om = fnp.copy(w32c[:, :r_old])     # fixed sketch over the kept rows
                        Qp = (fnp.reshape(w1l, (-1, 1)) * fnp.take(Qc, al_prev, axis=0)) if ka > 0 else None
                    else:
                        Om = fnp.copy(w32[:, :r_old])          # fixed sketch (n, r)
                        Qp = (w1c * Qc) if ka > 0 else None     # post-wick old basis
    # optc: with riders (QPASS = 1) the joiner's legs are staged as one batch-2 operand
                    rfs = riders and QPASS == 1
                    if WARM and ka > 0 and rfs:
                        # cn6: warm-started range finder. All legs are suffix products of the same
                        # transport chain and share its leading subspace, so the previous basis is
                        # a far better sketch than a fixed random slice: one subspace-iteration
                        # step from Qp (same cost) loses ~40% less old content per join.
                        Om = Qp
                    msj = ms_list[j] if rfs else n          # cn2: the joiner's live columns
                    if rfs and al_prev is not None:
                        S2j = _stage("rs", (2, nl, msj), f32)
                        fnp.take(Aj[:, :msj], al_prev, axis=0, out=S2j[0])
                        fnp.take(Pj[:, :msj], al_prev, axis=0, out=S2j[1])
                    elif rfs:
                        S2j = _stage("rs", (2, n, msj), f32)
                        fnp.copyto(S2j[0], Aj[:, :msj]); fnp.copyto(S2j[1], Pj[:, :msj])
                    keep_basis = rfs and ka > 0 and (KEEP == "all" or (KEEP == "alt" and li % 2 == 0))
                    if keep_basis:
                        # cn9: keep the transported basis (no range finder): the old content is kept
                        # exactly and the joiner is projected onto the shared leading subspace
                        Qn, _ = fnp.linalg.qr(Qp)
                    for _pass in range(0 if keep_basis else QPASS):
                        # Yq = G Om with G = Aj dA Aj^T + Pj dP Pj^T + Qp Sg Qp^T
                        if rfs:
                            T1j = _stage("rt", (2, r_old, msj), f32)
                            _smm(Om.T[None], S2j, RLEV, T1j, mb=RMB)        # (S2_b^T Om)^T
                            fnp.multiply(T1j, fnp.reshape(fnp.stack([dA_list[j][:msj], dP_list[j][:msj]]), (2, 1, msj)), out=T1j)
                            Yq = (_hub(S2j, T1j, RLEV, 2, RMB) if RFHUB
                                  else Aj @ T1j[0].T + Pj @ T1j[1].T)
                        else:
                            Yq = Aj @ (dAj * (Aj.T @ Om)) + Pj @ (dPj * (Pj.T @ Om))
                        if ka > 0:
                            # V24: full core of the confined legs = tier-1 core + lifted tier-2 core
                            Sfull = Sg if kb == 0 else Sg + U @ (S2 @ U.T)
                            Yq = Yq + Qp @ (Sfull @ _sd(Qp.T, Om))
                        Qn, _ = fnp.linalg.qr(Yq)           # (n, r) orthonormal
                        Om = Qn
                    if rfs:
                        F2j = fnp.zeros((2, r_old, n), dtype=f32) if msj < n else fnp.empty((2, r_old, n), dtype=f32)
                        _smm(Qn.T[None], S2j, RLEV, F2j[:, :, :msj], mb=RMB)
                        FAj, FPj = F2j[0:1], F2j[1:2]
                    else:
                        FAj = fnp.reshape(Qn.T @ Aj, (1, r_old, n))
                        FPj = fnp.reshape(Qn.T @ Pj, (1, r_old, n))
                    if ka > 0:
                        Tq = Qn.T @ Qp                      # (r, r) rotation of the old factors
                        if ka > kb:
                            TT = LeftTree(fnp.reshape(Tq, (1, Tq.shape[0], Tq.shape[1])), RLEV, n, "TT", RMB)
                            FAn = fnp.empty(FAo.shape, dtype=f32)
                            FPn = fnp.empty(FPo.shape, dtype=f32)
                            _mmc2(TT, FAo, FAn, FPo, FPn, RLEV, 8, RMB)
                            FAo, FPo = FAn, FPn
                            Sg = Tq @ (Sg @ Tq.T)
                            FAo = fnp.concatenate([FAo, FAj], axis=0)
                            FPo = fnp.concatenate([FPo, FPj], axis=0)
                        else:
                            FAo, FPo, Sg = FAj, FPj, None
                        if kb > 0:
                            U = Tq @ U                      # V24: the sub-basis rides along
                    else:
                        FAo, FPo = FAj, FPj
                    if riders and msj < n:
                        Fa, Fp = FAj[0][:, :msj], FPj[0][:, :msj]
                        Sj = _sd(Fa, dAj[:msj] * Fa.T) + _sd(Fp, dPj[:msj] * Fp.T)
                    else:
                        Sj = _sd(FAj[0], dAj * FAj[0].T) + _sd(FPj[0], dPj * FPj[0].T)
                    Sg = Sj if Sg is None else Sg + Sj
                    # V24: nested move of the oldest tier-1 member into tier 2, in the
                    # orthonormal coordinates Qn of this rebuild (lean: youngest gate first,
                    # then the older gate on the already-confined legs).
                    kb_t = kb
                    if nest:
                        kb_t = max(kb, min(ka_t - 1, li - age_old2))
                    if kb_t > kb:
                        assert kb_t == kb + 1, (li, kb, kb_t)
                        jb = kb                             # stack slot of the mover
                        FA1 = FAo[0]
                        FP1 = FPo[0]
                        dAb = fnp.reshape(dA_list[jb], (-1, 1))
                        dPb = fnp.reshape(dP_list[jb], (-1, 1))
                        S_s = _sd(FA1, dAb * FA1.T) + _sd(FP1, dPb * FP1.T)   # (r1, r1)
                        G2 = S_s if kb == 0 else S_s + U @ (S2 @ U.T)
                        Om2 = fnp.copy(w32[:r_old, :r2])                        # sketch (r1, r2)
                        if WARM and kb > 0:
                            Om2 = U                                             # cn6: warm start (rotated sub-basis)
                        for _pass in range(int(self.QPASS2)):
                            Un, _ = fnp.linalg.qr(G2 @ Om2)
                            Om2 = Un
                        FA2j = fnp.reshape(Un.T @ FA1, (1, r2, n))
                        FP2j = fnp.reshape(Un.T @ FP1, (1, r2, n))
                        if kb > 0:
                            T2 = Un.T @ U                                       # (r2, r2)
                            FA2 = fnp.concatenate([fnp.einsum("pq,kqn->kpn", T2, FA2), FA2j], axis=0)
                            FP2 = fnp.concatenate([fnp.einsum("pq,kqn->kpn", T2, FP2), FP2j], axis=0)
                        else:
                            FA2, FP2 = FA2j, FP2j
                        S2 = Un.T @ (G2 @ Un)
                        U = Un
                        FAo = FAo[1:]
                        FPo = FPo[1:]
                        Sg = Sg - S_s
                        kb = kb_t
                    Qc = _sd(Wc if al_prev is not None else W, Qn, 4, 24)   # wick already inside Qn
                    ka = ka_t
                elif ka > 0:
                    Qc = (WDc @ fnp.take(Qc, al_prev, axis=0)) if al_prev is not None else WD @ Qc
                if ka > kb:
                    # formed dense legs of the tier-1 sources (pre-wick of this layer)
                    Qc1 = fnp.reshape(Qc, (1, n, Qc.shape[1]))
                    TQ = LeftTree(Qc1, RLEV, n, "TQ", RMB)
                    w1t = max(ms_list[kb:ka])
                    if w1t < n:
                        # cn2: tier-1 columns >= w1t are zero in the factors: form the live block
                        _mmc2(TQ, FAo[:, :, :w1t], legs["A1"][kb:ka, :, :w1t],
                              FPo[:, :, :w1t], legs["P1"][kb:ka, :, :w1t], RLEV, T1B, RMB)
                        zb = _zb(ka - kb, n - w1t)
                        fnp.copyto(legs["A1"][kb:ka, :, w1t:], zb)
                        fnp.copyto(legs["P1"][kb:ka, :, w1t:], zb)
                    else:
                        _mmc2(TQ, FAo, legs["A1"][kb:ka], FPo, legs["P1"][kb:ka], RLEV, T1B, RMB)
                if kb > 0:
                    # V24: tier-2 legs through the sub-basis
                    QU = Qc @ U
                    TU = LeftTree(fnp.reshape(QU, (1, n, QU.shape[1])), RLEV, n, "TU", T2MB)
                    _mmc2(TU, FA2, legs["A1"][:kb], FP2, legs["P1"][:kb], RLEV, 8, T2MB)
                wt = max(ms_list[ka:k]) if ka < k else n
                if ka < k and al_prev is not None:
                    # cn3: kept rows only; cn5: each batch-SB chunk of the young stack on its
                    # own live column block (same Strassen calls as one stack-wide width)
                    for c0 in range(ka, k, SBK):
                        c1 = min(k, c0 + SBK)
                        wc = max(ms_list[c0:c1])
                        GA = _stage("gA", (c1 - c0, nl, wc), f32)
                        fnp.take(A_st[c0:c1, :, :wc], al_prev, axis=1, out=GA)
                        _mmc(TWc, GA, SLEV, legs["A1"][c0:c1, :, :wc])
                        fnp.take(P_st[c0:c1, :, :wc], al_prev, axis=1, out=GA)
                        _mmc(TWc, GA, SLEV, legs["P1"][c0:c1, :, :wc])
                        if wc < n:
                            zb = _zb(c1 - c0, n - wc)
                            fnp.copyto(legs["A1"][c0:c1, :, wc:], zb)
                            fnp.copyto(legs["P1"][c0:c1, :, wc:], zb)
                elif ka < k and wt < n:
                    # cn: the young legs' columns >= wt are dead-at-birth (zero): transport the
                    # live block only and keep the rest of the slots zero
                    _mmc(TW, A_st[ka:, :, :wt], SLEV, legs["A1"][ka:k, :, :wt])
                    _mmc(TW, P_st[ka:, :, :wt], SLEV, legs["P1"][ka:k, :, :wt])
                    zb = _zb(k - ka, n - wt)
                    fnp.copyto(legs["A1"][ka:k, :, wt:], zb)
                    fnp.copyto(legs["P1"][ka:k, :, wt:], zb)
                elif ka < k:
                    _mmc(TW, A_st[ka:], SLEV, legs["A1"][ka:k])
                    _mmc(TW, P_st[ka:], SLEV, legs["P1"][ka:k])
                if al_prev is not None:
                    WDcb = fnp.reshape(WDc, (1, n, nl))
                    Z_st = fnp.matmul(WDcb, fnp.take(Z_st, al_prev, axis=1))
                    if Zf_st is not None:
                        Zf_st = fnp.matmul(WDcb, fnp.take(Zf_st, al_prev, axis=1))
                else:
                    Z_st = fnp.matmul(WDb, Z_st)
                    if Zf_st is not None:
                        Zf_st = fnp.matmul(WDb, Zf_st)
                legs["A0"], legs["A1"] = legs["A1"], legs["A0"]
                legs["P0"], legs["P1"] = legs["P1"], legs["P0"]
            else:
                k = 0
            if newborn is not None and not skip_src:
                a_b, Rr, Lr, s_b, e_b, Ff = newborn
                if legs is None:
                    legs = {nm: fnp.empty((L - 1, n, n), dtype=f32)
                            for nm in ("A0", "A1", "P0", "P1")}
                if riders and al_prev is not None:
                    # cn3: newborn leg over the kept rows; W C keeps the full inner dimension
                    mbn = ms_list[k]                      # cn6: the newborn's live column block
                    GB = _stage("gB", (1, nl, mbn), f32)
                    fnp.take(STG[0][:, :mbn], al_prev, axis=0, out=GB[0])
                    _mmc(TWc, GB, SLEV, legs["A0"][k:k + 1, :, :mbn])
                    if mbn < n:
                        fnp.copyto(legs["A0"][k:k + 1, :, mbn:], _zb(1, n - mbn))
                    _mmc(TW, STG[1:2], SLEV, STGo[1:2])
                elif riders:
                    _mmc(TW, STG, SLEV, STGo)                 # [W a_b, W C] in one call
                    fnp.copyto(legs["A0"][k], STGo[0])
                else:
                    _mmc(TWD, fnp.reshape(a_b, (1, n, n)), SLEV, legs["A0"][k:k + 1])  # a_b holds C_off
                if ckeep is not None and perm_list[k] is not None:
                    fnp.take(W, perm_list[k], axis=1, out=legs["P0"][k])     # cn: live-first columns
                    fnp.multiply(legs["P0"][k], fnp.reshape(ckeep, (1, -1)), out=legs["P0"][k])
                elif ckeep is not None:
                    fnp.multiply(W, fnp.reshape(ckeep, (1, -1)), out=legs["P0"][k])
                else:
                    fnp.copyto(legs["P0"][k], W)
                A_st = legs["A0"][:k + 1]
                P_st = legs["P0"][:k + 1]
                z_n = fnp.reshape(W @ Rr, (1, n, Rr.shape[1]))
                l_n = fnp.reshape(Lr, (1, n, Lr.shape[1]))
                if Z_st is None:
                    Z_st, L_st = z_n, l_n
                else:
                    Z_st = fnp.concatenate([Z_st, z_n], axis=0)
                    L_st = fnp.concatenate([L_st, l_n], axis=0)
                if Ff is not None:
                    zf_n = fnp.reshape(W @ Ff, (1, n, Ff.shape[1]))
                    Zf_st = zf_n if Zf_st is None else fnp.concatenate([Zf_st, zf_n], axis=0)
                s_list.append(s_b)
                e_list.append(e_b)
                newborn = None

            if riders:
                Ywc = W if li == 0 else STGo[1]           # W C (C = I at layer 0)
                if trim:
                    var = fnp.maximum(fnp.sum(Ywc * W, axis=1), 1e-10)
                else:
                    C_pre = _cpre_sym(Ywc, w32, n, lev=5)   # cn51: 5 levels, leaf 16 x 32 x 16 (teammate note: deeper Strassen on the covariance)
            # ---- WK slices ----
            if trim:
                C_off = None
            else:
                var = fnp.maximum(fnp.diag(C_pre), 1e-10)
                C_off = _zero_diag(C_pre)
            mode = 0 if A_st is None else 1
            if mode == 1 and skip_src:
                D3, D21 = fnp.zeros(n, dtype=f32), None
                if regen:
                    WW = W * W
                    dG = WW @ (g_prev - var_prev * lam_prev) + var * lam_prev
                    g4row = dG * METRIC_C
                    wk4m = wk431 = None
                elif riders:
                    g4row = ((W * W) @ K4_vec) * 0.5 * float(st["wk4_c4"] * metric2)
                    wk4m = None
                else:
                    g4row = ones_n * (K4_sigma * float(st["wk4_c4"] * metric2))
                    wk4m = None
            elif mode == 1:
                if bufs is None:
                    bufs = {nm: fnp.empty((L - 1, n, n), dtype=f32)
                            for nm in ("ap", "pp", "t", "mp", "la", "lp", "xt", "yt", "u")}
                D3, D21 = self._dslices(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,
                                        c1_list, c2_list, y_list, n, bufs,
                                        r, rfb, Zf_st, R1T_st, R2T_st,
                                        need_d21=not trim,
                                        ka=ka, FAo=FAo, FPo=FPo, Qc=Qc,
                                        kb=kb, FA2=FA2, FP2=FP2, U2=U,
                                        yw=max(ms_list[ka:A_st.shape[0]]) if ka < A_st.shape[0] else n,
                                        w1t=max(ms_list[kb:ka]) if ka > kb else n,
                                        msl=ms_list)
                if regen:
                    # F68: exact transported diagonal of the regenerated core
                    # G = diag(g) + lam C_off:  dG = (W*W) g + lam (diag(C_pre)
                    # - (W*W) var).  NOT an einsum: same-object repeated operands
                    # crash flopscope (F45).
                    WW = W * W
                    if BETA != 0.0:
                        # V25: adaptive lambda. dG is affine in lam: t_g + lam * t_v.
                        t_g = WW @ g_prev
                        t_v = var - WW @ var_prev
                        dG0 = t_g + t_v * lam_prev                # table value first
                        rr = fnp.mean(dG0) / fnp.mean(var)
                        ref = float(REF_R[min(li - 1, len(REF_R) - 1)])
                        # clamp: an odd MLP must never turn the rule into NaN/inf (zero fallback)
                        lam_prev = lam_prev * fnp.power(fnp.clip(rr / ref, 0.5, 2.0), BETA)
                        dG = t_g + t_v * lam_prev                 # var == diag(C_pre)
                    else:
                        dG = WW @ (g_prev - var_prev * lam_prev) + var * lam_prev  # var == diag(C_pre)
                    g4row = dG * METRIC_C
                    g22c = fnp.reshape(dG * (METRIC_C / 6.0), (-1, 1))
                    wk4m = _zero_diag(g22c + g22c.T)
                    wk431 = None if trim else C_off * (0.5 * METRIC_C * lam_prev)
                    if learned:
                        star = (self._star(A_st, P_st, g3_list, cv_list, bufs, n)
                                if self._k34_star else None)
                        D3, g4row = self._k34_step(li, D3, g4row, var, mu, WW, w32, K4p, t_v, pre,
                                                   A_st[A_st.shape[0] - 1], w1_prev, k4_prev, star,
                                                   perm=perm_list[A_st.shape[0] - 1])
                        if self._mp is not None and li >= MP_LI_MIN:
                            if D21 is not None:
                                D3, g4row, wk431 = self._mp_step(C_off, D21, wk431)
                            else:                                  # no D21 here (trimmed last layer): S1 outputs
                                x_, _, s3_, s4_ = self._mpx
                                D3, g4row = x_[:, 13] * s3_, x_[:, 12] * s4_
                        g22c = fnp.reshape(g4row * (1.0 / 6.0), (-1, 1))
                        wk4m = _zero_diag(g22c + g22c.T)
                    k4_prev = g4row
                    if _os.environ.get("V17_DEBUG", "0") == "1":
                        DEBUG.append(dict(layer=li, dG=dG, g_prev=g_prev, var_prev=var_prev,
                                          lam=lam_prev, var=var, D3=D3, D21=D21))
                elif riders:
                    # vec+ww: per-neuron K4 diagonal content transported by the
                    # true (W*W) row action; one avg-metric cup replaced
                    # (/metric_c = *0.5). NOT an einsum: same-object repeated
                    # operands crash flopscope (F45).
                    gv = ((W * W) @ K4_vec) * 0.5
                    g4row = gv * float(st["wk4_c4"] * metric2)
                    g22 = gv * float(0.5 * st["wk4_c22"] * metric2)
                    g22c = fnp.reshape(g22, (-1, 1))
                    wk4m = _zero_diag(g22c + g22c.T)
                else:
                    g4v = K4_sigma * float(st["wk4_c4"] * metric2)
                    g22v = K4_sigma * float(st["wk4_c22"] * metric2)
                    g4row = ones_n * g4v
                    wk4m = _zero_diag(ones2 * g22v)
            else:
                D3 = D21 = g4row = wk4m = None

            # ---- wick matrix ----
            sigma = fnp.sqrt(var)
            alpha = mu / sigma
            phi = flops.stats.norm.pdf(alpha).astype(f32)
            Phi = flops.stats.norm.cdf(alpha).astype(f32)
            a2 = alpha * alpha
            a3 = a2 * alpha
            a4 = a3 * alpha
            a5 = a4 * alpha
            APOW = fnp.stack([ones_n, alpha, a2, a3, a4, a5], axis=1)
            inv = 1.0 / sigma
            i2 = inv * inv
            i3 = i2 * inv
            i4 = i3 * inv
            i5 = i4 * inv
            i6 = i5 * inv
            s2 = sigma * sigma
            s3 = s2 * sigma
            s4 = s3 * sigma
            SB = fnp.stack([i6, i5, i4, i3, i2, inv, ones_n, sigma, s2, s3, s4], axis=1)
            phic = fnp.reshape(phi, (-1, 1))
            Phic = fnp.reshape(Phi, (-1, 1))
            W_all = ((APOW @ C1) * phic + (APOW @ C2) * Phic) * (SB @ SELC)
            WT = W_all.T

            # ---- nonlin terms (fused) ----
            pmode = mode if (mode == 0 or (regen and not NO_WK431)) else 2
            prog = self._progs[pmode]
            fpm = FP[pmode]
            b2 = {"ones2": ones2, "c_off": C_off}
            if XC3 != 0.0 and C_off is not None:
                b2["c3"] = C_off * C_off * C_off
            b1 = {"ones1": ones_n}
            if mode == 1:
                if not trim:
                    d3col = fnp.reshape(D3, (-1, 1)) * ones_row
                    g4col = fnp.reshape(g4row, (-1, 1)) * ones_row
                    b2.update({"d21": D21, "d21T": D21.T, "wk4m": wk4m,
                               "d3col": d3col, "d3row": d3col.T,
                               "g4col": g4col, "g4row": g4col.T})
                    if regen and not NO_WK431:
                        b2["wk431"] = wk431
                b1.update({"d3": D3, "g4": g4row})
            ab_cache = {}

            def ab(pair, tbl):
                if pair in ab_cache:
                    return ab_cache[pair]
                a, b = pair
                if a in ("ones2", "ones1"):
                    val = tbl[b]
                elif b in ("ones2", "ones1"):
                    val = tbl[a]
                else:
                    val = tbl[a] * tbl[b]
                ab_cache[pair] = val
                return val

            if not trim:
                ab2 = prog["AB2"]
                WL2 = fpm["SL2"] @ WT
                WR2 = fpm["SR2"] @ WT
                if XMARG2 != 0.0 and riders:
                    tail = (alpha <= -2.0).astype(f32)
                    for t_ in prog["MGL"]:
                        fnp.multiply(WL2[t_], tail, out=WL2[t_])
                    for t_ in prog["MGR"]:
                        fnp.multiply(WR2[t_], tail, out=WR2[t_])
                PK2 = None
                npr = prog["NPROD"]
                for c0 in range(0, npr, TCH):
                    c1 = min(npr, c0 + TCH)
                    for t_ in range(c0, c1):
                        a_, b_ = ab2[t_]
                        fnp.multiply(b2[a_], b2[b_], out=abv[t_ - c0])
                    part = fnp.einsum("tij,ti,tj,tg->gij", abbuf[:c1 - c0], WL2[c0:c1], WR2[c0:c1],
                                      fpm["IND2"][c0:c1])
                    PK2 = part if PK2 is None else PK2 + part
                for x_, t0, t1 in prog["OG"]:
                    part = fnp.einsum("ij,ti,tj,tg->gij", b2[x_], WL2[t0:t1], WR2[t0:t1],
                                      fpm["IND2"][t0:t1])
                    PK2 = part if PK2 is None else PK2 + part
            B1stack = fnp.stack([ab(p, b1) for p in prog["B1"]], axis=0)
            WL1 = fpm["SL1"] @ WT
            PK1 = fnp.einsum("ti,ti,tg->gi", B1stack, WL1, fpm["IND1"])

            if not trim:
                pk11 = PK2[0]
                pk11 = _zero_diag((pk11 + pk11.T) * 0.5)
                pk21 = _zero_diag(PK2[1])
                pk22 = PK2[2]
                pk22 = _zero_diag((pk22 + pk22.T) * 0.5)
            pk1v, pk2v, pk3v, pk4v = PK1[0], PK1[1], PK1[2], PK1[3]

            # ---- online mean correction: delta = feats @ beta[l] (13 dots/neuron);
            # mu still holds the pre-nonlin mean here, pk1v == K(1,) is the pred.
            if riders and not NO_CORR:
                if mode == 1:
                    D3f = D3
                    D21n = fnp.abs(D21) @ ones_n if D21 is not None else fnp.zeros(n, dtype=f32)
                    k4f = ones_n * K4_sigma
                else:
                    D3f = fnp.zeros(n, dtype=f32)
                    D21n = fnp.zeros(n, dtype=f32)
                    k4f = fnp.zeros(n, dtype=f32)
                feats = fnp.stack([ones_n, mu, var, sigma, alpha, fnp.abs(alpha),
                                   phi, Phi, pk1v, D3f, D21n, k4f, sigma * phi],
                                  axis=1)
                delta = feats @ beta_rows[li]
            else:
                delta = None

            if last:
                rows.append(pk1v if delta is None else pk1v + delta)
                break

            # ---- wick old blocks + dslice scalings ----
            w1 = W_all[:, self._i11]
            w2 = W_all[:, self._i21]
            w1col = fnp.reshape(w1, (-1, 1))
            w1_prev = w1  # stacks pick up this wick via WD at the next linear
            al_prev = None
            if riders and RW_TAU > 0:
                nd = int(fnp.sum(Phi < RW_TAU))
                qd = min((nd // 32) * 32, n - CN_MIN)   # cn6: rows in steps of 32
                if qd > 0:
                    al_prev = fnp.sort(fnp.argsort(Phi)[qd:])
            ckeep = (Phi >= CM_TAU).astype(f32) if (riders and CM_TAU > 0) else None
            cperm = None
            if ckeep is not None and CN:
                cperm = fnp.argsort(-Phi)                    # live (large Phi) first, dead last
                m_live = int(fnp.sum(ckeep))
                ms_new = min(n, max(CN_MIN, -(-m_live // 32) * 32))
            if riders and A_st is not None:
                # optc: pre-scale the young legs (incl. next layer's joiner) by w1
                w1c3 = fnp.reshape(w1, (1, n, 1))
                fnp.multiply(A_st[ka:], w1c3, out=A_st[ka:])
                fnp.multiply(P_st[ka:], w1c3, out=P_st[ka:])
            if mode == 1:
                D3_w = D3 * w1 ** 3
                D21_w = (w1col * w1col) * D21 * fnp.reshape(w1, (1, -1))
            else:
                D3_w = None
                D21_w = None

            # ---- new source (V1.6, struct-free Y1 = a_b * D(w2)) ----
            a_b = fnp.multiply(w1col, C_off, out=STG[0]) if riders else w1col * C_off
            if ckeep is not None:
                fnp.multiply(a_b, fnp.reshape(ckeep, (1, -1)), out=a_b)   # cm: no dead-at-birth columns
            if rfb > 0 and mode == 1:
                # V18 (F69): D21 feedback thin legs. D21 ~ Qf Bf (rank rfb range finder,
                # one power iteration, sketch = a slice of the layer weight).
                w3 = W_all[:, self._i31]
                Omf = fnp.copy(w32[:, :rfb + FB_OS])  # V20: contiguous sketch (cn8: + FB_OS oversampling)
                Yf = D21 @ Omf
                Qf, _ = fnp.linalg.qr(Yf)
                Yf = D21 @ (D21.T @ Qf)
                Qf, _ = fnp.linalg.qr(Yf)
                Bf = Qf.T @ D21                       # D21 ~= Qf @ Bf
                if FB_OS > 0:
                    # cn8: oversampled range finder truncated to rank rfb (Halko-Martinsson-Tropp: a few
                    # extra sketch columns bring a small-rank range finder close to the optimal subspace)
                    Uf, sf, Vf = fnp.linalg.svd(Bf, full_matrices=False)
                    Qf = Qf @ Uf[:, :rfb]
                    Bf = fnp.reshape(sf[:rfb], (-1, 1)) * Vf[:rfb]
                F1_b = fnp.reshape(w2, (-1, 1)) * Qf            # Xt = F1 R1, R1 = 1.5 Bf
                R1T_b = Bf.T * 1.5                              # (n, rfb) = R1^T
                F2_b = w1col * Bf.T                             # Yt = F2 R2, R2 = 0.5 Qf^T d(w3)
                R2T_b = Qf * fnp.reshape(w3 * 0.5, (-1, 1))     # (n, rfb) = R2^T
                Xt_b = F1_b @ R1T_b.T
                Yt_b = F2_b @ R2T_b.T
                X1_b = a_b * 3.0 + Xt_b
                Y1_b = a_b * fnp.reshape(w2, (1, -1)) + Yt_b
                xd = fnp.diag(Xt_b)
                yd = fnp.diag(Yt_b)
                # birth dslices of B1 with P = I (general form; diag(a_b) = 0)
                D21_new = (fnp.reshape(xd, (-1, 1)) * Y1_b.T + X1_b * Y1_b
                           + fnp.reshape(yd, (-1, 1)) * X1_b.T) * (1.0 / 3.0)
                D3_new = xd * yd
            else:
                # birth dslices of B1 with P = I: diag(a_b) = 0, so only the a*Y1 term
                # survives: D21_new = a_b^2 * D(w2)-row; D3_new = 0.
                D21_new = (a_b * a_b) * fnp.reshape(w2, (1, -1))
                D3_new = None
                if rfb > 0:
                    F1_b = fnp.zeros((n, rfb), dtype=f32)
                    F2_b = F1_b
                    R1T_b = F1_b
                    R2T_b = F1_b
            if regen and mode == 1 and not NO_FEED:
                # K4->K3 feed (F68): X3 = diag(w1^2 dG) + lam a_b d(w1), Y3 = y 1^T
                # with y = (m/4) w2; M_t1 = u v^T, u = (m/4) w2*dG, v = w1^2.
                # Birth (P = I) dslices: D21 += (1/3)[dgw y^T + 2 y (.) X3_off]
                # + (1/3) v u^T;  D3 += dgw*y + u*v.
                w1sq = w1 * w1
                y_b = w2 * (0.25 * METRIC_C)
                dgw = w1sq * dG
                u_b = y_b * dG
                c1_b = w1 * lam_prev
                D21_new = (D21_new
                           + fnp.reshape(dgw, (-1, 1)) * fnp.reshape(y_b, (1, -1)) * (1.0 / 3.0)
                           + fnp.reshape(y_b * (2.0 / 3.0), (-1, 1)) * (a_b * fnp.reshape(c1_b, (1, -1)))
                           + fnp.reshape(w1sq, (-1, 1)) * fnp.reshape(u_b * (1.0 / 3.0), (1, -1)))
                D3_new = ((dgw * y_b + u_b * w1sq) if D3_new is None
                          else D3_new + dgw * y_b + u_b * w1sq)
            else:
                y_b = fnp.zeros(n, dtype=f32)
                u_b = y_b
                c1_b = y_b
                dgw = y_b
                w1sq = y_b
            rep21 = D21_new if D21_w is None else D21_w + D21_new
            rep21 = _zero_diag(rep21)

            # ---- pK -> K (per-entry; d=1 outputs are vector ops, d=2 few entries) ----
            pk = {(1,): pk1v, (2,): pk2v, (3,): pk3v, (4,): pk4v,
                  (1, 1): pk11, (2, 1): pk21, (2, 2): pk22}

            def pk_slice(vec):
                nz = tuple(sorted((x for x in vec if x > 0), reverse=True))
                base = pk[nz]
                if len(vec) == 1:
                    return base
                if len(nz) == 2:
                    return base if vec == tuple(sorted(vec, reverse=True)) else base.T
                return (fnp.reshape(base, (-1, 1)) if vec[0] > 0
                        else fnp.reshape(base, (1, -1)))

            # optc: PK2K_TABLE written out (same terms, ~half the flopscope ops). pk11, pk21,
            # pk22 have zero diagonals, so every product below does too (no fill_diagonal).
            p1, p2, p3, p4 = pk1v, pk2v, pk3v, pk4v
            p1s = p1 * p1
            K1v = p1
            K2v = p2 - p1s
            K3v = p3 + p1 * (p1s * 2.0 - p2 * 3.0)
            K4v = p4 - p1 * (p3 * 4.0) - (p2 * p2) * 3.0 + p1s * (p2 * 12.0 - p1s * 6.0)
            c1 = fnp.reshape(p1, (-1, 1))
            r1 = fnp.reshape(p1, (1, -1))
            K11 = pk11
            K21 = pk21 - (c1 * 2.0) * pk11
            S = r1 * pk21
            K22 = pk22 + pk11 * ((c1 * r1) * 4.0 - pk11 * 2.0) - (S + S.T) * 2.0

            # ---- assemble ----
            mu = K1v if delta is None else K1v + delta
            if riders:
                C = STG[1]                                 # optc: rides the W family next layer
                fnp.add(K11, K11.T, out=C)
                fnp.multiply(C, 0.5, out=C)
                fnp.add(C, fnp.diagflat(K2v), out=C)
            else:
                C = fnp.diagflat(K2v) + (K11 + K11.T) * 0.5
                C = flops.as_symmetric(C, symmetry=(0, 1))
            S3c = K3v - D3_w if D3_w is not None else K3v
            if D3_new is not None:
                S3c = S3c - D3_new
            S21 = _zero_diag(K21 - rep21)
            # S21 = S_sep + Rres: S_sep = exact leading (2,1) Wick term (the (c_off,
            # ones2) term of pk21 minus 2*pk1*[(c_off, ones2) term of pk11]); M_b^T
            # part 3*S_sep^T = 3*a_b*diag(e) rides on the A leg for free. Rres is
            # compressed to rank r (randomized range finder, one power iteration).
            e_b = W_all[:, self._i12] - 2.0 * pk1v * w1
            S_sep = _zero_diag(fnp.reshape(e_b, (-1, 1)) * C_off * fnp.reshape(w1, (1, -1)))
            Rres = S21 - S_sep
            Om = fnp.copy(w32[:, :r])  # V20: contiguous sketch
            Y = Rres @ Om
            Q, _ = fnp.linalg.qr(Y)
            Y = Rres @ (Rres.T @ Q)
            Q, _ = fnp.linalg.qr(Y)
            Bm = Q.T @ Rres                      # Rres ~= Q @ Bm
            # M_b = diag(S3c) + 3 S21^T = diag(S3c) + 3 a_b diag(e_b) + 3 Rr Lr^T
            # with Rr = Bm^T (n,r) [transported as Z = P Rr] and Lr = Q (static).
            # extra thin columns: M_t1 = u v^T -> Z column P u, L column v; and the Y3
            # row vector y is a LEFT factor of the hub tensor, so it is transported
            # like a leg (y(l) = P(l) y): Z column P y with a ZERO L column (keeps it
            # out of the M leg). Zeros off-suite / at layer 0 so every source keeps
            # r+2 columns.
            Rr_full = fnp.concatenate([Bm.T * 3.0, fnp.reshape(u_b, (-1, 1)),
                                       fnp.reshape(y_b, (-1, 1))], axis=1)
            Lr_full = fnp.concatenate([Q, fnp.reshape(w1sq, (-1, 1)),
                                       fnp.zeros((n, 1), dtype=f32)], axis=1)
            # V18: thin feedback legs [F1 | F2] travel in their own stack Zf (transported
            # like Z, never entering the M-leg einsums); right factors R1T/R2T are static.
            Ff_b = fnp.concatenate([F1_b, F2_b], axis=1) if rfb > 0 else None
            newborn = (C_off, Rr_full, Lr_full, S3c, e_b, Ff_b)  # W a_b = WD C_off (optc)
            if rfb > 0:
                if cperm is not None:                     # cn: rows of R1T/R2T are source columns
                    R1T_b = fnp.take(R1T_b, cperm, axis=0)
                    R2T_b = fnp.take(R2T_b, cperm, axis=0)
                r1n = fnp.reshape(R1T_b, (1, n, rfb))
                r2n = fnp.reshape(R2T_b, (1, n, rfb))
                if R1T_st is None:
                    R1T_st, R2T_st = r1n, r2n
                else:
                    R1T_st = fnp.concatenate([R1T_st, r1n], axis=0)
                    R2T_st = fnp.concatenate([R2T_st, r2n], axis=0)
            w2b_list.append(w2)
            if self._k34_star:
                g3_list.append((alpha * phi) * (-1.0 / var))
                cv_list.append(Phi * var)
            # V21: hub-column Gram weights of this source's legs (X1 = 3A, Y1 ~ A d(w2),
            # M ~ P d(s) + 3 A d(e)): A-type 9 + w2^2 + 9 e^2, P-type 1 + s^2
            dA_list.append(9.0 + w2 * w2 + 9.0 * e_b * e_b)
            dP_list.append(1.0 + S3c * S3c)
            c1_list.append(c1_b)
            c2_list.append(dgw)
            y_list.append(y_b)
            if cperm is not None:
                # cn: relabel this source's columns live-first (exact; birth slices above are
                # neuron-space and already formed)
                w2b_list[-1] = fnp.take(w2b_list[-1], cperm)
                if self._k34_star:
                    g3_list[-1] = fnp.take(g3_list[-1], cperm)
                    cv_list[-1] = fnp.take(cv_list[-1], cperm)
                dA_list[-1] = fnp.take(dA_list[-1], cperm)
                dP_list[-1] = fnp.take(dP_list[-1], cperm)
                c1_list[-1] = fnp.take(c1_list[-1], cperm)
                c2_list[-1] = fnp.take(c2_list[-1], cperm)
                newborn = (C_off, Rr_full, fnp.take(Lr_full, cperm, axis=0),
                           fnp.take(S3c, cperm), fnp.take(e_b, cperm), Ff_b)
                fnp.copyto(STG[0], fnp.take(STG[0], cperm, axis=1))   # newborn birth leg a_b
                ckeep = fnp.take(ckeep, cperm)                         # live-first mask for P
                perm_list.append(cperm)
                ms_list.append(ms_new)
            else:
                perm_list.append(None)
                ms_list.append(n)
            if regen:
                # post-ReLU kappa4 diagonal core (r=1 matrix-core harmonic projection)
                k22row = K22 @ ones_n
                g_prev = ((K4v + k22row) * float(st["cA"])
                          + (fnp.sum(K4v) + fnp.sum(k22row)) * float(st["cI"]))
                var_prev = K2v
                lam_prev = float(LAM[min(li, len(LAM) - 1)])
            K4_sigma = (fnp.sum(K4v) * float(st["k4_c4"])
                        + fnp.sum(K22) * float(st["k4_c22"])) * float(st["P2"])
            # per-neuron K4 diagonal content (mean equals K4_sigma by construction;
            # K22 is symmetric so K22 @ ones == its column sums)
            if riders:
                K4_vec = (K4v * float(st["k4_c4"])
                          + (K22 @ ones_n) * float(st["k4_c22"])) * float(n * st["P2"])
            K4p = K4v
            pre = (alpha, var, D3 if D3 is not None else fnp.zeros(n, dtype=f32))
            rows.append(mu)

        return fnp.stack(rows, axis=0)

    # ------------------------------------------------------------------
    def _star(self, A_st, P_st, g3_list, cv_list, bufs, n):
        """opts: transported kappa4 star terms of the sources older than the newest,
        sum_k sum_j P_kij g3_kj v_kij^3 with v = A + P diag(cv_k) (O(n^2) per source)."""
        ko = A_st.shape[0] - 1
        if ko < 1:
            return fnp.zeros(n, dtype=fnp.float32)
        V = bufs["t"][:ko]
        U = bufs["u"][:ko]
        fnp.multiply(P_st[:ko], fnp.reshape(fnp.stack(cv_list[:ko], axis=0), (ko, 1, n)), out=V)
        fnp.add(V, A_st[:ko], out=V)
        fnp.multiply(V, V, out=U)
        fnp.multiply(U, V, out=U)
        fnp.multiply(U, P_st[:ko], out=U)
        return fnp.einsum("kij,kj->i", U, fnp.stack(g3_list[:ko], axis=0))

    def _k34_step(self, li, D3, g4, var, mu, WW, w32, K4p, t_v, pre, Anb, w1p, k4_prev, star=None,
                  perm=None):
        """features5 of optb/k4feat.py -> MLP -> (D3, kappa4 diagonal). WW = (w32 * w32).T."""
        p = self._k34
        f32 = fnp.float32
        n = var.shape[0]
        ap, vp, D3p = pre
        if perm is not None:
            # cn: the newest source's columns (Anb, birth neurons) are stored live-first; put the
            # birth-layer vectors and weight rows in the same order (the features sum over them)
            ap, vp, D3p = fnp.take(ap, perm), fnp.take(vp, perm), fnp.take(D3p, perm)
            w1p, k4_prev = fnp.take(w1p, perm), fnp.take(k4_prev, perm)
            w32 = fnp.take(w32, perm, axis=0)
        s1 = fnp.sqrt(var)
        s3 = var * s1
        s4 = var * var
        alpha = mu / s1
        own4 = ((WW * WW) @ K4p) / s4
        ph = flops.stats.norm.pdf(ap).astype(f32)
        Ph = flops.stats.norm.cdf(ap).astype(f32)
        sp = fnp.sqrt(vp)
        ivp = 1.0 / vp
        v = Anb.T + fnp.reshape(w1p * vp, (-1, 1)) * w32       # v_ai ~ Cov(z'_a, z_i)
        wv2 = (w32 * v) * v
        wv3 = wv2 * v
        q3 = ph / sp
        S1 = ((-ap * ph * ivp) @ wv3) / s4
        M1 = ((Ph * k4_prev * (ivp * ivp * ivp)) @ wv3) / s4
        T1 = ((q3 * D3p * (ivp * ivp) * 3.0) @ wv3) / s4
        S3 = (q3 @ wv2) / s3
        M3 = ((Ph * D3p * (ivp * ivp)) @ wv2) / s3
        g3 = D3 / s3
        g4c = g4 / s4
        cols = [alpha, g3, g4c, own4, t_v / var, fnp.full(n, li / 15.0, dtype=f32), S1, M1, T1, S3, M3]
        if star is not None:
            cols.append(star / s4)
        F = fnp.stack(cols, axis=1)
        x = (F - p["mu"]) / p["sd"]
        h = x @ p["W0"] + p["b0"]
        h = h * flops.stats.norm.cdf(h).astype(f32)               # exact GELU
        h = h @ p["W1"] + p["b1"]
        h = h * flops.stats.norm.cdf(h).astype(f32)
        o = h @ p["W2"] + p["b2"]
        y3 = g3 + o[:, 1] * 0.1
        y4 = g4c + o[:, 0]
        if self._mp is not None and li >= MP_LI_MIN:      # hooked layer: _mp_step (or the caller) forms D3, g4
            if self._mpones is None or self._mpones.shape[0] != y4.shape[0]:
                self._mpones = fnp.ones((y4.shape[0],), dtype=fnp.float32)
            self._mpx = (fnp.concatenate([F, fnp.stack([y4, y3, self._mpones], axis=1)], axis=1), s1, s3, s4)
            return None, None
        return y3 * s3, y4 * s4

    def _mlp3(self, x, pre):
        """optg: Linear-GELU-Linear-GELU-Linear with the weights of mp.npz prefix `pre`."""
        p, f32 = self._mp, fnp.float32
        h = x @ p[pre + "_W0"]                                  # bias = row 14 (x[:, 14] = 1)
        h = h * flops.stats.norm.cdf(h)                         # float64 from here (cdf promotes)
        h = h @ p[pre + "_W1"] + p[pre + "_b1"]
        h = h * flops.stats.norm.cdf(h)
        return (h @ p[pre + "_W2"] + p[pre + "_b2"]).astype(f32)

    def _mp_step(self, C_off, D21, wk431):
        """optg (G2 low-residual layout): message-passing residual on (D3, kappa4 diagonal) and on wk431 (optg/screen.py
        NodeMP rounds=2, PairMP full). One (13, n, n) buffer Q: 0 ones, 1 chain, 2 d21 d21^T, 3 rho, 4 rho^2, 5 d21,
        6 d21^T, 7 rho d21, 8 rho d21^T, 9 rho^3, 10 |rho|, 11 d21^2, 12 d21^T^2; node edges = Q[3:13], pair basis = Q[0:9]."""
        f32 = fnp.float32
        x, s1, s3, s4 = self._mpx
        n = x.shape[0]
        if self._mpbuf is None or self._mpbuf[0].shape[1] != n:
            Q = fnp.empty((13, n, n), dtype=f32)
            Ab = fnp.empty((10, n, n), dtype=f32)
            LR = fnp.empty((9, n, n), dtype=f32)
            self._mpbuf = (Q, Ab, LR, [Q[c] for c in range(13)], Q[3:13], Q[0:9], Q[5:7], Q[7:9], Q[11:13])
            fnp.copyto(self._mpbuf[3][0], fnp.ones((n, n), dtype=f32))
        Q, Ab, LR, q, E, Bp, Qd, Qrd, Qd2 = self._mpbuf
        is1 = 1.0 / s1
        is2 = is1 * is1
        fnp.multiply(C_off, fnp.outer(is1, is1), out=q[3])
        fnp.multiply(D21, fnp.outer(is2, is1), out=q[5])
        fnp.copyto(q[6], q[5].T)
        fnp.multiply(q[3], q[3], out=q[4])
        fnp.multiply(q[3], Qd, out=Qrd)
        fnp.multiply(q[4], q[3], out=q[9])
        fnp.abs(q[3], out=q[10])
        fnp.multiply(Qd, Qd, out=Qd2)
        fnp.multiply(q[5], q[6], out=q[2])
        fnp.multiply(wk431, fnp.outer(is1, is2 * is1), out=q[1])
        fnp.abs(E, out=Ab)
        inv = fnp.reshape(float(n) / fnp.sum(fnp.sum(Ab, axis=2), axis=1), (1, 10, 1))   # no tuple axis: the remote codec sends it as a list
        a = self._mlp3(x, "A")
        m1 = fnp.reshape(fnp.einsum("cij,jd->icd", E, a[:, :16]) * inv, (n, 160))
        psi2 = self._mlp3(fnp.concatenate([x, m1], axis=1), "B")
        m2 = fnp.reshape(fnp.einsum("cij,jd->icd", E, psi2) * inv, (n, 160))
        o = self._mlp3(fnp.concatenate([x, m1, m2], axis=1), "O") + x[:, 12:14]
        D3 = o[:, 1] * s3
        g4 = o[:, 0] * s4
        fnp.einsum("icr,jcr->cij", fnp.reshape(a[:, 16:52], (n, 9, 4)), fnp.reshape(a[:, 52:88], (n, 9, 4)), out=LR)
        Dl = fnp.einsum("cij,cij->ij", LR, Bp)
        fnp.fill_diagonal(Dl, 0.0)
        return D3, g4, wk431 + Dl * fnp.outer(s1, s3)

    # ------------------------------------------------------------------
    def _dslices(self, A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,
                 c1_list, c2_list, y_list, n, bufs, rres, rfb, Zf_st, R1T_st, R2T_st,
                 need_d21=True, ka=0, FAo=None, FPo=None, Qc=None,
                 kb=0, FA2=None, FP2=None, U2=None, yw=None, w1t=None, msl=None):
        """(3,) and (2,1) dslices with the M leg expressed through A, P and the thin
        residual leg: M = P*diag(s) + 3*A*diag(e) + Z L^T (the factor 3 of the
        residual is folded into Z at birth). Two dense contractions (right factors
        A and P) + O(n^2 r) thin terms.
          D21 = 2H + T2 + T3/3 + 2T4/3,  H=(A*P*w2)A^T, T2=(A*A*w2)P^T,
                T3=(P*P)M^T, T4=(M*P)P^T
          D3  = 3 sum_j w2_j A_ij^2 P_ij + sum_j M_ij P_ij^2
        Every (k,n,n) intermediate is written with out= into pooled buffers
        (F53: fresh result buffers dominate residual time).
        need_d21=False (V19, final layer): D3 only -- no left factors, no dense
        contractions, no thin right-factor terms; returns D21 = None.
        """
        k = len(w2b_list)
        W2B = fnp.reshape(fnp.stack(w2b_list, axis=0), (k, 1, n))
        Sb = fnp.reshape(fnp.stack(s_list, axis=0), (k, 1, n))
        Eb = fnp.reshape(fnp.stack(e_list, axis=0), (k, 1, n))
        AP = fnp.multiply(A_st, P_st, out=bufs["ap"][:k])
        PP = fnp.multiply(P_st, P_st, out=bufs["pp"][:k])
        T = bufs["t"][:k]
        # M*P = PP*s + 3 AP*e + (Z L^T)*P
        MP = fnp.einsum("kiq,kjq->kij", Z_st, L_st, out=bufs["mp"][:k])
        fnp.multiply(MP, P_st, out=MP)
        fnp.multiply(PP, Sb, out=T)
        fnp.add(MP, T, out=MP)
        fnp.multiply(AP, Eb * 3.0, out=T)
        fnp.add(MP, T, out=MP)
        if need_d21:
            # leftA = 2 AP*w2 + PP*e   (right factor A)
            LA = fnp.multiply(AP, W2B * 2.0, out=bufs["la"][:k])
            fnp.multiply(PP, Eb, out=T)
            fnp.add(LA, T, out=LA)
            # leftP = A*A*w2 + PP*(s/3) + (2/3) M*P   (right factor P)
            LP = fnp.multiply(A_st, A_st, out=bufs["lp"][:k])
            fnp.multiply(LP, W2B, out=LP)
            fnp.multiply(PP, Sb * (1.0 / 3.0), out=T)
            fnp.add(LP, T, out=LP)
            fnp.multiply(MP, 2.0 / 3.0, out=T)
            fnp.add(LP, T, out=LP)
        fnp.multiply(P_st, W2B, out=T)
        D3 = (fnp.einsum("kij,kij,kij->i", A_st, A_st, T) * 3.0
              + fnp.einsum("kij,kij->i", MP, P_st))
        if rfb > 0:
            # V18 (F69): B1 pair with X1 = 3A + Xt, Y1 = A*w2 + Yt; Xt = F1 R1, Yt = F2 R2
            # (F1/F2 = transported thin Z columns, R1T/R2T static (k, n, rfb)).
            F1 = Zf_st[:, :, :rfb]
            F2 = Zf_st[:, :, rfb:]
            Xt = fnp.einsum("kiq,kjq->kij", F1, R1T_st, out=bufs["xt"][:k])
            Yt = fnp.einsum("kiq,kjq->kij", F2, R2T_st, out=bufs["yt"][:k])
            U = bufs["u"][:k]
            # LPadd = A*Yt + (1/3) Xt*A*w2 + (1/3) Xt*Yt  (in T);  D3 += 3 rowsum(P*LPadd)
            fnp.multiply(A_st, Yt, out=T)
            fnp.multiply(Xt, A_st, out=U)
            fnp.multiply(U, W2B * (1.0 / 3.0), out=U)
            fnp.add(T, U, out=T)
            fnp.multiply(Xt, Yt, out=U)
            fnp.multiply(U, 1.0 / 3.0, out=U)
            fnp.add(T, U, out=T)
            D3 = D3 + fnp.einsum("kij,kij->i", P_st, T) * 3.0
            if need_d21:
                fnp.add(LP, T, out=LP)
                # LA += (1/3) Xt*P*w2 + P*Yt
                fnp.multiply(Xt, P_st, out=T)
                fnp.multiply(T, W2B * (1.0 / 3.0), out=T)
                fnp.add(LA, T, out=LA)
                fnp.multiply(P_st, Yt, out=U)
                fnp.add(LA, U, out=LA)
                # thin right factors:  GY = AP + (1/3) Xt*P      -> D21 += (GY R2^T) F2^T
                #                      GX = (1/3)(AP*w2 + P*Yt)  -> D21 += (GX R1^T) F1^T
                fnp.multiply(Xt, P_st, out=T)
                fnp.multiply(T, 1.0 / 3.0, out=T)
                fnp.add(T, AP, out=T)                 # GY
                GYR = fnp.einsum("kij,kjq->kiq", T, R2T_st)
                D21_fb = fnp.einsum("kiq,kcq->ic", GYR, F2)
                fnp.multiply(AP, W2B, out=T)
                fnp.add(T, U, out=T)                  # AP*w2 + P*Yt   (U holds P*Yt)
                fnp.multiply(T, 1.0 / 3.0, out=T)     # GX
                GXR = fnp.einsum("kij,kjq->kiq", T, R1T_st)
                D21_fb = D21_fb + fnp.einsum("kiq,kcq->ic", GXR, F1)
            else:
                D21_fb = None
        else:
            D21_fb = None
        # F68 feed pair (X3 = A*c1 + P*c2 column scalings, Y3 = y 1^T):
        #   D21 += (1/3) R y^T + (2/3) (y (.) X3) P^T,  D3 += y * R,
        #   R = rowsum(X3*P) = (A*P) c1 + (P*P) c2; y is the TRANSPORTED Y3 row vector
        #   (last Z column).   MP/T are free scratch now.
        C1 = fnp.stack(c1_list, axis=0)                 # (k, n) static column scalings
        C2 = fnp.stack(c2_list, axis=0)
        Yk = Z_st[:, :, rres + 1]                        # (k, n) transported y = P(l) y
        R = fnp.einsum("kij,kj->ki", AP, C1) + fnp.einsum("kij,kj->ki", PP, C2)
        D3 = D3 + fnp.einsum("ki,ki->i", R, Yk)
        if not need_d21:
            return D3, None
        fnp.multiply(P_st, fnp.reshape(C2, (k, 1, n)), out=MP)
        fnp.multiply(A_st, fnp.reshape(C1, (k, 1, n)), out=T)
        fnp.add(T, MP, out=T)
        fnp.multiply(T, fnp.reshape(Yk * (2.0 / 3.0), (k, n, 1)), out=T)
        fnp.add(LP, T, out=LP)
        if ka > 0:
            # V21: old sources through the shared basis: [sum LA FAo^T + LP FPo^T] Qc^T
            # V24: tier-2 sources through U: [sum LA FA2^T + LP FP2^T] U2^T lifted first (U2 = sub-basis; U is the scratch buffer)
            inner = None
            if ka > kb and w1t is not None and w1t < n:
                inner = _hub2(LA[kb:ka, :, :w1t], FAo[:, :, :w1t], LP[kb:ka, :, :w1t],
                              FPo[:, :, :w1t], RLEV, T1B, RMB)
            elif ka > kb:
                inner = _hub2(LA[kb:ka], FAo, LP[kb:ka], FPo, RLEV, T1B, RMB)
            if kb > 0:
                inner2 = _hub2(LA[:kb], FA2, LP[:kb], FP2, RLEV, 8, T2MB)
                lift = inner2 @ U2.T
                inner = lift if inner is None else inner + lift
            D21 = _sd(inner, Qc.T, 4, 24)
            if ka < k and msl is not None:
                # cn: young columns beyond a source's live block are dead-at-birth zeros on both
                # factors; cn5: each batch-SB chunk on its own block (cn26: also when another
                # young source is full width -- exact)
                for c0 in range(ka, k, SBK):
                    c1 = min(k, c0 + SBK)
                    wc = max(msl[c0:c1])
                    if wc < n:
                        # cn51: LA and LP of the chunk in ONE batch-4 Strassen call when the depth-3 stacks
                        # (7^4, 4, n/16, wc/16) float32 stay under the grader's 134.9 MB per-array cap (wc <= 864)
                        D21 = D21 + _hub2(LA[c0:c1, :, :wc], A_st[c0:c1, :, :wc], LP[c0:c1, :, :wc],
                                          P_st[c0:c1, :, :wc], SLEV, bmax=(4 if wc <= 864 else SB), mb=24)
                    else:
                        D21 = D21 + _hub2(LA[c0:c1], A_st[c0:c1], LP[c0:c1], P_st[c0:c1], SLEV)
            elif ka < k:
                D21 = D21 + _hub2(LA[ka:], A_st[ka:], LP[ka:], P_st[ka:], SLEV)
            D21 = D21 + fnp.einsum("ki,kc->ic", R, Yk) * (1.0 / 3.0)
        else:
            D21 = (_hub2(LA, A_st, LP, P_st, SLEV)
                   + fnp.einsum("ki,kc->ic", R, Yk) * (1.0 / 3.0))
        PPL = fnp.einsum("kij,kjq->kiq", PP, L_st)
        D21 = D21 + fnp.einsum("kiq,kcq->ic", PPL, Z_st) * (1.0 / 3.0)
        if D21_fb is not None:
            D21 = D21 + D21_fb
        return D3, _zero_diag(D21)


if __name__ == "__main__":
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=1024, depth=16, seed=0)
    compare_against_monte_carlo(Estimator(), mlp, seed=0)
