# Carried created-error corrector (cn62)

The idea is from forum 18254 (Susupro, #336537: raw x0.887 on the grader). Each layer's chain error splits exactly
into a part propagated through the linearised map J_l(e) = Phi_l * (W_l^T e) and a part created at the layer
(c_l). A per-neuron model predicts delta_l ~ -c_l from that layer's chain statistics and the previous layer's
statistics carried through W and W*W. The correction rides s_l = J_l(s_{l-1}) + delta_l and is added to the final
prediction; the chain itself is unchanged.

1. `scripts/hf/runfast_hf_ldump.py` (fast mode, scratch estimator with the per-layer recorder) dumps per-layer
   statistics for public networks.
2. `prep.py <tag> <ids>` builds X (16, n, 24), the created errors C and the errors E from the 1e9-sample all-layer
   means.
3. `fitmlp.py <tag> <train> <test> [epochs] [hidden]` trains a 3x64 GELU MLP on -C_l (layer weights rho^(15-l))
   and scores the test networks by propagation. `fitlin.py` holds the linear baselines.
4. `export.py` writes `corr.npz` for `candidates/v5b_cn62`. `torch_pred.py` gives the torch-side corrected
   prediction for a parity check.

First signals (mini, 37 train / 38 test networks): per-layer ridge -0.3%; joint final-objective ridge overfits;
MLP -1.01% (28/38 better). cn62 in chain vs the torch pipeline: max difference 2.3e-6 on m50.
