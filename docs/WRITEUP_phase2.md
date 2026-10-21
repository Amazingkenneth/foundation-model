# A cost-engineered K=3 cumulant chain with a learned ReLU step and a carried corrector

*ARC White-Box Estimation Challenge 2026, Phase 2 (width 1024, depth 16). Algorithmic write-up, draft of
Oct 21. Team: **[team name / handles: to fill in]**.*

**Nominated builds.** `candidates/v5b_cn71` (submission 337265, score **3.3183e-9**: raw final-layer MSE
1.7297e-8, C/B 0.1918) and its robust twin `candidates/v5b_cn71r` (337275, 3.5407e-9: same estimator, young
families one Strassen level shallower for wall-clock margin). Both graded clean on all networks.

**What this write-up claims as ours.** We started from our team's own graded v5b submission (score 4.4e-9, raw
1.75e-8, C/B 0.2518). v5b was 504aldo's MIT-licensed factorized K=3 chain plus the team's learned ReLU-step
operator. Section 2 lists everything we built on. Our contributions are:

1. **Cost engineering of the chain.** C/B 0.2518 -> 0.191 (-24%) with raw MSE held at 1.73-1.82e-8
   (section 3), over more than two dozen graded builds.
2. **A learned carried corrector**, following the design Susupro published in forum post 18254 (section 4):
   - our implementation and training on 1,000 public networks;
   - a gain-weighted per-layer loss built from measured propagation gains;
   - correlation-structure inputs and a small, cheap model.

   It takes 3.9% off graded raw MSE.
3. **Measurement studies with numbers**, useful beyond our entry (section 5):
   - the leaders' likely construction (a carried kappa4 core with coherent scale-mode extraction, after
     il.miscusi's 18248), reproduced in an untruncated chain (0.70 of our best build's error) and ported into a
     production chain, with a layer-by-layer diagnosis of why the port loses its gain there;
   - a structured-sampling (cubature) baseline at 1024 x 16;
   - an error anatomy of the chain;
   - a table of measured dead ends.

---

## 1. Task, score and limits

The networks are bias-free ReLU MLPs, `h_{l+1} = relu(h_l @ W_l)`, with n = 1024, L = 16, x ~ N(0, I) and
He-initialized weights. The estimator returns per-layer means; only the final layer is scored:
`score = mean over networks of MSE_final x max(0.1, C/B)`, with B = 2^41 FLOPs metered by flopscope (0.12.3).
One 1024^3 product is 2^31 FLOPs, so B is 1,024 dense products: about 64 per layer at C/B = 1, about 12 at ours.

Per-network caps on the grader: 120 s wall and 0.4 s client-side Python ("residual") time, a ~134.9 MB limit
per server-side array, and finite outputs on eight small smoke shapes. A network over a cap scores zero, which
costs about 0.009 of mean score (ours is 3e-9), so robustness constrained every design choice.

## 2. What we built on (not claimed)

| piece | source |
|---|---|
| factorized K=3 cumulant propagation; term tables; Wick coefficients | ARC's reference (Wu, Lecomte, Winer, Robinson, Hilton, Christiano, arXiv:2605.05179; `mlp_cumulant_propagation`, MIT) |
| the chain: V16b source restructure (A, P legs; M leg never transported), V17 memoryless kappa4 regeneration `G = diag(g) + lambda C_off` with its lambda table, V18 D21 feedback thin legs, V19 mean-only final layer, V21/V24 age-gated shared bases for old sources, V25 adaptive lambda; Strassen-Winograd products (V26-V29) | 504aldo, `whest-p2-cumulant-k3` (MIT; notice kept in `LICENSE_504aldo` of every shipped folder) |
| learned ReLU step: a per-neuron kappa3/kappa4 correction (12 features -> 64 -> 64 -> 2) at every layer, and message passing on the (3,1) slice from layer 8 (`k34.npz`, `mp.npz`) | our team's earlier v5b submission (`learned_relu_step_v5b_mp_l8_strassen5`; see the repository README), trained before this phase |
| carried corrector design: a per-layer created-error target propagated through the ReLU slopes and W, with a state carried through W and W∘W | Susupro, forum 18254 |
| the scale-mode extraction used in our study of section 5.1 | il.miscusi, forum 18248 |
| the SSC-type (2,2) structure behind one closure knob | Puffi, forum 18175 (ported into a V29 chain by Susupro, 18254) |
| the marginal-Edgeworth consistency rule (k = 2 marginal terms kept only in the dead tail, alpha <= -2) | Artyom_Mazur, forum 18249 |
| the age-rank law that sized our shared-basis ranks | 504aldo (findings F72/F73) |

The public labels we validate against are the dataset's own per-layer means (1e9 samples) of the 100 mini and
1,000 full public networks. We read them through HTTP range requests and regenerated the weights from
`mlp_seed`, exactly as `whest dataset bake` does.

## 3. Cost engineering: C/B 0.2518 -> 0.191

Where v5b spent its budget:
- about 76% went to the young kappa3 sources: transporting their dense legs and contracting them into the (2,1)
  slice D21 ("hub");
- the rest went to the shared-basis tiers of old sources, the covariance sandwich, and the term program.

Each row is a graded build (cumulative). "exact" means bitwise or within float32 regrouping; "trade" means the
step changes the chain's accuracy, so its graded raw MSE is the result.

| build | change | kind | raw MSE | C/B | score |
|---|---|---|---|---|---|
| v5b (start) | | | 1.75e-8 | 0.2518 | 4.4e-9 |
| rg | Strassen on the joiner hub and on the n x 384 Gram / basis products | exact | 1.7534e-8 | 0.2378 | 4.171e-9 |
| cn2, cn3 | **live-column chain**: a newborn source's columns for neurons dead at birth (Phi < 0.01) are zeroed and stored last; transports and hubs run on the live block only; the next layer's transports contract over live rows only | trade (cut columns) | 1.7637e-8 | 0.2254 | 3.976e-9 |
| cn6 | warm-started range finders for the shared basis (the previous basis, transported, as the sketch); per-chunk young widths; covariance sandwich at 4 Strassen levels | trade (better) | 1.7575e-8 | 0.2221 | 3.904e-9 |
| cn7, cn8 | age gates 5 / 8 at the ranks 504aldo's age-rank law gives (320 / 192); oversampled D21-feedback range finder (Halko-Martinsson-Tropp) | trade | 1.7365e-8 | 0.2195 | 3.811e-9 |
| cn9 | tier-1 basis rebuilt only on alternate joins (kept and re-orthonormalized in between); tier 2 from age 9 at rank 160 | trade | 1.7391e-8 | 0.2156 | 3.749e-9 |
| cn11a | rows of neurons with Phi < 0.01 dropped from the next transports (in steps of 32, at least 768 kept) | trade | 1.7367e-8 | 0.2122 | 3.686e-9 |
| cn21 | D21 feedback thin legs at rank 2 | trade | 1.7950e-8 | 0.1995 | 3.582e-9 |
| cn37 | S21 residual leg at rank 4 | trade | 1.8235e-8 | 0.1934 | 3.526e-9 |
| cn49b | Artyom_Mazur's marginal-Edgeworth tail rule (18249) + feedback oversampling 8 | trade (better) | 1.8204e-8 | 0.1929 | 3.512e-9 |
| cn53b | Strassen-Winograd kernel with a slab buffer pool and no billed reshapes | exact | 1.8205e-8 | 0.1915 | 3.486e-9 |
| cn54b | nonlinear term program accumulated elementwise (3 n^2 per term instead of a 4-operand einsum) | exact | 1.8202e-8 | 0.1898 | 3.454e-9 |
| cn61 | residual rank 6 and a late dense window (from layer 14 the age gate is one older) | trade (better) | 1.8012e-8 | 0.1910 | 3.440e-9 |

Over these steps raw MSE stayed within 1.73-1.82e-8 while C/B fell by 24%. Alongside them we cut the client round
trips: lazy views, persistent per-layer buffers, and batched PK2 accumulation (43.1k round trips per network).
That keeps the residual under the 0.4 s cap at 5 Strassen levels.

Two lessons we paid for on the grader:
- **Level-6 Strassen failed four times** (wall), though it bills 1% less.
- **Wall-clock margin is thin.** On loaded days, identical code lost 40-70% of its networks.

We therefore always graded a twin with the young families at 4 levels. It costs +6.8% C but runs 24% faster in
process (81.6 s vs 106.9 s), and it graded clean every time. It is our second nomination.

**Where the remaining error is.** All measured on the 100 mini networks with exact labels:
- Turning off every truncation (shared bases, rank cuts, masks) lowers the MSE by only 4.5%. The rest is closure
  error, not truncation.
- On the cheap side every knob costs 1.5-3% MSE per 1% C; on the expensive side every knob buys only ~0.6% MSE
  per 1% C.
- The final layer creates about 22% of the final MSE. Created error grows ~9x from layer 2 to 15, and the
  measured propagation gain of a created error to the output rises from 0.031 (layer 0) to 1.0 (layer 15).
- Bias is negligible: per-network oracle fits of any final-layer affine fix recover only 0.4-1.6%.

## 4. A learned carried corrector (design after 18254; implementation, objective and inputs ours)

**Target.** Let `e_l` be the chain's mean error at layer l. The *created* error is
`c_l = e_l - Phi_l o (W_l e_{l-1})`, the part not explained by propagating the previous error through the ReLU
slopes `Phi_l` and the weights. A small MLP predicts `delta_l ~ -c_l` from the chain's own per-neuron statistics.
The predictions are propagated with the chain's linearized map and added once, at the output:

```
s_0 = delta_0,   s_l = Phi_l o (W_l s_{l-1}) + delta_l,   prediction_L = chain_L + s_L
```

So the chain itself is untouched, and the correction costs O(n^2) per layer.

**Model.**
- One shared MLP for all layers: ReLU, 3 hidden layers of 64, output `1 + 16 + 8`.
- Per-neuron inputs (68 dims):
  - 20 local chain statistics: mean, variance, alpha, phi, Phi, the post-ReLU moments pk1..pk4, the D3 and kappa4
    diagonals, row and column sums of D21, and row sums of C_off^2, |C_off| and rho^2;
  - 5 previous-layer statistics passed through W and 3 through W∘W;
  - a layer one-hot;
  - a 16-dimensional learned state carried through W, `(W s_{l-1}) / sqrt(sum W^2 / n)`;
  - an 8-dimensional state carried through W∘W, normalized the same way.
- Trained through depth on whole networks (weights regenerated per network), Adam, 16 epochs, on the 1,000
  public full-split networks. Held out: the mini networks.

**Objective (ours).** The loss is the per-layer MSE of `delta_l`, weighted by the measured propagation gain of
layer l to the output. Alternatives we measured, all worse:
- an end-to-end final-error term: -1.05% vs -2.62% at 200 networks (one noisy scalar per network);
- decay weights `0.75^(15-l)`: -1.94% vs -2.44%;
- a pooled final-MSE fine-tune (forum 18255's terminal objective): training loss falls, held-out worsens;
- a 30/70 per-layer/transported mixture (mliston 18170): -1.65% vs -3.73%.

**Measured history.** Held-out mini dMSE against the uncorrected chain; graded raw MSE against the uncorrected
cn61 (or cn61r for twins):

| step | held-out | graded |
|---|---|---|
| per-layer MLP, no state (974 networks) | -2.23% | cn62r -1.51% (an earlier 297-network model) |
| + 32-d state through W | -3.92% | cn63r -2.83% |
| + W∘W state (16), ReLU | -4.60% | cn64 -3.78% |
| + 16 epochs | -4.72% | cn66 -3.90% |
| + correlation-structure inputs (rho^2, C_off row sums) | -4.90% | cn68 -4.15% |
| **small model (H 128 -> 64, S 32 -> 16, S2 16 -> 8)** | -4.76% | **cn69 -3.90%, score -0.20% (corrector bill -0.46%)** |

The local gain transfers to the grader at 0.73-0.86x. Parity between the in-chain corrector and the torch
training pipeline is 2.3e-6 max abs.

**What did not help:**
- capacity: H 192, S 64, S2 32, 24 epochs;
- a backward state;
- extra inputs (rho^4, signed rho, max rho^2, kappa4-slice mass, participation ratios);
- root transforms of heavy-tailed inputs, EMA, per-layer standardization or output scales, tanh states;
- incoming-correction inputs;
- training on all 1,098 labelled networks (graded +0.10%, noise);
- network-context inputs (per-layer means over neurons), +0.19pp over 3 seeds at 200 networks.

At 1,000 networks the corrector is saturated for this input set.

**End-to-end closure-knob refit (cn71).** We fit 12 multiplicative knobs on the consumed slices (D3, the kappa4
diagonal, the (3,1) and (2,1) slices, the kappa4 -> kappa3 feed, an SSC-type (2,2) term), with the corrector in
place:
- method: a finite-difference Jacobian of the final means on 24 mini networks, then one ridge Gauss-Newton step;
- local result: all 100 mini -0.50%, the 50 networks outside the fit -0.59%;
- graded: -0.07% raw (transfer ~0.15x);
- a second round over 17 knobs found every knob at its optimum (<= 0.03% each).

## 5. Studies

### 5.1 The leaders' construction, reproduced and taken apart

The top six teams sit at raw 1.10-1.36e-8 for C/B 0.107-0.128. That is 0.64-0.79 of our error at 0.55-0.67 of
our bill. il.miscusi (18248) reported that ARC's AUGMENT chain (kappa4 carried as an r = 1 matrix core
`G <- W G W^T`) with **coherent scale-mode extraction** reaches 0.58 of V29's error. Because the networks are
bias-free, a common random scale of the activations is invisible to every mean. Extraction fits its strength eps
per layer from the (2,1) slice and removes it consistently:
- from C: `(C - eps m m^T)/(1+eps)`;
- from kappa3: `-2 eps Sym(C x m)`;
- from kappa4: `G -= H[4 eps Pair3(C x C)]`.

We asked two questions: does it beat our build, and can our engine carry it?

**Untruncated reference.** A helper run patched 504aldo's float64 lean chain to parity with ARC's current factored
AUGMENT rule (1e-14 at n = 48) and added the extraction exactly. On 8 public mini networks:

| lean chain | final MSE | vs cn71 (corrector included) |
|---|---|---|
| AUGMENT, no extraction | 1.81e-8 | 0.98 |
| **+ extraction** | **1.29e-8** | **0.70 (8/8 better)** |
| + K211 hub content from the newborn source only | 1.33e-8 (3 nets) | 0.79 |
| without the K211 content | 2.45e-8 | 1.40 |
| with the smaller budget-2 term table (`augment=False`) | 3.49e-8 | 1.89 |
| keeping only the 3 / 5 newest kappa3 sources | 38e-8 / 16e-8 | diverges |

Extraction's gain grows with depth (B/A 0.95 at layer 1, 0.71 at layer 15). The per-layer eps we fit matches
18248's medians: 7.9e-4 at layer 0, 1.5e-4 at layer 14.

**In our production engine** (`scripts/tchain/estimator_c80.py`, flags off = cn71's chain), we added:
- ARC's current AUGMENT term table, including the new (3,1) output;
- the carried core with its r = 1 harmonic production;
- the capped (2,1,1) production and the newborn's hub contraction;
- the extraction, with its kappa3 part as thin legs on the newborn source.

What we found:
- **Layer 0 matches the lean chain to float32 precision.** The term table, K(4), K(2,2), K(3,1), the capped K211
  production and G all agree to ~5e-7.
- **From layer 1 the core's off-diagonal drifts** (13%, then 35%), while its slice inputs still agree to 1e-3. The
  difference sits in what the newborn sources carry.
- **Without extraction the carried core diverges** (8x), as 18248 found in e30_a's engine.
- **Our learned k34 / MP operators break on the extracted state** (x50). They were trained on the lambda chain.
- **Walking the lean chain toward our engine** (window-0 K211, extraction 0.5, m70):

  | change | error |
  |---|---|
  | (start) | 1.33e-8 |
  | + our older birth rule | +7% |
  | + dropped B2 hub pair / WK22 term | +2% |
  | **+ our rank-2 D21 feedback leg** | **+17%** |
  | + our rank-6 residual leg | +2% |
  | our tiers and dead-neuron masks | <= 2% |

  The rank-2 feedback leg, harmless in the lambda chain, is the main loss for a carried-core chain.
- **With feedback rank 64 and residual rank 32:** -9% vs our chain on m70, but **+1.7% on 8 networks**. It bills
  C/B **0.40** (0.33 at our ranks; cn71 0.19). An optimized core loop would still be ~0.25-0.28.
- **Making old kappa3 cheaper is not available:** an earlier join into the shared basis hurts the carried chain
  more than ours (+51% vs +38%).

Our reading: the construction is worth -20 to -30% in an untruncated chain, but it needs every old kappa3 source
and full-rank feedback, which per-source hub representations cannot afford at this width. The leaders must carry
kappa3 in a cheaper representation than per-source hubs; none has been published.
(`docs/leaders_2026-10-20.md`, `scripts/tchain/`.)

### 5.2 Structured sampling is far from enough at 1024 x 16

Phase 1's winner used Kerdock / MUB spherical cubature, and 8,192 forward passes cost exactly the leaders' C/B of
0.12. Final-layer MSE on mini m70 (2 seeds):

| points | iid Gaussian | iid spherical | random orthonormal frames, antipodal | Hadamard x random signs |
|---|---|---|---|---|
| 2,048 | 2.99e-5 | 3.03e-5 | 1.95e-5 | 1.83e-5 |
| 8,192 | 7.32e-6 | 7.26e-6 | 4.51e-6 | 4.57e-6 |

Integrating degrees <= 3 exactly buys only 1.6x: the variance of a depth-16 output sits in high-degree harmonics.
At 4.5e-6 this is 400x short of the leaders.

### 5.3 Measured dead ends

dMSE on exact labels, dC on the 0.12.3 bill. Full table in `docs/ALGORITHM.md` section 6.

| idea | result |
|---|---|
| E1 kappa4 pendant edge (18252) | -1.1% raw chain-only, but +3.4% C in our D21 form (one more product per layer): net loss |
| C0^3 Mehler pair term | -0.2% +- 1.9 (null) |
| (3,1) two-term law (18246); per-neuron metric diag(W W^T) | absorbed by the learned operator |
| post-July term tables in a lambda-law chain | +5% (also seen by 18248) |
| tiers by the age-rank law (ages 3-5 at rank 640) | C/B 0.246 vs 0.191 |
| skipping the final layer's source work (D3 := 0 at the last layer) | -4.2% C, x11.7 MSE |
| RW_TAU 0.02, CM_TAU 0.02, AGE_OLD 4 / 6, R_OLD 288 / 352, R_OLD2 128, CN_RRES 2 / 8 | every one off the optimum (rates in section 3) |
| dead / saturated rows or dead columns of the D21 hub | +8% to +55% MSE: they carry real content |
| half-precision billing | float16 billed like float32 in 0.12.3 |
| final-layer ridge / per-network scale riders | null (bias negligible) |

## 6. Validation gate and reproducibility

The gate below ran on every build before submission. Tools are in `scripts/gate`, `scripts/hf` and
`scripts/corr2`; commands are in `docs/ALGORITHM.md` section 8.

1. Held-out accuracy of the corrector (torch).
2. Torch-vs-chain parity (<= ~3e-6 on m50).
3. In-chain fast mode on the 100 mini networks against exact labels, with bootstrap se and even/odd halves.
4. Real Strassen kernels on a public network.
5. The 0.12.3 bill and peak memory.
6. A local flopscope 0.12.3 client/server emulator (wall and residual), plus the 8 smoke shapes.
7. Client round-trip counts.

Every graded submission and its receipt is listed in `README.md`. The full experiment log is
`docs/final_phase_2026-10-17.md`.

## 7. Disclosure

**[Team: confirm or edit.]** The code, experiments and documents in this repository were produced by an AI coding
agent (Claude, Anthropic) in sessions directed by the team. The team set the direction and the priorities,
decided what to submit and nominate, and owns the result. Section 2 credits the methods and code we built on;
their licenses are kept in the shipped folders.
