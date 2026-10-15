# Evaluation of notes 13-15 (Option B improvement brainstorms, leader reconstruction)

Date: 2026-10-15. Measured on this repository's 4-core container, 11 local validation
networks (1e6-sample Monte Carlo targets, noise-corrected final-layer MSE), flopscope 0.12.
"Score" below is the per-network mean of MSE x C/B, the leaderboard quantity.

## What our submission actually computes

The shipped estimator is the factorized K=3 cumulant chain (504aldo V29 lineage) with the
team's trained v5b operator (`k34.npz` f_self + `mp.npz` message passing) as the ReLU-step
correction. It does **not** run the literal Option B of the Oct 14 PDF: there is no
128-wide pair MLP over all 523,776 pairs and no quartic Q contraction. That matters for
notes 13 and 14, whose headline savings target exactly those two components.

| Note item | Applies to our code? | Status / measured effect |
|---|---|---|
| 13.1/14.I03 shared contractions | partly | already shared (one transport family per layer; LA/LP legs reused by D3 and D21) |
| 13.2/14.I02 final-layer pruning | yes | already in (last layer: diag(W C W^T) only, no D21, no birth) |
| 13.3/14.I01 exact first layer | yes | already exact (the chain starts from the exact Gaussian input) |
| 13.4/14.I04 float32 | yes | already float32 everywhere |
| 13.5/14.I27 Strassen | yes | 5 levels on the n^3 families, chunked under the 100 MiB cap, fused last level |
| 13.6/14.I05-07 cheaper pair network | no | no pair MLP in our path (the v5b operator is per-neuron + 16-dim messages, ~0.005 B) |
| 13.9/14.I10-12 Q approximation | no | no Q in our path |
| 13.18/14.I17-18 rollout training | yes (accuracy) | `scripts/train.py --v5b` trains the v5b operator end to end; needs far more networks than our 64 |
| 15: Strassen min block / depth | yes | see table below: -3.6% C/B at v5b-level residual |
| 15: cheaper old-source tier | yes | knob sweeps below: both directions lose |
| 15: young transport "only what D3/D21 need" | no exact form found | D3 and D21 read the full dense legs through Hadamard products; no exact shortcut |

So the 10-50% figures do not transfer: they are reductions of a pair-network + Q bill our
estimator never pays. Note 15's diagnosis does transfer: our raw MSE (1.35e-8) is already
inside the leaders' band (1.15-1.65e-8), and cost is the whole lever.

## Measured cost/accuracy knobs

Source-memory knobs (our build, shipped Strassen depths), 11 networks:

| change | MSE | C/B | score | vs shipped |
|---|---|---|---|---|
| shipped (AGE_OLD 4, R_OLD 384, R_OLD2 224) | 1.345e-8 | 0.2663 | 3.58e-9 | -- |
| AGE_OLD 3 (sources join the shared basis one layer earlier) | 1.578e-8 | 0.2563 | 4.05e-9 | worse (+13%) |
| R_OLD 256, R_OLD2 160 | 2.038e-8 | 0.2310 | 4.72e-9 | worse (+32%) |

The tiers sit at a rank/age cliff (504aldo reports the same): a cheaper old tier needs a new
mechanism, not a smaller one.

Strassen depth (residual time is the binding cap; graded v5b runs at 1.21-1.29 s residual
and 128-150 s wall on this container, single BLAS thread, and passes the grader):

| build | C/B | residual | wall |
|---|---|---|---|
| ours, shipped (5/4/3/3/3/3, min block 32) | 0.2665 | 0.91-0.95 s | 74-80 s |
| ours, `candidates/configs/deep16.json` (5/5/4/4/4/4, min block 16) | 0.257 | 1.21 s | 103 s |
| ours, family 6 | 0.2545 | 1.45 s | 133 s |
| team v5b as graded | 0.2517 | 1.21-1.29 s | 128-150 s |
| `candidates/v5b_small` (v5b, RLEV 4, min blocks 24/28) | 0.2452 | 1.36-1.42 s | 141-164 s |
| `candidates/v5b_rfhub` (+ joiner hub on Strassen) | 0.2424 | 1.42-1.47 s | 143-166 s |

On MSE x C/B the team's v5b code stays ~3% ahead of our port at equal residual, so the best
near-term leaderboard move is v5b_rfhub (-3.7% cost at unchanged arithmetic), provided the
grader absorbs +15-20% residual/wall over the graded v5b. Only a graded run can tell.

## Why 0.10-0.13 B needs a new idea

Per layer, in dense n^3 units before Strassen: young sources ~18 (2 transports + 2 hub
products per source, ~4 young sources, + newborn + covariance), old tiers ~12 (formings of
the dense legs + inner contractions), joiner ~2, covariance sandwich 1. The leaders'
September bills equal V29 minus the old tier, i.e. their old content is nearly free, and
they have since cut 20-30% more. In our formulation the old tier needs its dense legs
only to form the Hadamard products LA = 2 A*P*w2 + P*P*e + ..., LP = A*A*w2 + ... that D3
and D21 read; any route to the leaders' bill has to produce those slice contributions
without materializing the legs (e.g. a source representation whose Hadamard structure
factorizes, or a regenerated rather than transported old-source contribution, as V29 does
for kappa4). None of the notes supplies such a construction, and 504aldo's measured dead
ends (dropping old sources, Tucker on all legs, per-source rank ladders) rule out the
obvious ones. This is the research direction worth the team's remaining time; everything
cheaper is a few percent.
