# Pre-submission gate tools

These are the tools that gated cn62-cn69. Until Oct 20 they lived only under `/tmp/claude-0/`. The order and pass
criteria are in `docs/ALGORITHM.md` section 8.

## Environment the scripts assume

| path | contents | made by |
|---|---|---|
| `/tmp/claude-0/venv` | main venv: flopscope (in-process), torch, numpy, whestbench | session setup |
| `/tmp/claude-0/venv123` | in-process flopscope **0.12.3** (the grader's billing) | `pip install flopscope==0.12.3` |
| `/tmp/claude-0/emu123/{srv,cli,stub}` (`EMU_DIR`) | flopscope-server 0.12.3 and flopscope-client 0.12.3 venvs, plus a whestbench stub | `emu123.sh` on first run |
| `/tmp/claude-0/hfwork/{mini,full}.npz` | 1e9-sample per-layer means of the public v2-phase2 networks (100 mini + 1,000 full) | `scripts/hf/hf_labels.py` |
| `/tmp/claude-0/hfwork/regen.py` | weight regeneration from `mlp_seed` (same file as `scripts/hf/regen.py`) | copy |
| `/tmp/claude-0/acc/<tag>/pred_<key>.npy` | predictions written by the runners | the runners |

Network keys look like `m12` (mini `mlp_id` 12) or `f345` (full `mlp_id` 345).

## Tools

| tool | what it checks | usage |
|---|---|---|
| `../hf/runfast_hf.py`, `../hf/gohf.sh` | accuracy in **fast mode** (Strassen kernels patched to plain matmul; predictions match real Strassen up to float32 regrouping), parallel over N processes | `cd /tmp/claude-0/acc && ./gohf.sh <dir> <tag> m0,m1,... 4` (copy `gohf.sh` and `runfast_hf.py` there; `RUNNER=runfast_hf_ldump.py` makes dumps instead) |
| `../hf/eval_hf.py` | paired final-layer MSE against the 1e9 truth: pooled dMSE, bootstrap se, even/odd halves, better count | `eval_hf.py <base tag> <tag> ...`; base `ld61` = the uncorrected cn61 chain, `c68f1000` = cn68 |
| `runstr_hf.py` | the **real Strassen kernels** on public networks (the production path; fast mode bypasses it) | `OMP_NUM_THREADS=1 /tmp/claude-0/venv123/bin/python runstr_hf.py <dir> str_<tag> m0` (~2 min per network) |
| `runv.py` | **0.12.3 bill** (C/B), in-process wall, VmHWM for given `make_weights` seeds | `/tmp/claude-0/venv123/bin/python runv.py <dir> <tag> 1` |
| `prof_fn.py` | **cost breakdown**: every billed op attributed to the innermost `estimator.py` function and line | `/tmp/claude-0/venv123/bin/python -I prof_fn.py <dir> 1` |
| `emu123.sh` | the grader's **client/server** setup (server 0.12.3, client 0.12.3, every op an IPC round trip): wall, C/B and residual per MLP | `EMU_DIR=/tmp/claude-0/emu123 bash emu123.sh <repo-relative dir> 1` |
| `gate_emu.sh` (+ `drive_shapes.py`) | `emu123.sh` on one suite MLP, then the **8 smoke shapes** (widths 4..256) must be finite | `scripts/gate/gate_emu.sh candidates/v5b_cnXX tag` |
| `run_cnt.sh` (+ `drive_cnt.py`) | **client round trips per MLP by op** (the residual cap scales with them; cn68/cn69: 43,136 steady, 49,945 cold) | `scripts/gate/run_cnt.sh candidates/v5b_cnXX tag` |
| `poll.py` | grading status of **our own** submission ids | `AICROWD_API_KEY=... python poll.py 337211` |
| `e1_screen/` | the E1 kappa4 pendant-edge screen (forum 18252; closed for this chain, see ALGORITHM.md section 6) | below |

Corrector-specific gate steps (held-out eval, torch vs chain parity, in-chain mini 100) are in `../corr2/`
(`eval_rnn.py`, `gate_corr.sh`).

## E1 screen (`e1_screen/`)

1. Build the screening estimator: copy `candidates/v5b_cn68/{estimator.py,strassen4.py,k34.npz,mp.npz,LICENSE_504aldo}`
   to `/tmp/claude-0/sc/c70/` (no `corr.npz`: chain only), then `patch -p0 estimator.py < c70_e1.patch`. The patch
   reproduces the screened file byte for byte.
2. Knobs: `XE1` sets the scale (0 = off; the theoretical coefficient 12 is built in). `XE1_LMIN` is the first layer
   it applies to. `XE1_DBG=1` prints the E1 and g4row RMS per layer.
3. Run `e1sweep.sh` (sign and scale on even mini nets) and `e1win.sh` (layer windows on odd mini nets). Their logs
   are kept here.

The screen recovers `M_old = D21_old W^-T` by a float64 solve. That costs far more than the term itself, and it is
only meant to measure accuracy.

## Recorder for the corrector dumps

The recorder is `scripts/hf/recorder_xs_d3/` (`estimator.py`, `strassen2.py`). Copy `k34.npz` and `mp.npz` from
cn69; they are identical. Run it with
`XMARG2=1 CN_FBOS=8 CM_TAU=0.01 RW_TAU=0.01 CN_RRES=6 XAGE6FROM=14` through
`RUNNER=runfast_hf_ldump.py gohf.sh` (see `scripts/hf/queue_ld3c.sh`). Its own defaults are older settings.
