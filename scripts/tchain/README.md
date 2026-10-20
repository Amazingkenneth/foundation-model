# T-chain prototype (Oct 20): the leaders' construction on our engine

See `docs/leaders_2026-10-20.md` for the measurements and the decision (not shipped).

- `estimator_c80.py`: cn71's `estimator.py` plus env flags `XAUG`, `XG`, `XK211`, `XEXT`, `XTK34`, `XKNON`,
  `XEXT_PARTS`, `XK211REP` (all off = cn71 chain). Run it from a directory that also holds cn71's `strassen4.py`,
  `k34.npz`, `mp.npz`, `LICENSE_504aldo` and the two table files here, without `corr.npz` (chain only):
  `XAUG=1 XG=1 XTK34=0 XEXT=0.25 python scripts/hf/runfast_hf.py <dir> <tag> m70,m71`.
- `aug_tables.py`, `t211.py`: ARC `mlp_cumulant_propagation` (MIT) current main, `augment=True`, converted to the
  estimator's `TERM_SPECS` format by `gen_aug_specs.py` / `gen211.py` (they need the reference package and
  504aldo's `lean/lean_k3_aug.py` on `sys.path`).
- `patch_c80.py`, `patch_c80_k211.py`: the scripted part of the change from cn71.
- `cubtest.py`, `wcache.py`: spherical-cubature / structured-sampling test on public mini networks.
- `mse.py`: final-layer MSE per prediction tag (`/tmp/claude-0/acc/<tag>/pred_m*.npy`) against the 1e9 labels.
- Layer-by-layer comparison (Oct 20): `dump_ours.py` (our estimator with `XDUMP=1`, fast mode) and `dump_lean.py`
  (the helper-patched lean `/tmp/claude-0/lean_ext/lean_k3_aug_ext.py`, `debug_state`) write per-layer pickles;
  `cmpdump.py` diffs them; `perlayer.py` prints per-layer MSE of prediction sets; `run_mine.py` runs the patched
  lean with arbitrary keyword arguments (used for the walk toward our engine in `docs/leaders_2026-10-20.md`).
