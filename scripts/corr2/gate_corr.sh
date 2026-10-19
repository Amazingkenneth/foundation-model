#!/bin/bash
# usage: gate_corr.sh <model.pt> <name> [src scratch, default c68]: export corr.npz, build candidates/<name> + <name>r,
# torch-vs-chain parity on m50, then the in-chain mini 100 (fast mode, 4 workers) scored against ld61 (no corrector)
# and c68f1000 (cn68's corrector)
set -e
M=$1; N=$2; SRC=${3:-c68}
C=/tmp/claude-0/sc/corr2/corr_$N.npz
cd /tmp && /tmp/claude-0/venv/bin/python -I /tmp/claude-0/sc/corr2/export_rnn.py $M $C
SRC=$SRC /tmp/claude-0/sc/corr2/build_cn66.sh v5b_$N $C | tail -1
/tmp/claude-0/venv/bin/python -I /tmp/claude-0/sc/corr2/torch_pred_rnn.py $M ld61c_v3 m50 ld61c
cd /tmp/claude-0/acc; rm -rf ${N}_par
OMP_NUM_THREADS=1 timeout 900 /tmp/claude-0/venv/bin/python runfast_hf.py /home/user/foundation-model/candidates/v5b_$N ${N}_par m50 2>&1 | grep -v -i warn | tail -1
/tmp/claude-0/venv/bin/python -I -c "
import numpy as np
a = np.load('/tmp/claude-0/acc/${N}_par/pred_m50.npy')[-1].astype(np.float64); t = np.load('/tmp/claude-0/sc/corr2/torchpred_m50.npy')
print('max |chain - torch| on m50: %.2e (rms pred %.2e)' % (np.abs(a - t).max(), np.sqrt(np.mean(t ** 2))))
"
rm -rf /tmp/claude-0/acc/${N}f1000
./gohf.sh /home/user/foundation-model/candidates/v5b_$N ${N}f1000 $(seq -s, -f "m%g" 0 99) 4
/tmp/claude-0/venv/bin/python -I /tmp/claude-0/sc/eval_hf.py ld61 c68f1000 ${N}f1000
echo "gate_corr done"
