#!/bin/bash
# E1 at scale 0.45: all layers vs late windows, odd mini nets (independent of the sweep's even ids)
cd /tmp/claude-0/acc
IDS=$(seq -s, -f "m%g" 1 4 97)
XE1=0 ./gohf.sh /tmp/claude-0/sc/c70 e1w_ctl "$IDS" 4
for lm in 1 8 11; do XE1=0.45 XE1_LMIN=$lm ./gohf.sh /tmp/claude-0/sc/c70 e1w_l$lm "$IDS" 4; done
/tmp/claude-0/venv/bin/python -I /tmp/claude-0/sc/eval_hf.py e1w_ctl e1w_l1 e1w_l8 e1w_l11
echo "e1win done"
