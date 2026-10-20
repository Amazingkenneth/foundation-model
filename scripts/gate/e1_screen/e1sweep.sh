#!/bin/bash
# E1 sign/scale sweep, chain only (c70, no corrector), 24 mini nets each
cd /tmp/claude-0/acc
IDS=$(seq -s, -f "m%g" 0 4 95)
for v in 0 0.5 -0.5 0.25 -0.25; do
  t=e1s_$(echo $v | tr '.-' 'pm')
  XE1=$v ./gohf.sh /tmp/claude-0/sc/c70 $t "$IDS" 4
done
/tmp/claude-0/venv/bin/python -I /tmp/claude-0/sc/eval_hf.py e1s_0 e1s_0p5 e1s_m0p5 e1s_0p25 e1s_m0p25
echo "e1sweep done"
