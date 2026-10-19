#!/bin/bash
# v3 per-layer dumps for f200-999 at cn61 settings (finished networks are skipped)
cd /tmp/claude-0/acc
B="XMARG2=1 CN_FBOS=8 CM_TAU=0.01 RW_TAU=0.01 CN_RRES=6 XAGE6FROM=14"
for c in 1 2 3 4; do
  F=$(seq -s, -f "f%g" $((c*200)) $((c*200+199)))
  env $B RUNNER=runfast_hf_ldump.py ./gohf.sh /tmp/claude-0/sc/xs_d3 ld61c "$F" 3; echo "ld61c_chunk$c complete"
done
echo done
