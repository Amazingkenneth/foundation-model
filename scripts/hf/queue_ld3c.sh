#!/bin/bash
# v3 per-layer dumps (correlation-structure features) at cn61 settings: f0-199 + m0-99 first (A/B), then f200-999
cd /tmp/claude-0/acc
B="XMARG2=1 CN_FBOS=8 CM_TAU=0.01 RW_TAU=0.01 CN_RRES=6 XAGE6FROM=14"
W=${1:-1}
F=$(seq -s, -f "m%g" 0 99),$(seq -s, -f "f%g" 0 199)
env $B RUNNER=runfast_hf_ldump.py ./gohf.sh /tmp/claude-0/sc/xs_d3 ld61c "$F" $W; echo "ld61c_ab complete"
echo done
