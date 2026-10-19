#!/bin/bash
# bill + in-process wall for Strassen level mixes (transport trees T, D21 hub H) on the 0.12.3 venv, seed 1
cd /tmp/claude-0/acc123
for th in "$@"; do
  T=${th%,*}; H=${th#*,}
  CN_SLEVT=$T CN_SLEVH=$H OMP_NUM_THREADS=1 timeout 900 /tmp/claude-0/venv123/bin/python runv.py /tmp/claude-0/sc/c63w q_mix_${T}${H} 1 2>&1 | grep "C/B" | sed "s/^/T$T H$H /"
done
echo wallmix done
