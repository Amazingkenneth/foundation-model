#!/bin/bash
# usage: goall.sh <dir> <tag> [nproc]  (env knobs inherited): fast mode on all 75 labeled networks
cd /tmp/claude-0/acc
NP=${3:-4}
ALL=$(seq 1000 1063; seq 5000 5010)
for i in $(seq 0 $((NP-1))); do
  S=$(echo $ALL | tr ' ' '\n' | awk -v i=$i -v np=$NP 'NR%np==i' | paste -sd,)
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/claude-0/venv/bin/python ${RUNNER:-runfast.py} $1 $2 $S > $2.$i.log 2>&1 &
done
wait
