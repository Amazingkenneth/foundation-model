#!/bin/bash
# usage: gohf.sh <dir> <tag> <ids comma list> [nproc] (env knobs inherited): fast mode on public WhestBench networks
cd /tmp/claude-0/acc
NP=${4:-4}
for i in $(seq 0 $((NP-1))); do
  S=$(echo $3 | tr ',' '\n' | awk -v i=$i -v np=$NP 'NR%np==i' | paste -sd,)
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/claude-0/venv/bin/python ${RUNNER:-runfast_hf.py} $1 $2 $S > $2.h$i.log 2>&1 &
done
wait
