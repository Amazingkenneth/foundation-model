#!/bin/bash
# usage: build_cn66.sh <name> <corr.npz>: candidates/<name> (5 levels) and <name>r (4 levels) from the c66 estimator
set -e
cd /home/user/foundation-model/candidates
N=$1; C=$2
rm -rf $N ${N}r; mkdir -p $N ${N}r
for d in $N ${N}r; do cp /tmp/claude-0/sc/${SRC:-c66}/{LICENSE_504aldo,estimator.py,k34.npz,mp.npz,strassen4.py} $d/; cp $C $d/corr.npz; done
python3 - ${N}r/estimator.py <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = 'SLEV, SB, RLEV, RMB = 5, 2, 4, 20   # rank-320 families keep 4 levels'
b = 'SLEV, SB, RLEV, RMB = int(__import__("os").environ.get("CN_SLEV", "4")), 2, 4, 20   # 4-level twin: young families at 4 Strassen levels (wall / residual margin under grader load); rank-320 families keep 4 levels'
assert s.count(a) == 1; open(p, "w").write(s.replace(a, b))
PY
diff $N/estimator.py ${N}r/estimator.py | head -4; ls $N ${N}r
