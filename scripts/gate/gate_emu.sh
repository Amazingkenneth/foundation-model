#!/bin/bash
# usage: scripts/gate/gate_emu.sh <candidate dir, repo-relative> <tag>
# 0.12.3 client/server emulator: one suite-shape MLP (C/B, wall, residual), then the 8 smoke shapes (finite outputs).
# EMU_DIR (default /tmp/claude-0/emu123) holds the two virtualenvs that emu123.sh creates on first use.
set -e
cd "$(dirname "$0")/../.."
D=$1; T=$2; E=${EMU_DIR:-/tmp/claude-0/emu123}; G=scripts/gate
EMU_DIR=$E timeout 2400 bash $G/emu123.sh $D 1 > /tmp/emu_$T.log 2>&1 || true
grep -v -i warn /tmp/emu_$T.log | grep -E "MLP|Error|Trace|C/B|resid" | head -6
cp $G/drive_shapes.py $E/drive_shapes.py
SOCK=$E/fs_$T.sock; rm -f $SOCK
FLOPSCOPE_MAX_ARRAY_BYTES=136000000 $E/srv/bin/python -m flopscope_server --url "ipc://$SOCK" --timeout 3600 > $E/server_$T.log 2>&1 &
SRV=$!
for i in $(seq 1 30); do [ -S $SOCK ] && break; sleep 1; done
FLOPSCOPE_SERVER_URL="ipc://$SOCK" EMU_STUB=$E/stub EMU_ROOT=$(pwd) timeout 1200 $E/cli/bin/python $E/drive_shapes.py $D 2>&1 \
  | grep -v -i warn | grep -E "^[0-9]+ |FAILED|Error" | tr '\n' ' '
kill $SRV
echo; echo "gate_emu done"
