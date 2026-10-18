#!/bin/bash
# usage: gate_emu.sh <candidate dir> <tag>: 0.12.3 client/server emulator at the suite shape, then the smoke shapes
cd /home/user/foundation-model
D=$1; T=$2
EMU_DIR=/tmp/claude-0/emu123 timeout 2400 bash /tmp/claude-0/sc/dbg55/emu123.sh $D 1 > /tmp/claude-0/sc/dbg55/emu_$T.log 2>&1
grep -v -i warn /tmp/claude-0/sc/dbg55/emu_$T.log | grep -E "MLP|Error|Trace|C/B|resid" | head -6
E=/tmp/claude-0/emu123; SOCK=$E/fs_$T.sock; rm -f $SOCK
FLOPSCOPE_MAX_ARRAY_BYTES=136000000 $E/srv/bin/python -m flopscope_server --url "ipc://$SOCK" --timeout 3600 > $E/server_$T.log 2>&1 &
SRV=$!
for i in $(seq 1 30); do [ -S $SOCK ] && break; sleep 1; done
FLOPSCOPE_SERVER_URL="ipc://$SOCK" EMU_STUB=$E/stub EMU_ROOT=/home/user/foundation-model timeout 1200 $E/cli/bin/python $E/drive_shapes.py $D 2>&1 | grep -v -i warn | grep -E "^[0-9]+ |FAILED|Error" | tr '\n' ' '
kill $SRV
echo; echo "gate_emu done"
