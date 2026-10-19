#!/bin/bash
# back-to-back 0.12.3 client/server emulator walls for two estimator dirs (suite shape, one network)
cd /home/user/foundation-model
for D in "$@"; do
  T=$(basename $D)
  EMU_DIR=/tmp/claude-0/emu123 timeout 2400 bash /tmp/claude-0/sc/dbg55/emu123.sh $D 1 > /tmp/claude-0/sc/s5/emu_$T.log 2>&1
  echo "$T: $(grep -v -i warn /tmp/claude-0/sc/s5/emu_$T.log | grep -E 'MLP 0|Error|Trace' | head -2)"
done
echo emu_pair done
