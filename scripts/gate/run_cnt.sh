#!/bin/bash
# usage: scripts/gate/run_cnt.sh <candidate dir, repo-relative> <tag>: client/server round trips per MLP, by op
# (two MLPs: the first is cold). Needs the virtualenvs of emu123.sh under EMU_DIR (default /tmp/claude-0/emu123).
cd "$(dirname "$0")/../.."
E=${EMU_DIR:-/tmp/claude-0/emu123}; cp scripts/gate/drive_cnt.py $E/drive_cnt.py
SOCK=$E/fs_cnt_$2.sock; rm -f $SOCK
FLOPSCOPE_MAX_ARRAY_BYTES=136000000 $E/srv/bin/python -m flopscope_server --url "ipc://$SOCK" --timeout 3600 > $E/server_cnt_$2.log 2>&1 &
SRV=$!
for i in $(seq 1 30); do [ -S $SOCK ] && break; sleep 1; done
FLOPSCOPE_SERVER_URL="ipc://$SOCK" EMU_STUB=$E/stub EMU_ROOT=$(pwd) timeout 2000 $E/cli/bin/python $E/drive_cnt.py $1 2>&1 \
  | grep -v -i warn | grep -E "^MLP|^by op"
kill $SRV
