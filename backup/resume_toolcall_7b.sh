#!/usr/bin/env bash
# Resume toolcall-7b ops. Fully self-contained (no undefined vars).
set -euo pipefail

P=51027615
HOST=ssh7.vast.ai
PORT=27614
KEY="$HOME/.ssh/runpod_cuda"
VAST="${VAST:-/tmp/pv/bin/vastai}"
R=/workspace/coderepair
LOCAL=/Users/sushanthtiruvaipati/Documents/Default\ Project/crowd-finetune
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -p $PORT root@$HOST"

echo "== start $P =="
$VAST start instance "$P" 2>&1 || true

echo "== poll for running (up to 15 min) =="
ready=no
for i in $(seq 1 60); do
  sleep 15
  st=$($VAST show instances --raw 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print([a['actual_status'] for a in d if a['id']==${P}][0])" 2>/dev/null || echo unknown)
  echo "  ($i/60) status=$st"
  if [ "$st" = "running" ]; then ready=yes; echo "  READY"; break; fi
done
[ "$ready" = yes ] || { echo "  DID NOT START"; $VAST show instances --raw 2>/dev/null | $TMP/pv/bin/python3 -c "import sys,json; [print(a['id'],a.get('label'),a.get('actual_status')) for a in json.load(sys.stdin) if a['id']==${P}]"; exit 1; }

for i in $(seq 1 30); do
  sleep 10
  $SSH 'true' 2>/dev/null && { echo "  sshd up"; break; }
done

echo "== recon adapter state =="
$SSH "cd $R && git log --oneline -1; ls outputs/toolcall-7b-lora/ | grep -E 'adapter_model|README' || echo '  no adapter yet'"
$SSH "tail -3 /workspace/train_7b.log 2>/dev/null || echo 'no train log'"