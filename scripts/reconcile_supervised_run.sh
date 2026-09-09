#!/usr/bin/env bash
# Read-only, bounded reconciliation for one supervised run.  This script
# never signals or discovers processes by name.
set -u

STATUS_FILE="${1:-$HOME/work/sglang-exl3/server.status.json}"
printf 'reconcile_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'status_file=%s\n' "$STATUS_FILE"

if [ ! -f "$STATUS_FILE" ]; then
  echo 'server_state=NO_STATUS_FILE'
else
  cat "$STATUS_FILE"
  PID="$(python3 - "$STATUS_FILE" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    print((data.get("child") or {}).get("pid", ""))
except Exception:
    print("")
PY
)"
  LOG="$(python3 - "$STATUS_FILE" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    print(data.get("log", ""))
except Exception:
    print("")
PY
)"
  if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
    PGID="$(ps -o pgid= -p "$PID" | tr -d ' ' || true)"
    echo "leader_pid=$PID"
    ps -o pid=,ppid=,pgid=,lstart=,stat=,cmd= -p "$PID" || true
    if [[ "$PGID" =~ ^[0-9]+$ ]]; then
      echo "owned_process_group=$PGID"
      ps -eo pid=,ppid=,pgid=,stat=,cmd= | awk -v pgid="$PGID" '$3 == pgid {print}' || true
    fi
  else
    echo "leader_state=ABSENT_OR_NOT_ALIVE pid=$PID"
  fi
  if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    echo "---log_tail---"
    tail -n 80 "$LOG" || true
  fi
fi

echo '---memory---'
free -h || true
echo '---port_30000---'
ss -ltnp '( sport = :30000 )' 2>/dev/null || true
echo '---health---'
if command -v curl >/dev/null 2>&1; then
  timeout 5 curl --connect-timeout 1 --max-time 3 -fsS http://127.0.0.1:30000/health || true
  echo
else
  echo 'curl_unavailable'
fi
echo '---gpu---'
timeout 5 nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>&1 || true
