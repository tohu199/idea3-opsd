#!/bin/bash
# Usage: check_arm.sh ARM [PATTERN]
ARM=${1:-A0}
PAT=${2:-loss}
IDEA3_HOME=${IDEA3_HOME:-$(cd "$(dirname "$0")/.." && pwd)}
LOG=$(ls -t $IDEA3_HOME/logs/${ARM}_*.log 2>/dev/null | grep -v vram | grep -v memavail | head -1)
echo "LOG=$LOG"
if [ -n "$LOG" ]; then
  $IDEA3_HOME/.venv/bin/python $IDEA3_HOME/scripts/show_log.py "$LOG" "$PAT" | tail -8
  echo "--- last lines ---"
  tail -3 "$LOG"
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
