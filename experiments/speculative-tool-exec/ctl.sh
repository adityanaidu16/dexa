#!/usr/bin/env bash
# Process control for the batch driver, kept in a file so the patterns never appear in an interactive shell's command line.
cd /home/user/dexa/experiments/speculative-tool-exec
case "$1" in
  stop)
    for p in $(pgrep -f "bash ./run_batches.sh"); do kill "$p" 2>/dev/null; done
    for p in $(pgrep -f "python3 replay.py --sessions"); do kill "$p" 2>/dev/null; done
    sleep 1; docker ps -q --filter name=spec_ | xargs -r docker rm -f >/dev/null 2>&1; echo stopped ;;
  start)
    nohup ./run_batches.sh >> runs/driver.log 2>&1 & echo "started pid $!" ;;
  status)
    pgrep -af "bash ./run_batches.sh" | head -2; pgrep -af "python3 replay.py --sessions" | cut -c1-80 | head -2; tail -3 runs/driver.log ;;
esac
