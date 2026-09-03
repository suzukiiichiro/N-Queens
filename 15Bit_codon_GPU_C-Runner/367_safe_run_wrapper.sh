#!/usr/bin/env bash
# 367_safe_run_wrapper.sh
#
# Generic safety wrapper for any command that might exhaust host
# resources -- written after the N=23 incident (366Py's constellation
# generation for N=23 grew unboundedly, exhausted host memory, and
# made the whole machine unresponsive over SSH and serial console,
# requiring a force-stop that destroyed an ephemeral instance-store
# volume). This wrapper adds NO Codon source changes at all -- it is a
# pure shell-level guard, deliberately kept outside the single-
# variable-discipline code deltas (361-366) since it touches
# infrastructure, not kernel/dispatch logic.
#
# Design principle: DO NOT try to predict how big N=23+ will be.
# The observed N=21->N=22 growth ratio was ~14.18x (2,025,282 ->
# 28,719,035 records), but constellation counts are expected to grow
# combinatorially with N, not at a fixed ratio -- extrapolating a
# single data point is unreliable (see header comment in the .sh
# harness for the arithmetic). Instead of guessing a safe N, this
# wrapper enforces HARD resource ceilings so that if a run is going to
# exceed them, it fails in a controlled way (the wrapped process is
# killed, this script exits nonzero, the host stays responsive) rather
# than taking the whole machine down.
#
# Three independent guards, all active by default:
#   1. Memory ceiling (ulimit -v) -- the wrapped process cannot
#      allocate more virtual memory than MAX_MEM_PERCENT of total
#      system RAM. If it tries, the allocation fails and the process
#      exits (Codon/Python raise MemoryError-equivalent, or the OS
#      kills just that process) -- NOT the whole-system OOM killer
#      picking arbitrary victims or the box locking up.
#   2. Wall-clock timeout (timeout(1)) -- the wrapped process is
#      killed after TIMEOUT_DURATION regardless of what it's doing.
#   3. Disk-space monitor (background loop) -- checks free space on
#      DISK_PATH every DISK_CHECK_INTERVAL_SEC seconds while the
#      wrapped process runs; if free space drops below
#      MIN_FREE_DISK_GB, the wrapped process is killed immediately
#      (the ulimit/timeout guards protect against memory/time blowup,
#      but generation writes to disk incrementally, so a
#      still-growing bin file needs its own live check, not just a
#      pre-flight one).
#
# Usage:
#   367_safe_run_wrapper.sh -- <command> [args...]
#
# Config via environment variables (all have conservative defaults):
#   MAX_MEM_PERCENT        (default 70)   -- cap as % of total system RAM
#   TIMEOUT_DURATION        (default 2h)   -- passed straight to timeout(1)
#   MIN_FREE_DISK_GB        (default 20)   -- pre-flight AND live-monitor floor
#   DISK_PATH               (default .)    -- filesystem to check free space on
#   DISK_CHECK_INTERVAL_SEC (default 10)   -- live-monitor poll interval
#
# Example (N=23 attempt, conservative defaults):
#   ./367_safe_run_wrapper.sh -- \
#     ./366Py_maxd_check -g 23 23 32 484 1 0 8 34 3 7 0 0 1 2 2048 9
#
# Example (tighter caps, e.g. on a smaller/shared host):
#   MAX_MEM_PERCENT=50 TIMEOUT_DURATION=30m MIN_FREE_DISK_GB=50 \
#     DISK_PATH=/data/nq \
#     ./367_safe_run_wrapper.sh -- \
#     ./366Py_maxd_check -g 23 23 32 484 1 0 8 34 3 7 0 0 1 2 2048 9

set -u

MAX_MEM_PERCENT="${MAX_MEM_PERCENT:-70}"
TIMEOUT_DURATION="${TIMEOUT_DURATION:-2h}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-20}"
DISK_PATH="${DISK_PATH:-.}"
DISK_CHECK_INTERVAL_SEC="${DISK_CHECK_INTERVAL_SEC:-10}"

echo "===== 367 safety wrapper: config ====="
echo "MAX_MEM_PERCENT=$MAX_MEM_PERCENT%  TIMEOUT_DURATION=$TIMEOUT_DURATION"
echo "MIN_FREE_DISK_GB=$MIN_FREE_DISK_GB  DISK_PATH=$DISK_PATH  DISK_CHECK_INTERVAL_SEC=$DISK_CHECK_INTERVAL_SEC"
echo "========================================"
echo ""

# ---------------------------------------------------------------------
# Parse "-- <command> [args...]"
# ---------------------------------------------------------------------
if [[ "${1:-}" != "--" ]]; then
  echo "Usage: $0 -- <command> [args...]" >&2
  exit 1
fi
shift
if [[ $# -eq 0 ]]; then
  echo "ERROR: no command given after --" >&2
  exit 1
fi

# ---------------------------------------------------------------------
# 1. Pre-flight disk-space check.
# ---------------------------------------------------------------------
if ! command -v df >/dev/null 2>&1; then
  echo "WARN: df not found, skipping pre-flight disk check" >&2
else
  free_kb=$(df -Pk "$DISK_PATH" 2>/dev/null | awk 'NR==2 {print $4}')
  if [[ -z "$free_kb" ]]; then
    echo "WARN: could not determine free space on $DISK_PATH, skipping pre-flight check" >&2
  else
    free_gb=$((free_kb / 1024 / 1024))
    echo "Pre-flight: $free_gb GB free on $DISK_PATH (require >= $MIN_FREE_DISK_GB GB)"
    if [[ "$free_gb" -lt "$MIN_FREE_DISK_GB" ]]; then
      echo "ABORT: free disk space ($free_gb GB) is below MIN_FREE_DISK_GB ($MIN_FREE_DISK_GB GB)." >&2
      echo "Not starting the wrapped command. Free up space or lower MIN_FREE_DISK_GB if you accept the risk." >&2
      exit 2
    fi
  fi
fi

# ---------------------------------------------------------------------
# 2. Compute memory ceiling in KB (ulimit -v unit) from total system RAM.
# ---------------------------------------------------------------------
mem_limit_kb=""
if command -v free >/dev/null 2>&1; then
  total_mem_kb=$(free -k 2>/dev/null | awk '/^Mem:/ {print $2}')
  if [[ -n "$total_mem_kb" ]]; then
    mem_limit_kb=$(( total_mem_kb * MAX_MEM_PERCENT / 100 ))
    mem_limit_gb=$(( mem_limit_kb / 1024 / 1024 ))
    echo "Memory ceiling: ${mem_limit_gb} GB (${MAX_MEM_PERCENT}% of $(( total_mem_kb / 1024 / 1024 )) GB total)"
  fi
fi
if [[ -z "$mem_limit_kb" ]]; then
  echo "WARN: could not determine total system RAM (no 'free' command); memory ceiling NOT applied." >&2
  echo "WARN: this run is NOT protected against a repeat of the N=23 OOM incident." >&2
fi

# ---------------------------------------------------------------------
# 3. Launch the wrapped command under timeout + ulimit, in the
#    background, so we can run the disk-space monitor loop alongside it.
# ---------------------------------------------------------------------
LOGFILE="367_wrapped_$(date +%Y%m%d_%H%M%S).log"
echo "Wrapped command: $*"
echo "Log: $LOGFILE"
echo ""

(
  if [[ -n "$mem_limit_kb" ]]; then
    ulimit -v "$mem_limit_kb"
  fi
  exec timeout --signal=TERM --kill-after=30s "$TIMEOUT_DURATION" "$@"
) > "$LOGFILE" 2>&1 &
CHILD_PID=$!

# ---------------------------------------------------------------------
# 4. Live disk-space monitor loop, runs alongside the child.
# ---------------------------------------------------------------------
MONITOR_TRIGGERED=0
if command -v df >/dev/null 2>&1; then
  (
    while kill -0 "$CHILD_PID" 2>/dev/null; do
      sleep "$DISK_CHECK_INTERVAL_SEC"
      free_kb=$(df -Pk "$DISK_PATH" 2>/dev/null | awk 'NR==2 {print $4}')
      if [[ -n "$free_kb" ]]; then
        free_gb=$((free_kb / 1024 / 1024))
        if [[ "$free_gb" -lt "$MIN_FREE_DISK_GB" ]]; then
          echo "[367-monitor] free disk on $DISK_PATH dropped to ${free_gb} GB (< $MIN_FREE_DISK_GB GB) -- killing wrapped process (pid $CHILD_PID)" >> "$LOGFILE"
          touch "${LOGFILE}.disk_triggered"
          kill -TERM "$CHILD_PID" 2>/dev/null
          sleep 5
          kill -KILL "$CHILD_PID" 2>/dev/null
          break
        fi
      fi
    done
  ) &
  MONITOR_PID=$!
fi

# ---------------------------------------------------------------------
# 5. Wait for the wrapped command, tail the log as it goes.
# ---------------------------------------------------------------------
tail -f "$LOGFILE" &
TAIL_PID=$!

wait "$CHILD_PID"
CHILD_EXIT=$?

kill "$TAIL_PID" 2>/dev/null
[[ -n "${MONITOR_PID:-}" ]] && kill "$MONITOR_PID" 2>/dev/null

echo ""
echo "===== 367 safety wrapper: summary ====="
if [[ -f "${LOGFILE}.disk_triggered" ]]; then
  echo "RESULT: killed by disk-space monitor (free space fell below ${MIN_FREE_DISK_GB} GB on $DISK_PATH)."
  rm -f "${LOGFILE}.disk_triggered"
  echo "The host should remain responsive -- this is the controlled failure this wrapper exists to produce."
  exit 3
elif [[ "$CHILD_EXIT" -eq 124 || "$CHILD_EXIT" -eq 137 ]]; then
  echo "RESULT: killed by timeout ($TIMEOUT_DURATION elapsed, exit code $CHILD_EXIT)."
  echo "The host should remain responsive -- this is the controlled failure this wrapper exists to produce."
  exit 4
elif [[ "$CHILD_EXIT" -ne 0 ]]; then
  echo "RESULT: wrapped command exited nonzero ($CHILD_EXIT) -- check $LOGFILE."
  echo "If this looks like an out-of-memory failure (e.g. Codon/Python allocation error,"
  echo "or a bare 'Killed' with no further output), the ulimit ceiling did its job:"
  echo "this process died alone, instead of triggering a whole-host OOM again."
  exit "$CHILD_EXIT"
else
  echo "RESULT: wrapped command completed normally (exit 0). See $LOGFILE for output."
  exit 0
fi
