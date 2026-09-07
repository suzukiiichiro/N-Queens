#!/usr/bin/env bash
# 388_debug_run.sh
#
# rev388 -- everyday-use debug launcher (NOT a *_validate.sh harness).
# Builds and runs 388Py_kernel_maxd14_final.py with -d, capturing the
# full build+run session into a rev-numbered directory, following the
# same bash-tee pattern every *_validate.sh in this project already
# uses (nothing new invented here, just packaged for daily use instead
# of one-off validation).
#
# Usage:
#   bash 388_debug_run.sh              # runs: ./388Py..._final -g -d
#   bash 388_debug_run.sh -g 21 21 32 484 1 0 7 37 -d
#                                       # runs whatever args you pass,
#                                       # -d is appended automatically
#                                       # if you forget it
#
# Does NOT run ncu. See 388's README append for why: ncu stays a
# separate, N-fixed script (the 375 method -- N=21 or the N=18 trick),
# not something this per-N loop launches automatically. A companion
# ncu script writing into the SAME directory this script creates is a
# natural next small step, not bundled into this one.

set -u
PY_SRC="${PY_SRC:-388Py_kernel_maxd14_final.py}"
BIN="${BIN:-388Py_kernel_maxd14_final}"
CODON="${CODON:-codon}"
REV_TAG="${REV_TAG:-388}"

if [[ ! -f "$PY_SRC" ]]; then
  echo "FAIL: $PY_SRC not found in $(pwd)"
  exit 1
fi

DEBUG_DIR="${REV_TAG}_debug_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEBUG_DIR"
echo "Debug log directory: $DEBUG_DIR"

# Build (only if the binary doesn't already exist -- rebuilding on
# every debug run would be wasteful for a same-session repeat).
if [[ ! -x "$BIN" ]]; then
  echo "Building $PY_SRC with $CODON build -release..."
  "$CODON" build -release -o "$BIN" "$PY_SRC" 2>&1 | tee "$DEBUG_DIR/build.log"
  if [[ ! -x "$BIN" ]]; then
    echo "FAIL: binary $BIN was not produced -- see $DEBUG_DIR/build.log"
    exit 1
  fi
else
  echo "Reusing existing binary $BIN (delete it first to force a rebuild)."
fi

# Args: whatever was passed to this script, defaulting to bare -g if
# nothing was given. -d is appended if not already present, since the
# whole point of this launcher is a debug run.
if [[ $# -eq 0 ]]; then
  RUN_ARGS=(-g -d)
else
  RUN_ARGS=("$@")
  has_d=0
  for a in "${RUN_ARGS[@]}"; do
    [[ "$a" == "-d" ]] && has_d=1
  done
  if [[ "$has_d" -eq 0 ]]; then
    RUN_ARGS+=(-d)
  fi
fi

echo "Running: ./$BIN ${RUN_ARGS[*]}"
./"$BIN" "${RUN_ARGS[@]}" 2>&1 | tee "$DEBUG_DIR/run.log"

echo ""
echo "Debug session captured in: $DEBUG_DIR/"
ls -la "$DEBUG_DIR"
