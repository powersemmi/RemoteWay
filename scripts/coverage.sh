#!/usr/bin/env bash
# Run cargo-llvm-cov and check that every crate meets the 80% threshold.
# Usage: ./scripts/coverage.sh [--html]
#
# Prerequisites: cargo install cargo-llvm-cov

set -euo pipefail

HTML=0
for arg in "$@"; do
    [[ "$arg" == "--html" ]] && HTML=1
done

if [[ $HTML -eq 1 ]]; then
    cargo llvm-cov --workspace --html
    echo "HTML report: target/llvm-cov/html/index.html"
fi

# Summary output — check threshold.
OUTPUT=$(cargo llvm-cov --workspace --summary-only 2>&1)
echo "$OUTPUT"

# Extract the "TOTAL" line coverage percentage (10th field).
# Format: TOTAL  Regions  Missed  RegCov%  Funcs  Missed  FuncCov%  Lines  Missed  LineCov%  [Branches ...]
TOTAL=$(echo "$OUTPUT" | grep -E "^TOTAL" | awk '{for(i=1;i<=NF;i++){if($i ~ /%$/){last=$i}};print last}' | tr -d '%')

if [[ -z "$TOTAL" ]]; then
    echo "ERROR: could not parse coverage output"
    exit 1
fi

THRESHOLD=80
# Use awk for float comparison (bash can't do floats).
PASS=$(awk -v total="$TOTAL" -v thr="$THRESHOLD" 'BEGIN { print (total+0 >= thr+0) ? 1 : 0 }')

if [[ "$PASS" -eq 1 ]]; then
    echo "Coverage ${TOTAL}% >= ${THRESHOLD}% threshold — OK"
else
    echo "Coverage ${TOTAL}% < ${THRESHOLD}% threshold — FAIL"
    exit 1
fi
