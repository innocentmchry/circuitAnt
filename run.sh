#!/usr/bin/env bash
# Run every testcase in a folder (as cada1067's run.sh does):
#
#   circuitAnt/run.sh testcase              folder relative to the current directory or circuitAnt/
#   circuitAnt/run.sh /full/path/testcase
#
# Each <folder>/<case>/prompt.txt is fed to circuitant; its responses are printed.
# The prompts name their files relative to the folder's parent (testcase/<case>/...),
# so circuitant runs there.  Its outputs (<case>.log, written netlists, result lists)
# go to config.yaml's output_dir, or to that same parent folder if it is not set.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -eq 1 ]] || { echo "usage: run.sh <testcase folder>" >&2; exit 2; }
INPUT="$1"
[[ -d "$INPUT" ]] || INPUT="$DIR/$1"
[[ -d "$INPUT" ]] || { echo "run.sh: folder not found: $1" >&2; exit 1; }
INPUT="$(cd "$INPUT" && pwd)"
cd "$(dirname "$INPUT")"
OUT="$(python3 -c 'import sys, yaml; print((yaml.safe_load(open(sys.argv[1])) or {}).get("output_dir") or "")' "$DIR/config.yaml")"
if [[ -n "$OUT" ]]; then OUT="$(cd "$DIR" && mkdir -p "$OUT" && cd "$OUT" && pwd)"; else OUT="$PWD"; fi

export CIRCUITANT_RUN="${CIRCUITANT_RUN:-$(date +%Y%m%d_%H%M%S)}"  # one log folder for this run's testcases
PASS=0
FAIL=0
for test_dir in "$INPUT"/*/; do
    name="$(basename "$test_dir")"
    if [[ ! -f "$test_dir/prompt.txt" ]]; then
        echo "[SKIP] $name: no prompt.txt in $test_dir"
        continue
    fi
    echo "=== $name"
    rm -f "$OUT/$name.log"
    "$DIR/circuitant" -config "$DIR/config.yaml" < "$test_dir/prompt.txt" || true
    if [[ -f "$OUT/$name.log" ]]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); echo "[FAIL] $name: no $name.log"; fi
done
echo "$PASS testcases ran, $FAIL failed (outputs in $OUT; logs in debugging/logs/$CIRCUITANT_RUN)"
[[ $FAIL -eq 0 ]]
