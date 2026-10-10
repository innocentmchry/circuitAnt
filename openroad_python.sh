#!/usr/bin/env bash
# Run a Python script inside OpenROAD's embedded Python interpreter.
#
#   circuitAnt/openroad_python.sh circuitAnt/tests_automation/test_engine.py
#   circuitAnt/openroad_python.sh some_script.py
#
# The `openroad`/`odb` modules are compiled into the openroad binary and are
# not importable by a plain python3, so everything runs through
# `openroad -python`.  Extra packages (openai) come from the system Python
# 3.10 user site (pip install --user openai).
set -euo pipefail

OPENROAD="${OPENROAD_BIN:-$HOME/OpenROAD/build/bin/openroad}"
export ABC_BIN="${ABC_BIN:-$HOME/abc/abc}"  # ABC used for equivalence proofs
[[ -x "$OPENROAD" ]] || { echo "openroad_python.sh: missing $OPENROAD" >&2; exit 127; }
[[ $# -ge 1 ]] || { echo "usage: openroad_python.sh <script.py> [args...]" >&2; exit 2; }

exec "$OPENROAD" -no_init -no_splash -no_settings -python -exit "$@"
