#!/usr/bin/env bash
# Set up a native macOS Python environment for EpiSBI and rerun the whole
# measles pipeline (Steps 1-14), timing each step.
#   Usage:  bash run_all_mac.sh          # setup (first time) + run all steps
#           bash run_all_mac.sh 9 14     # run only steps 9..14
# Previous MSI (Windows) results are kept in output_MSI/ for comparison.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"            # EpiSBI-main
VENV="$ROOT/.venv_mac"               # the bundled .venv is Windows-only
FIRST="${1:-1}"; LAST="${2:-14}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "== Creating Mac environment in $VENV"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.13 "$VENV" || exit 1
    PIPI="uv pip install --python $VENV/bin/python"
  else
    PY=$(command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3)
    "$PY" -c 'import sys; assert sys.version_info>=(3,10), "need Python>=3.10 (brew install python@3.13)"' || exit 1
    "$PY" -m venv "$VENV" || exit 1
    "$VENV/bin/python" -m pip install -q --upgrade pip
    PIPI="$VENV/bin/python -m pip install"
  fi
  $PIPI -r "$HERE/requirements.txt" tensorboard || exit 1
  $PIPI -e "$ROOT" || exit 1      # installs the episbi package itself
fi
PY="$VENV/bin/python"
"$PY" -c "import episbi, sbi, torch, sklearn; print('episbi OK | sbi', sbi.__version__, '| torch', torch.__version__)" || exit 1

cd "$HERE"; mkdir -p output
LOG="output/mac_timing_$(date +%Y%m%d_%H%M).csv"
echo "step,script,seconds,status" > "$LOG"
echo "== Running steps $FIRST-$LAST on $(sysctl -n machdep.cpu.brand_string 2>/dev/null) ($(sysctl -n hw.ncpu 2>/dev/null) cores)"
T0=$(date +%s)
for i in $(seq "$FIRST" "$LAST"); do
  S=$(ls step${i}_*.py); echo; echo "################ Step $i: $S ################"
  t=$(date +%s); "$PY" "$S" 2>&1 | tee "output/step${i}_mac.log"; rc=${PIPESTATUS[0]}
  dt=$(( $(date +%s) - t )); st=$([ $rc -eq 0 ] && echo ok || echo FAILED)
  echo "$i,$S,$dt,$st" >> "$LOG"; echo ">> Step $i $st in $((dt/60))m $((dt%60))s"
  [ $rc -ne 0 ] && { echo "Stopping: step $i failed (see output/step${i}_mac.log)"; break; }
done
TT=$(( $(date +%s) - T0 )); echo "total,,$TT," >> "$LOG"
echo; echo "== Done in $((TT/60))m $((TT%60))s. Per-step times: $HERE/$LOG"; column -s, -t "$LOG"
