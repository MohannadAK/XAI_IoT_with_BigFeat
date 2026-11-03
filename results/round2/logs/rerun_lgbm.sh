#!/usr/bin/env bash
# Re-run LightGBM (now with max_delta_step=1) on every v2 task, then re-analyse.
cd "$(dirname "$0")/../../.."
export REVISION_OUT="$PWD/results/round2"
PY="nice -n 10 python3 -W ignore experiments/run_grid.py"
echo "== LightGBM re-run"; date
$PY --dataset nbaiot --feature-mode infold20 --units 1-9 --seeds 0-4 --configs lgbm --workers 2
$PY --dataset nbaiot --feature-mode all --units 1-9 --seeds 0-4 --configs lgbm --workers 2
$PY --dataset edgeiiot --units all --seeds 0-4 --configs lgbm --workers 1
$PY --dataset ciciot2023 --units all --seeds 0-4 --configs lgbm --workers 2
echo "== analysis"; date
python3 -W ignore experiments/analyze.py > "$REVISION_OUT/logs/analyze.log" 2>&1
echo "== done"; date
