#!/usr/bin/env bash
# Revision grid v2: no min-max scaling (all models are tree-based), 5 seeds everywhere,
# two new configurations (split10, lgbm). Resumable: finished runs are skipped.
#
# Start (keeps the laptop awake while it runs):
#   nohup systemd-inhibit --what=sleep:idle --why="revision grid" \
#     bash experiments/run_round2.sh > results/round2/logs/run_all.log 2>&1 &
# Stop:  pkill -f run_round2.sh; pkill -f run_grid.py
#
# BigFeat phase: set BIGFEAT_CONFIGS once the team decides how BigFeat is run.
#   "bigfeat both"          BigFeat as-is (it min-max scales internally and passes scaled originals on)
#   "bigfeat_raw both_raw"  BigFeat's generated features + the unscaled original features
#   "bigfeat bigfeat_raw both both_raw"  run both variants
set -u
cd "$(dirname "$0")/.."
export REVISION_OUT="${REVISION_OUT:-$PWD/results/round2}"
mkdir -p "$REVISION_OUT/logs"
PY="nice -n 10 python3 -W ignore experiments/run_grid.py"
SEEDS=0-4
CHEAP="original split10 lgbm"
BIGFEAT_CONFIGS="${BIGFEAT_CONFIGS:-}"

echo "== phase 1: N-BaIoT in-fold top-20, cheap configs"; date
$PY --dataset nbaiot --feature-mode infold20 --units 1-9 --seeds $SEEDS --configs $CHEAP --workers 2
echo "== phase 2: CIC-IoT-2023, cheap configs"; date
$PY --dataset ciciot2023 --units all --seeds $SEEDS --configs $CHEAP --workers 2
echo "== phase 3: Edge-IIoTset, cheap configs"; date
$PY --dataset edgeiiot --units all --seeds $SEEDS --configs $CHEAP --workers 1
echo "== phase 4: N-BaIoT all 115 features, cheap configs"; date
$PY --dataset nbaiot --feature-mode all --units 1-9 --seeds $SEEDS --configs $CHEAP --workers 2
echo "== phase 5: N-BaIoT pre-built 20 (data audit only), baseline"; date
$PY --dataset nbaiot --feature-mode prebuilt20 --units 1-9 --seeds $SEEDS --configs original --workers 2
echo "== phase 6: TPOT"; date
$PY --dataset ciciot2023 --units all --seeds $SEEDS --configs tpot --workers 2
$PY --dataset edgeiiot --units all --seeds $SEEDS --configs tpot --workers 1
$PY --dataset nbaiot --feature-mode infold20 --units 1-9 --seeds $SEEDS --configs tpot --workers 2
if [ -n "$BIGFEAT_CONFIGS" ]; then
  echo "== phase 7: BigFeat ($BIGFEAT_CONFIGS)"; date
  $PY --dataset ciciot2023 --units all --seeds $SEEDS --configs $BIGFEAT_CONFIGS --workers 2
  $PY --dataset edgeiiot --units all --seeds $SEEDS --configs $BIGFEAT_CONFIGS --workers 1
  $PY --dataset nbaiot --feature-mode infold20 --units 1-9 --seeds $SEEDS --configs $BIGFEAT_CONFIGS --workers 2
  NOTPOT=$(echo $BIGFEAT_CONFIGS | tr ' ' '\n' | grep -v '^both' | tr '\n' ' ')
  [ -n "$NOTPOT" ] && $PY --dataset nbaiot --feature-mode all --units 1-9 --seeds $SEEDS --configs $NOTPOT --workers 2
else
  echo "== phase 7 skipped: BIGFEAT_CONFIGS not set (waiting for the BigFeat decision)"
fi
echo "== analysis"; date
python3 -W ignore experiments/analyze.py > "$REVISION_OUT/logs/analyze.log" 2>&1
echo "== done"; date
