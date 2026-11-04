#!/usr/bin/env bash
# Revision grid v3: two datasets with accuracy headroom, deduplicated during data preparation.
#   ciciot34  CIC-IoT-2023 with all 34 sub-attack labels (5,000 per class)
#   unswnb15  UNSW-NB15, 10 attack categories (10,000 per class)
# Same protocol and configurations as round 2 (no scaling, 5 seeds), plus two configurations on UNSW-NB15
# in which the tools choose the model (bigfeat_auto, tpot_auto).
# CIC-IoT-2023 (34 classes) did not fit in 14 GB of RAM and is incomplete; it is not used in the paper.
# Resumable: finished runs are skipped.
#
# Start (keeps the laptop awake while it runs):
#   nohup systemd-inhibit --what=sleep:idle --why="revision grid" \
#     bash experiments/run_round3.sh > results/round3/logs/run_all.log 2>&1 &
# Stop:  pkill -f run_round3.sh; pkill -f run_grid.py
set -u
cd "$(dirname "$0")/.."
export REVISION_OUT="${REVISION_OUT:-$PWD/results/round3}"
mkdir -p "$REVISION_OUT/logs"
PY="nice -n 10 python3 -W ignore experiments/run_grid.py"
SEEDS=0-4
CHEAP="original split10 lgbm"
BIGFEAT_CONFIGS="bigfeat bigfeat_raw both both_raw"
# CIC-IoT-2023 (34 classes) forests are several GB each; one worker keeps memory in bounds.
W_UNSW="${W_UNSW:-2}"
W_CIC="${W_CIC:-1}"

echo "== phase 1: UNSW-NB15, cheap configs"; date
$PY --dataset unswnb15 --units all --seeds $SEEDS --configs $CHEAP --workers $W_UNSW
echo "== phase 2: CIC-IoT-2023 34 classes, cheap configs"; date
$PY --dataset ciciot34 --units all --seeds $SEEDS --configs $CHEAP --workers $W_CIC
echo "== phase 3: TPOT"; date
$PY --dataset unswnb15 --units all --seeds $SEEDS --configs tpot --workers $W_UNSW
$PY --dataset ciciot34 --units all --seeds $SEEDS --configs tpot --workers $W_CIC
echo "== phase 4: BigFeat ($BIGFEAT_CONFIGS)"; date
$PY --dataset unswnb15 --units all --seeds $SEEDS --configs $BIGFEAT_CONFIGS --workers $W_UNSW
$PY --dataset ciciot34 --units all --seeds $SEEDS --configs $BIGFEAT_CONFIGS --workers $W_CIC
echo "== phase 5: model chosen by BigFeat / TPOT (UNSW-NB15)"; date
$PY --dataset unswnb15 --units all --seeds $SEEDS --configs bigfeat_auto tpot_auto --workers $W_UNSW
echo "== phase 6: ExtraTrees with min_samples_split chosen by a CV grid (UNSW-NB15)"; date
$PY --dataset unswnb15 --units all --seeds $SEEDS --configs split_grid --workers $W_UNSW
echo "== analysis"; date
python3 -W ignore experiments/analyze.py > "$REVISION_OUT/logs/analyze.log" 2>&1
echo "== done"; date
