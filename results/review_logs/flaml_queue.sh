#!/usr/bin/env bash
# Additional runs (2026-11-05): LightGBM with max_delta_step chosen by 3-fold CV (lgbm_cv), and FLAML
# with a 300 s budget (flaml), on the two datasets with headroom. Resumable: finished runs are skipped.
set -u
cd /home/victoryst/personal/dev/XAI_IoT_with_BigFeat
PY="nice -n 10 python3 -W ignore experiments/run_grid.py"
echo "== lgbm_cv"; date
REVISION_OUT=$PWD/results/round2 $PY --dataset edgeiiot --units all --seeds 0-4 --configs lgbm_cv --workers 2
REVISION_OUT=$PWD/results/round3 $PY --dataset unswnb15 --units all --seeds 0-4 --configs lgbm_cv --workers 2
echo "== flaml"; date
REVISION_OUT=$PWD/results/round2 $PY --dataset edgeiiot --units all --seeds 0-4 --configs flaml --workers 2
REVISION_OUT=$PWD/results/round3 $PY --dataset unswnb15 --units all --seeds 0-4 --configs flaml --workers 2
echo "== done"; date
