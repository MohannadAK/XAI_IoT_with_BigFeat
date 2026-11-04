#!/usr/bin/env bash
# Additional runs: LightGBM reruns (faithfulness in margin space), ExtraTrees with a CV grid over
# min_samples_split, and model selection by BigFeat / TPOT on Edge-IIoTset. Resumable.
set -u
cd /home/victoryst/personal/dev/XAI_IoT_with_BigFeat
PY="nice -n 10 python3 -W ignore experiments/run_grid.py"
echo "== A: upper bounds"; date
python3 -W ignore experiments/ceiling.py > results/ceiling.log 2>&1
echo "== B: LightGBM-family reruns (faithfulness in margin space)"; date
for f in results/round2/nbaiot/*/unit*/seed*/lgbm*.{json,npz} results/round2/edgeiiot/all/unitall/seed*/lgbm*.{json,npz} \
         results/round2/ciciot2023/all/unitall/seed*/lgbm*.{json,npz} results/round3/unswnb15/all/unitall/seed*/lgbm*.{json,npz} \
         results/round3/unswnb15/all/unitall/seed*/bigfeat_auto*.{json,npz}; do rm -f "$f"; done
REVISION_OUT=$PWD/results/round2 $PY --dataset edgeiiot --units all --seeds 0-4 --configs lgbm --workers 2
REVISION_OUT=$PWD/results/round3 $PY --dataset unswnb15 --units all --seeds 0-4 --configs lgbm --workers 2
REVISION_OUT=$PWD/results/round2 $PY --dataset nbaiot --feature-mode infold20 --units 1-9 --seeds 0-4 --configs lgbm --workers 2
REVISION_OUT=$PWD/results/round2 $PY --dataset nbaiot --feature-mode all --units 1-9 --seeds 0-4 --configs lgbm --workers 2
REVISION_OUT=$PWD/results/round2 $PY --dataset ciciot2023 --units all --seeds 0-4 --configs lgbm --workers 1
echo "== C: ExtraTrees with min_samples_split chosen by a small CV grid"; date
REVISION_OUT=$PWD/results/round2 $PY --dataset edgeiiot --units all --seeds 0-4 --configs split_grid --workers 2
REVISION_OUT=$PWD/results/round3 $PY --dataset unswnb15 --units all --seeds 0-4 --configs split_grid --workers 2
REVISION_OUT=$PWD/results/round2 $PY --dataset nbaiot --feature-mode infold20 --units 1-9 --seeds 0-4 --configs split_grid --workers 2
echo "== D: model chosen by BigFeat / TPOT on Edge-IIoTset"; date
REVISION_OUT=$PWD/results/round3 $PY --dataset unswnb15 --units all --seeds 0-4 --configs bigfeat_auto --workers 2
REVISION_OUT=$PWD/results/round2 $PY --dataset edgeiiot --units all --seeds 0-4 --configs bigfeat_auto --workers 2
REVISION_OUT=$PWD/results/round2 $PY --dataset edgeiiot --units all --seeds 0-4 --configs tpot_auto --workers 2
echo "== done"; date
