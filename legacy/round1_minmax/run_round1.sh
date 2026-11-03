#!/usr/bin/env bash
# Round 1 (kept for the record). It min-max scaled every configuration; experiments/protocol.py no
# longer does (NEEDS_SCALING is empty), so re-running this script does not reproduce these results.
# Full revision grid, resumable (finished runs are skipped). Runs at low CPU priority.
#   nohup bash experiments/run_round1.sh > legacy/round1_minmax/logs/run_all.log 2>&1 &
# Stop:   pkill -f run_round1.sh; pkill -f run_grid.py
set -u
cd "$(dirname "$0")/../.."
DATA=${REVISION_DATA:-/mnt/Projects/datasets}
PY="nice -n 10 python3 -W ignore experiments/run_grid.py"
CHEAP_SEEDS=0-4        # original, bigfeat
TPOT_SEEDS=0-2         # tpot, both (each TPOT run is ~30 min on N-BaIoT, ~2 h on Edge-IIoTset)

wait_nbaiot() {
  until [ -f "$DATA/nbaiot/nbaiot.zip" ]; do sleep 60; done
  if [ ! -f "$DATA/nbaiot/extracted/.done" ]; then
    mkdir -p "$DATA/nbaiot/extracted"
    unzip -oq "$DATA/nbaiot/nbaiot.zip" -d "$DATA/nbaiot/extracted"
    python3 experiments/extract_rars.py "$DATA/nbaiot/extracted"
    touch "$DATA/nbaiot/extracted/.done"
  fi
}
wait_cic() {
  local want; want=$(wc -l < "$DATA/ciciot2023/selected.txt")
  until [ "$(find "$DATA/ciciot2023/CSV" -name '*.csv' | wc -l)" -ge "$want" ] && \
        [ -z "$(find "$DATA/ciciot2023/CSV" -name '*.part')" ]; do sleep 60; done
}

echo "== phase 1: N-BaIoT in-fold top-20, cheap configs (primary)"; date
wait_nbaiot
$PY --dataset nbaiot --feature-mode infold20 --units 1-9 --seeds $CHEAP_SEEDS --configs original bigfeat --workers 2
echo "== phase 2: CIC-IoT-2023 cheap configs"; date
wait_cic
$PY --dataset ciciot2023 --units all --seeds $CHEAP_SEEDS --configs original bigfeat --workers 2
echo "== phase 3: Edge-IIoTset cheap configs"; date
$PY --dataset edgeiiot --units all --seeds $CHEAP_SEEDS --configs original bigfeat --workers 1
echo "== phase 4: N-BaIoT in-fold top-20, TPOT configs"; date
$PY --dataset nbaiot --feature-mode infold20 --units 1-9 --seeds $TPOT_SEEDS --configs tpot both --workers 2
echo "== phase 5: CIC-IoT-2023 TPOT configs"; date
$PY --dataset ciciot2023 --units all --seeds $TPOT_SEEDS --configs tpot both --workers 2
echo "== phase 6: Edge-IIoTset TPOT configs"; date
$PY --dataset edgeiiot --units all --seeds $TPOT_SEEDS --configs tpot both --workers 1
echo "== phase 7: N-BaIoT pre-built 20 features, cheap configs (continuity with the previous version)"; date
$PY --dataset nbaiot --feature-mode prebuilt20 --units 1-9 --seeds $CHEAP_SEEDS --configs original bigfeat --workers 2
echo "== phase 8: N-BaIoT all 115 features, cheap configs (sensitivity)"; date
$PY --dataset nbaiot --feature-mode all --units 1-9 --seeds 0-2 --configs original bigfeat --workers 2
echo "== phase 9: N-BaIoT pre-built 20, TPOT configs (continuity)"; date
$PY --dataset nbaiot --feature-mode prebuilt20 --units 1-9 --seeds $TPOT_SEEDS --configs tpot both --workers 2
echo "== analysis"; date
python3 -W ignore experiments/analyze.py > legacy/round1_minmax/logs/analyze.log 2>&1
echo "== done"; date
