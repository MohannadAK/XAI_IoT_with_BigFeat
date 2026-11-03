#!/usr/bin/env bash
# Waits for the running v2 driver (PID given as $1) to finish, then re-runs the driver with both
# BigFeat variants. Finished runs are skipped, so only the BigFeat phase and the analysis run.
cd "$(dirname "$0")/../../.."
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "== queued BigFeat phase starting"; date
BIGFEAT_CONFIGS="bigfeat bigfeat_raw both both_raw" \
  systemd-inhibit --what=sleep:idle --why="revision grid v2 BigFeat" bash experiments/run_round2.sh
