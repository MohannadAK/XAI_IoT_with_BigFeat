"""
Run the (task x seed x config) grid with a process pool. Finished runs are skipped, so the
script can be re-launched after an interruption.

Example:
  python experiments/run_grid.py --dataset nbaiot --feature-mode all --units 1-9 \
         --seeds 0-4 --configs original bigfeat tpot both --workers 6
"""
import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("REVISION_OUT", os.path.join(REPO, "results", "round3"))


def _range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def out_dir(dataset, feature_mode, unit, seed):
    return os.path.join(OUT, dataset, feature_mode, f"unit{unit}", f"seed{seed}")


def job(dataset, feature_mode, unit, seed, config):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import protocol, datasets
    d = out_dir(dataset, feature_mode, unit, seed)
    t = time.time()
    try:
        r = protocol.run_one(protocol.Task(dataset, str(unit), feature_mode), seed, config, d, datasets.load)
        return f"OK   {dataset}/{feature_mode}/unit{unit}/seed{seed}/{config}  F1={r['f1_macro']:.5f}  " \
               f"train={r['train_time_s']:.1f}s  total={time.time()-t:.0f}s"
    except Exception:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{config}.error"), "w") as fh:
            fh.write(traceback.format_exc())
        return f"FAIL {dataset}/{feature_mode}/unit{unit}/seed{seed}/{config}\n{traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--feature-mode", default="all")
    ap.add_argument("--units", default="all")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--configs", nargs="+", default=["original", "bigfeat", "tpot", "both"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    units = [a.units] if a.units == "all" else _range(a.units)
    jobs = [(a.dataset, a.feature_mode, u, s, c) for u in units for s in _range(a.seeds) for c in a.configs
            if not os.path.exists(os.path.join(out_dir(a.dataset, a.feature_mode, u, s), f"{c}.json"))]
    # cheap configs first so partial results are usable early
    jobs.sort(key=lambda j: (j[4] in ("tpot", "both", "both_raw", "tpot_auto"), j[4] in ("bigfeat", "bigfeat_raw", "both", "both_raw", "bigfeat_auto"), j[3], str(j[2])))
    print(f"{len(jobs)} jobs to run", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(job, *j) for j in jobs]
        for f in as_completed(futs):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
