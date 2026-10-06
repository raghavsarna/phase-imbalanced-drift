"""Detector comparison on SEA streams with known change points.

Six detectors run on the residuals of the same frozen reference model:
{class-conditional bank, pooled single test} x {Page-Hinkley, DDM, ADWIN}.
Streams: SEA with natural priors and minority shares 20/10/5/2/1% (changes at
t = 10k, 20k, 30k over 50k samples), and drift-free SEA (natural and 5%) for
false alarms.  Each run writes results/detection/<stream>__<detector>__<seed>.json.

    python detection_study.py --jobs 4
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

from data import load
from detector import match_change_points, segment

OUT = Path(os.environ.get("PHASE_RESULTS", Path(__file__).resolve().parent / "results")) / "detection"
STREAMS = ["sea", "sea20", "sea10", "sea5", "sea2", "sea1", "seanull", "seanull5"]
DETECTORS = [(m, t) for t in ("ph", "ddm", "adwin") for m in ("class", "pooled")]


def run(stream, mode, test, seed):
    path = OUT / f"{stream}__{mode}-{test}__{seed}.json"
    if path.exists():
        return
    X, y, meta = load(stream, seed)
    b, a = segment(X, y, meta["n_classes"], mode=mode, test=test, seed=seed)
    cps = meta["change_points"]
    rec = {"stream": stream, "mode": mode, "test": test, "seed": seed, "n": len(y),
           "minority_share": float((y == 0).mean()), "change_points": cps,
           "boundaries": list(map(int, b)), "alarms": list(map(int, a)),
           **match_change_points(cps, b, a)}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    tasks = [(s, m, t, seed) for s in STREAMS for (m, t) in DETECTORS for seed in args.seeds]
    with ProcessPoolExecutor(args.jobs) as pool:
        list(pool.map(run, *zip(*tasks)))
    print(f"{len(tasks)} detection runs in {OUT}")


if __name__ == "__main__":
    main()
