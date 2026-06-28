#!/usr/bin/env python3
"""Stage-level wall-clock breakdown for hive investigate/run cycles.

Operationalises the hivework.0061 R0001 (성능개선) diagnosis: a cycle's wall is
NOT bounded by its single slowest worker — it is the serial sum of stage critical
paths plus the local (non-worker) orchestration residual. This tool makes that
reproducible from the ledger so a perf change can be measured before/after.

Inputs (read-only) come straight from ``hive_ledger.db``:
  - ``runs.elapsed_s``        actual cycle wall-clock (the ground truth)
  - ``worker_calls.stage``    which pipeline stage a model call belongs to
  - ``worker_calls.latency_s`` per-call wall (INCLUDES copilot subprocess spawn,
                              since providers.py wraps ``time.monotonic()`` around
                              the whole subprocess — so spawn is already in here)

Three derived numbers per run:
  - ``max_lat``    the single slowest worker call  → perfect-parallel lower bound
  - ``modelpath``  Σ over stages of (max(latency) if the stage fans out
                   concurrently, else Σ(latency)) → the critical-path MODEL of wall
  - ``residual``   wall − modelpath → local inter-stage work NOT recorded as a
                   worker_call (parse / conflict-scan / reconcile / codemap /
                   retrieval setup / ledger / prompt assembly). Model swaps cannot
                   reclaim this; only structural/caching work can.

``serialization_factor = wall / max_lat`` (1.0 = perfectly parallel; higher = more
serial). The header KPI for R0001 is the median of this across cycles.

CLI:
    python perf/stage_breakdown.py                 # default db, wall>=120s
    python perf/stage_breakdown.py --min-wall 0    # include smoke cycles
    python perf/stage_breakdown.py --json out.json
"""
from __future__ import annotations
import argparse, json, os, sqlite3, statistics, sys

os.environ.setdefault("PYTHONUTF8", "1")
for _s in (sys.stdout, sys.stderr):
    rc = getattr(_s, "reconfigure", None)
    if rc:
        rc(encoding="utf-8", errors="replace")

# Stages whose calls run CONCURRENTLY within the stage (a thread pool fans them
# out), so the stage's wall contribution is max(latency), not the sum. Everything
# else is a single serial model call whose contribution is its own latency.
# Source: fanout.py / investigate.py ThreadPoolExecutor sites.
PARALLEL_STAGES = frozenset({
    "fanout", "swarm", "judge", "scout", "lens", "retrieve",
    "judge_prep", "judge_followup",
})


def compute_run_breakdown(stage_latencies: dict[str, list[float]], wall: float) -> dict:
    """Pure: per-run breakdown from {stage: [latency_s, ...]} and the wall.

    Stages with no positive latency are ignored. Returns None-safe numbers; the
    caller decides whether ``wall`` and ``max_lat`` are usable (e.g. > 0).
    """
    by = {s: [x for x in lats if x and x > 0] for s, lats in stage_latencies.items()}
    by = {s: lats for s, lats in by.items() if lats}
    if not by:
        return {"wall": wall, "sum_lat": 0.0, "max_lat": 0.0, "modelpath": 0.0,
                "residual": wall, "serialization_factor": None, "n_calls": 0}
    sum_lat = sum(sum(v) for v in by.values())
    max_lat = max(max(v) for v in by.values())
    modelpath = sum((max(v) if s in PARALLEL_STAGES else sum(v)) for s, v in by.items())
    n_calls = sum(len(v) for v in by.values())
    return {
        "wall": wall,
        "sum_lat": sum_lat,
        "max_lat": max_lat,
        "modelpath": modelpath,
        "residual": wall - modelpath,
        "serialization_factor": (wall / max_lat) if max_lat > 0 else None,
        "n_calls": n_calls,
    }


def stage_shares(per_run_stage_latencies: list[dict[str, list[float]]]) -> list[dict]:
    """Pure: aggregate stage share of total worker-seconds, descending.

    Each element of the input is one run's {stage: [latency_s, ...]}.
    """
    agg: dict[str, float] = {}
    cnt: dict[str, int] = {}
    for run in per_run_stage_latencies:
        for stage, lats in run.items():
            for x in lats:
                if x and x > 0:
                    agg[stage] = agg.get(stage, 0.0) + x
                    cnt[stage] = cnt.get(stage, 0) + 1
    total = sum(agg.values()) or 1.0
    rows = []
    for stage, secs in sorted(agg.items(), key=lambda kv: -kv[1]):
        rows.append({"stage": stage, "sum_s": secs, "share": secs / total,
                     "calls": cnt[stage], "avg_s": secs / cnt[stage]})
    return rows


def _fetch(db_path: str, min_wall: float):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    wall = {r["id"]: r["elapsed_s"]
            for r in cur.execute("select id, elapsed_s from runs where elapsed_s is not null")}
    runs: dict[int, dict[str, list[float]]] = {}
    for r in cur.execute("select run_id, stage, latency_s from worker_calls where latency_s is not null"):
        rid = r["run_id"]
        if wall.get(rid) is None or wall[rid] < min_wall:
            continue
        runs.setdefault(rid, {}).setdefault(r["stage"], []).append(r["latency_s"])
    con.close()
    return wall, runs


def analyze(db_path: str, min_wall: float = 120.0) -> dict:
    wall, runs = _fetch(db_path, min_wall)
    per_run = []
    factors = []
    fit_errs = []
    for rid, sl in sorted(runs.items()):
        w = wall[rid]
        b = compute_run_breakdown(sl, w)
        b["run_id"] = rid
        per_run.append(b)
        if b["serialization_factor"] is not None:
            factors.append(b["serialization_factor"])
            if w:
                fit_errs.append(abs(b["modelpath"] - w) / w)
    shares = stage_shares(list(runs.values()))
    return {
        "db": db_path,
        "min_wall": min_wall,
        "cycles": len(per_run),
        "median_serialization_factor": statistics.median(factors) if factors else None,
        "median_modelpath_fit_err": statistics.median(fit_errs) if fit_errs else None,
        "median_wall_s": statistics.median([b["wall"] for b in per_run]) if per_run else None,
        "stage_shares": shares,
        "per_run": per_run,
    }


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.join(repo, "hive_ledger.db"))
    ap.add_argument("--min-wall", type=float, default=120.0,
                    help="only analyze cycles with elapsed_s >= this (drop smoke runs)")
    ap.add_argument("--json", default=None, help="write the full result as JSON here")
    a = ap.parse_args()
    res = analyze(a.db, a.min_wall)

    print(f"cycles analyzed: {res['cycles']} (wall>={a.min_wall:g}s)")
    if res["median_wall_s"] is not None:
        print(f"median wall_s: {res['median_wall_s']:.1f}")
    if res["median_serialization_factor"] is not None:
        print(f"median serialization_factor (wall/max_lat): {res['median_serialization_factor']:.2f}x")
    if res["median_modelpath_fit_err"] is not None:
        print(f"median residual (1-modelpath/wall): {res['median_modelpath_fit_err']*100:.1f}%")
    print("\nstage share of worker-seconds:")
    print(f"  {'stage':<12}{'sum_s':>9}{'share':>7}{'calls':>7}{'avg_s':>8}")
    for r in res["stage_shares"]:
        print(f"  {r['stage']:<12}{r['sum_s']:>9.0f}{r['share']*100:>6.1f}%{r['calls']:>7}{r['avg_s']:>8.1f}")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
