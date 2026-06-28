#!/usr/bin/env python3
"""Attribute the local (non-model) wall RESIDUAL to named phases — L4 (0062.0006-T).

The 0061 diagnosis (perf/stage_breakdown.py) split a cycle's wall into:
  modelpath  — Σ over MODEL stages of (max latency if the stage fans out, else Σ)
  residual   — wall − modelpath: local inter-stage work that hit no model and so
               carried no ledger row → ~31.7% median, un-attributable.

L4 lays down instrumentation: hive/ledger.py ``timed_local`` now emits a
provider='local' row for each named local phase (decompose_local / parse /
conflict_scan / retrieve / db_read …). This tool reads those rows and reports, per
run and in aggregate, how much of the residual the instrumented phases now EXPLAIN
and how much remains unexplained (the next caching/parallelisation target).

  modelpath          critical path over provider != 'local' rows
  local_attributed   critical-path contribution of provider == 'local' rows
                     (max for parallel stages, Σ for serial stages)
  residual           wall − modelpath
  unexplained        residual after the instrumented local rows have joined modelpath

CLI:
    python perf/residual_breakdown.py                 # default db, wall>=0
    python perf/residual_breakdown.py --db <path> --min-wall 30
    python perf/residual_breakdown.py --json out.json
"""
from __future__ import annotations
import argparse, json, os, sqlite3, statistics, sys

os.environ.setdefault("PYTHONUTF8", "1")
for _s in (sys.stdout, sys.stderr):
    rc = getattr(_s, "reconfigure", None)
    if rc:
        rc(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Model stages that fan out concurrently (stage wall = max, not Σ). Mirrors
# stage_breakdown.PARALLEL_STAGES.
PARALLEL_STAGES = frozenset({
    "fanout", "swarm", "judge", "scout", "lens", "retrieve",
    "judge_prep", "judge_followup",
})


def _modelpath(stage_rows: dict[str, list[float]]) -> float:
    """Critical-path model of wall over ALL stages (stage_breakdown convention):
    a fan-out stage contributes max(latency); a serial stage contributes Σ. Local
    serial phases (parse / conflict_scan / decompose_local / db_read) are serial and
    therefore folded in here — which is exactly why instrumenting them SHRINKS the
    residual (they leave the unattributed bucket and join the accounted critical path)."""
    total = 0.0
    for stage, lats in stage_rows.items():
        pos = [x for x in lats if x and x > 0]
        if not pos:
            continue
        total += max(pos) if stage in PARALLEL_STAGES else sum(pos)
    return total


def _critical_path(stage_rows: dict[str, list[float]]) -> float:
    """Critical-path contribution for an arbitrary set of rows."""
    return _modelpath(stage_rows)


def breakdown(db: str, min_wall: float) -> dict:
    con = sqlite3.connect(db)
    runs = con.execute(
        "SELECT id, elapsed_s FROM runs WHERE elapsed_s IS NOT NULL AND elapsed_s>=?",
        (min_wall,)).fetchall()
    per_run = []
    local_by_stage: dict[str, list[float]] = {}
    for run_id, wall in runs:
        rows = con.execute(
            "SELECT stage, provider, latency_s FROM worker_calls WHERE run_id=?",
            (run_id,)).fetchall()
        stages: dict[str, list[float]] = {}        # ALL stages (model + local)
        local_stages: dict[str, list[float]] = {}  # provider='local' rows only
        local_stage_wall: dict[str, float] = {}
        serial_local: dict[str, float] = {}        # compatibility/detail field
        for stage, provider, lat in rows:
            if lat is None:
                continue
            stages.setdefault(stage, []).append(lat or 0.0)
            if (provider or "") == "local":
                local_stages.setdefault(stage, []).append(lat or 0.0)
                local_by_stage.setdefault(stage, []).append(lat or 0.0)
                if stage not in PARALLEL_STAGES:
                    serial_local[stage] = serial_local.get(stage, 0.0) + (lat or 0.0)
        mp = _modelpath(stages)                    # serial local already folded in
        sl = sum(serial_local.values())
        for stage, lats in local_stages.items():
            pos = [x for x in lats if x and x > 0]
            if pos:
                local_stage_wall[stage] = max(pos) if stage in PARALLEL_STAGES else sum(pos)
        local_cp = _critical_path(local_stages)
        residual_now = wall - mp                   # what is STILL unattributed
        residual_before = residual_now + local_cp  # residual before L4 instrumentation
        per_run.append({
            "run_id": run_id, "wall_s": round(wall, 1), "modelpath_s": round(mp, 1),
            "residual_now_s": round(residual_now, 1),
            "residual_before_instr_s": round(residual_before, 1),
            "serial_local_attributed_s": round(sl, 1),
            "local_critical_attributed_s": round(local_cp, 1),
            "residual_explained_pct": (round(local_cp / residual_before * 100, 1)
                                       if residual_before > 0 else None),
            "local_critical_by_stage": {k: round(v, 1)
                                        for k, v in sorted(local_stage_wall.items())},
            "serial_local_by_stage": {k: round(v, 1) for k, v in sorted(serial_local.items())},
        })
    con.close()

    walls = [r["wall_s"] for r in per_run]
    res = [r["residual_before_instr_s"] for r in per_run if r["residual_before_instr_s"] > 0]
    expl = [r["residual_explained_pct"] for r in per_run
            if r["residual_explained_pct"] is not None]
    agg_local = {s: {"n": len(v), "median_s": round(statistics.median(v), 2),
                     "total_s": round(sum(v), 1)}
                 for s, v in sorted(local_by_stage.items())}
    return {
        "db": db, "min_wall": min_wall, "runs": len(per_run),
        "median_wall_s": round(statistics.median(walls), 1) if walls else None,
        "median_residual_before_instr_s": round(statistics.median(res), 1) if res else None,
        "median_residual_explained_pct": round(statistics.median(expl), 1) if expl else None,
        "serial_local_stage_aggregate": agg_local,
        "per_run": per_run,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=os.path.join(REPO, "hive_ledger.db"))
    ap.add_argument("--min-wall", type=float, default=0.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = breakdown(args.db, args.min_wall)
    print(f"db={res['db']}  runs(wall>={args.min_wall})={res['runs']}")
    print(f"median wall={res['median_wall_s']}s "
          f"residual_before_instr={res['median_residual_before_instr_s']}s "
          f"residual_explained_by_L4={res['median_residual_explained_pct']}%")
    print("\ninstrumented local phases (provider='local'; aggregate row seconds):")
    for s, a in res["serial_local_stage_aggregate"].items():
        print(f"  {s:18} n={a['n']:5d} median={a['median_s']:7.2f}s total={a['total_s']:8.1f}s")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
