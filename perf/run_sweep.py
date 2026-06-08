#!/usr/bin/env python3
"""Unattended T901/TR901 model-performance sweep driver.

For every matrix cell, REPEATS times, straight through and serial (req #4):

  1. RESET the target branch to its clean baseline_sha (guarded — see _reset_branch).
  2. RUN the pipeline with the cell's --profile:
       investigate cells -> `hive.py investigate`             (queen/judge/converge/scout)
       specify/review    -> `hive.py investigate --specify`
       run cells         -> `hive.py run --specify`           (swarm/assemble; scored on edit-spec)
  3. SCORE the stage artifact against the golden loci (perf/score.py), read-only.
  4. Pull this run's cost from the cell's isolated ledger.db.
  5. RESET again, move to the next repeat/cell.

Results land in perf/results/<cell>/<rep>/ + a roll-up perf/results/summary.json|md.

MODES
  python perf/run_sweep.py --smoke        # connectivity only: all roles gpt-oss-20b, 1 investigate, NO write
  python perf/run_sweep.py --dry-run      # print the full plan (commands), touch nothing
  python perf/run_sweep.py --only queen-base,judge-up   # subset of cells
  python perf/run_sweep.py                # the real sweep (PAID — needs approval)

SAFETY
  - Every git reset is guarded: the codebase path must be a git work tree AND contain the
    matrix's branch_leaf, or the driver aborts. It never touches any repo but the target.
  - --dry-run / --smoke make no destructive change to the target tree.
  - The real sweep makes PAID model calls and resets the branch working tree between runs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

# Force our own stdout/stderr to UTF-8 so Korean seed paths / em-dashes don't crash
# on the Windows cp932 console (mirrors hive.py::_force_utf8_io).
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    _rc = getattr(_stream, "reconfigure", None)
    if _rc is not None:
        _rc(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HIVE = os.path.join(REPO, "hive.py")
MATRIX_PATH = os.path.join(HERE, "matrix.json")
GOLDEN = os.path.join(HERE, "golden", "manifest.json")
RESULTS = os.path.join(HERE, "results")

sys.path.insert(0, HERE)
import score as scorer  # noqa: E402  (sibling module)


def load_matrix() -> dict:
    with open(MATRIX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _git(codebase: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", codebase, *args],
                          capture_output=True, text=True)


def _assert_target(codebase: str, branch_leaf: str) -> None:
    """Refuse to touch anything that is not the expected target branch work tree."""
    if not os.path.isdir(codebase):
        raise SystemExit(f"ABORT: codebase not found: {codebase}")
    inside = _git(codebase, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        raise SystemExit(f"ABORT: not a git work tree: {codebase}")
    if branch_leaf not in codebase.replace("\\", "/"):
        raise SystemExit(
            f"ABORT: codebase {codebase!r} does not contain expected branch leaf "
            f"{branch_leaf!r} — refusing to reset an unexpected repo.")


def _reset_branch(codebase: str, sha: str, branch_leaf: str) -> None:
    """Hard-reset the target working tree to the clean baseline (guarded, destructive)."""
    _assert_target(codebase, branch_leaf)
    _git(codebase, "reset", "--hard", sha)
    # Remove untracked artifacts a prior run may have created (e.g. live RED probes).
    # -d dirs, -x also ignored files. Scoped to the work tree.
    _git(codebase, "clean", "-fdx")


def _run(cmd: list[str], log_path: str, env: dict | None = None,
         timeout: int = 3600) -> dict:
    """Run a subprocess, tee combined output to log_path, return {rc, elapsed}."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        try:
            proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                  text=True, env=env, timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            log.write("\n[TIMEOUT]\n")
            rc = -9
    return {"rc": rc, "elapsed": round(time.time() - t0, 1)}


def _ledger_cost(db_path: str) -> dict:
    """Roll up this cell-run's worker calls from its isolated ledger DB (best-effort)."""
    import sqlite3
    out = {"calls": 0, "model_calls": 0, "in_chars": 0, "out_chars": 0,
           "est_tokens": 0, "real_tokens": 0, "by_stage": {}}
    if not os.path.exists(db_path):
        return out
    try:
        con = sqlite3.connect(db_path)
        rows = con.execute(
            "SELECT stage, provider, model, in_chars, out_chars, est_tokens,"
            " real_tokens FROM worker_calls").fetchall()
        con.close()
    except Exception:  # noqa: BLE001
        return out
    for stage, provider, model, inc, outc, est, real in rows:
        out["calls"] += 1
        if provider != "local":
            out["model_calls"] += 1
        out["in_chars"] += inc or 0
        out["out_chars"] += outc or 0
        out["est_tokens"] += est or 0
        out["real_tokens"] += real or 0
        s = out["by_stage"].setdefault(stage or "?", {"calls": 0, "est_tokens": 0})
        s["calls"] += 1
        s["est_tokens"] += est or 0
    return out


def _spec_path(out_path: str) -> str:
    return os.path.splitext(out_path)[0] + ".edit_spec.json"


def _frozen_honey(m: dict) -> str:
    """Resolve the FROZEN baseline honey that specify/review cells re-author from.

    Module-only replay needs ONE fixed upstream honey shared by every specify/review
    cell (down/base/up) so the swept variable is the stage's OWN model and nothing
    upstream — that is the OFAT intent. Prefer an explicit matrix ``frozen_honey``;
    else fall back to a baseline honey already on disk (a specify-base rep — that cell
    IS the all-baseline config, so its honey is a valid baseline upstream).
    """
    explicit = m.get("frozen_honey")
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(HERE, explicit)
    for rep in ("2", "1", "3"):
        cand = os.path.join(RESULTS, "specify-base", rep, "verdict.honey.md")
        if os.path.exists(cand):
            return cand
    raise SystemExit(
        "ABORT: no frozen baseline honey for specify/review replay — set matrix "
        "'frozen_honey' or run a specify-base cell first.")


def cell_commands(m: dict, cell: dict, rep_dir: str) -> dict:
    """Build the pipeline command for one cell run."""
    tgt = m["target"]
    codebase = tgt["codebase"]
    profile = f"perf-{cell['id']}"
    common = [sys.executable, HIVE, "--profile", profile]
    if cell["path"] == "investigate":
        if cell["stage"] in {"specify", "review"}:
            # Module-only replay (the OFAT intent): swap ONLY the specify/review model
            # via --profile and re-author off a FROZEN baseline honey, instead of
            # re-running decompose→judge→converge for every cell. The seam is hive.py
            # `specify --honey`; freezing one baseline honey holds the upstream fixed
            # across down/base/up so the comparison isolates this stage's own model.
            out = os.path.join(rep_dir, "verdict.edit_spec.json")
            pipeline = common + [
                "specify", "--honey", _frozen_honey(m), "--codebase", codebase,
                "--out", out]
        else:  # queen / judge / converge / scout → verdict.json from a fresh investigate
            out = os.path.join(rep_dir, "verdict.json")
            pipeline = common + [
                "investigate", "--seed", m["seed"], "--codebase", codebase,
                "--docs", m["docs"], "--out", out]
    else:  # run / swarm
        out = os.path.join(rep_dir, "honey.md")
        # --specify on the run path too: honey grep alone can't score swarm/assemble
        # because assemble embeds the seed verbatim and the seed NAMES the golden files,
        # so every honey "contains" them (recall floored at 1.0). The edit-spec authored
        # loci ARE model-sensitive (specify won't author what it can't ground), so we
        # score honey.edit_spec.json instead (score.py prefers it when present).
        pipeline = common + [
            "run", "--seed", m["seed"], "--recipe", os.path.join(REPO, m["recipe"]),
            "--codebase", codebase, "--out", out, "--specify"]
    return {"pipeline": pipeline, "out": out, "codebase": codebase}


def run_cell(m: dict, cell: dict, repeats: int, dry: bool, golden: dict) -> list[dict]:
    tgt = m["target"]
    codebase, leaf, sha = tgt["codebase"], tgt["branch_leaf"], tgt["baseline_sha"]
    ledger_db = os.path.join(RESULTS, cell["id"], "ledger.db")
    runs = []
    for rep in range(1, repeats + 1):
        rep_dir = os.path.join(RESULTS, cell["id"], str(rep))
        cmds = cell_commands(m, cell, rep_dir)
        if dry:
            print(f"\n--- {cell['id']} rep {rep} ({cell['path']}) ---")
            print("  pipeline:", " ".join(cmds["pipeline"]))
            continue

        os.makedirs(rep_dir, exist_ok=True)
        # fresh ledger per cell so cost is the cell's only
        if rep == 1 and os.path.exists(ledger_db):
            try:
                os.remove(ledger_db)
            except OSError:
                pass

        _reset_branch(codebase, sha, leaf)
        pipe = _run(cmds["pipeline"], os.path.join(rep_dir, "pipeline.log"))

        result = {"cell": cell["id"], "stage": cell["stage"], "level": cell["level"],
                  "provider": cell["provider"], "model": cell["model"],
                  "path": cell["path"], "rep": rep, "ts": datetime.now(timezone.utc).isoformat(),
                  "pipeline_rc": pipe["rc"], "pipeline_s": pipe["elapsed"]}

        score = scorer.score_cell(cell["stage"], rep_dir, golden)
        result["verdict"] = score
        result["score"] = score

        result["cost"] = _ledger_cost(ledger_db)
        with open(os.path.join(rep_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        runs.append(result)
        _reset_branch(codebase, sha, leaf)
        measurement = score.get("measurement") or "n/a"
        core = score.get("core_recall", "n/a")
        print(f"[{cell['id']} rep{rep}] measurement={measurement} core_recall={core} "
              f"pipeline_rc={result['pipeline_rc']} "
              f"calls={result['cost']['model_calls']}")
    return runs


def run_smoke(m: dict) -> None:
    """Connectivity probe (req #5): all roles gpt-oss-20b, one investigate, NO write."""
    tgt = m["target"]
    rep_dir = os.path.join(RESULTS, "_smoke")
    os.makedirs(rep_dir, exist_ok=True)
    out = os.path.join(rep_dir, "verdict.json")
    cmd = [sys.executable, HIVE, "--profile", "perf-smoke", "investigate",
           "--seed", m["seed"], "--codebase", tgt["codebase"], "--docs", m["docs"],
           "--out", out, "--specify"]
    print("SMOKE (openai/gpt-oss-20b, propose-only, no branch reset):")
    print("  ", " ".join(cmd))
    r = _run(cmd, os.path.join(rep_dir, "smoke.log"))
    ok = r["rc"] == 0 and os.path.exists(_spec_path(out))
    print(f"  rc={r['rc']} elapsed={r['elapsed']}s spec_produced={os.path.exists(_spec_path(out))}"
          f"  => {'WIRING OK' if ok else 'CHECK smoke.log'}")


def write_summary(all_runs: list[dict]) -> None:
    def level_bucket(run: dict) -> str | None:
        # ascii levels from matrix.json: base / up / down / down1 / down2.
        # down1/down2 (queen, scout) collapse into the "down" bucket here; the
        # per-cell table below keeps them distinct.
        level = str(run.get("level", ""))
        cid = str(run.get("cell", ""))
        if level == "base" or cid.endswith("-base"):
            return "base"
        if level == "up" or cid.endswith("-up"):
            return "up"
        if level.startswith("down") or "-down" in cid:
            return "down"
        return None

    def avg_core(runs: list[dict]) -> float | None:
        vals = []
        for r in runs:
            sc = r.get("score") or r.get("verdict") or {}
            if sc.get("measurement") == "recall" and sc.get("core_recall") is not None:
                vals.append(float(sc["core_recall"]))
        if not vals:
            return None
        return round(sum(vals) / len(vals), 3)

    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    def down_verdict(down: float | None, base: float | None) -> str:
        if down is None or base is None:
            return "n/a"
        return "↓OK" if down >= base else "↓REGRESS"

    def up_verdict(up: float | None, base: float | None) -> str:
        if up is None or base is None:
            return "n/a"
        return "↑+" if up > base else "↑="

    by_cell: dict[str, list[dict]] = {}
    by_stage: dict[str, list[dict]] = {}
    for r in all_runs:
        by_cell.setdefault(r["cell"], []).append(r)
        by_stage.setdefault(r["stage"], []).append(r)
    summary = {"generated": datetime.now(timezone.utc).isoformat(),
               "stages": {}, "cells": {}}
    lines = ["# T901/TR901 module-unit model-perf sweep — summary", "",
             "| stage | down | base | up | ↓verdict | ↑verdict | reps |",
             "|---|---:|---:|---:|---|---|---|"]
    for stage, runs in by_stage.items():
        buckets = {"down": [], "base": [], "up": []}
        for r in runs:
            bucket = level_bucket(r)
            if bucket in buckets:
                buckets[bucket].append(r)
        down = avg_core(buckets["down"])
        base = avg_core(buckets["base"])
        up = avg_core(buckets["up"])
        reps = "/".join(str(len(buckets[b])) for b in ("down", "base", "up"))
        summary["stages"][stage] = {
            "down_core_recall": down,
            "base_core_recall": base,
            "up_core_recall": up,
            "down_verdict": down_verdict(down, base),
            "up_verdict": up_verdict(up, base),
            "reps": {"down": len(buckets["down"]), "base": len(buckets["base"]),
                     "up": len(buckets["up"])},
        }
        lines.append(f"| {stage} | {fmt(down)} | {fmt(base)} | {fmt(up)} "
                     f"| {down_verdict(down, base)} | {up_verdict(up, base)} | {reps} |")

    lines.extend(["", "## Cells", "",
                  "| cell | stage | level | provider/model | path | measurement | core_recall | full_recall |",
                  "|---|---|---|---|---|---|---:|---:|"])
    for cid, runs in by_cell.items():
        c = runs[0]
        scores = [r.get("score") or r.get("verdict") or {} for r in runs]
        measurements = [s.get("measurement") or "null" for s in scores]
        core = avg_core(runs)
        full_vals = [float(s["full_recall"]) for s in scores
                     if s.get("measurement") == "recall" and s.get("full_recall") is not None]
        full = round(sum(full_vals) / len(full_vals), 3) if full_vals else None
        summary["cells"][cid] = {"stage": c["stage"], "level": c["level"],
                                 "provider": c["provider"], "model": c["model"],
                                 "path": c["path"], "measurements": measurements,
                                 "core_recall": core, "full_recall": full}
        lines.append(f"| {cid} | {c['stage']} | {c['level']} | {c['provider']}/{c['model']} "
                     f"| {c['path']} | {','.join(measurements)} | {fmt(core)} | {fmt(full)} |")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULTS, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nSummary -> perf/results/summary.md")


def main() -> None:
    ap = argparse.ArgumentParser(description="T901 model-performance sweep driver")
    ap.add_argument("--smoke", action="store_true", help="connectivity probe only (gpt-oss-20b, no write)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    ap.add_argument("--only", default=None, help="comma-separated cell ids to run")
    ap.add_argument("--repeats", type=int, default=None, help="override matrix repeats")
    args = ap.parse_args()

    m = load_matrix()
    with open(GOLDEN, "r", encoding="utf-8") as f:
        golden = json.load(f)
    repeats = args.repeats or m.get("repeats", 3)

    # Always (re)generate profiles first so a matrix edit is reflected.
    subprocess.run([sys.executable, os.path.join(HERE, "gen_profiles.py")],
                   cwd=REPO, check=True)

    if args.smoke:
        run_smoke(m)
        return

    cells = m["cells"]
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        cells = [c for c in cells if c["id"] in want]

    if args.dry_run:
        print(f"DRY RUN — {len(cells)} cell(s) x {repeats} repeat(s) = "
              f"{len(cells) * repeats} pipeline runs. Nothing is executed.")
        for c in cells:
            run_cell(m, c, repeats, dry=True, golden=golden)
        return

    print(f"SWEEP — {len(cells)} cell(s) x {repeats} = {len(cells) * repeats} runs. "
          "This makes PAID model calls and resets the target branch between runs.")
    all_runs: list[dict] = []
    for c in cells:
        all_runs.extend(run_cell(m, c, repeats, dry=False, golden=golden))
        write_summary(all_runs)  # checkpoint after each cell
    write_summary(all_runs)


if __name__ == "__main__":
    main()
